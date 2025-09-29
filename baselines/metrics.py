import os.path as osp
import pathlib

import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

import pyvista as pv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

import metrics_NACA
from reorganize import reorganize
from dataset import Dataset

from tqdm import tqdm

NU = np.array(1.56e-5)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def rsquared(predict, true):
    '''
    Args:
        predict (tensor): Predicted values, shape (N, *)
        true (tensor): True values, shape (N, *)

    Out:
        rsquared (tensor): Coefficient of determination of the prediction, shape (*,)
    '''
    mean = true.mean(dim = 0)
    return 1 - ((true - predict)**2).sum(dim = 0)/((true - mean)**2).sum(dim = 0)

def rel_err(a, b):
    return np.abs((a - b)/a)

def WallShearStress(Jacob_U, normals):
    S = .5*(Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1 = 1, axis2 = 2).reshape(-1, 1, 1)*np.eye(2)[None]/3
    ShearStress = 2*NU.reshape(-1, 1, 1)*S
    ShearStress = (ShearStress*normals[:, :2].reshape(-1, 1, 2)).sum(axis = 2)

    return ShearStress

@torch.no_grad()
def Infer_test(device, models, hparams, data, coef_norm = None):
    # Inference procedure on new simulation
    outs = [torch.zeros_like(data.y)]*len(models)
    n_out = torch.zeros_like(data.y[:, :1])
    idx_points = set(map(tuple, data.pos[:, :2].numpy()))
    cond = True
    i = 0
    while cond: 
        i += 1       
        data_sampled = data.clone()
        idx = random.sample(range(data_sampled.x.size(0)),
                    min(hparams[0]['subsampling'], data_sampled.x.size(0)))          
        idx = torch.tensor(idx)
        idx_points = idx_points - set(map(tuple, data_sampled.pos[idx, :2].numpy()))
        data_sampled.pos = data_sampled.pos[idx]
        data_sampled.x = data_sampled.x[idx]
        data_sampled.y = data_sampled.y[idx]
        data_sampled.surf = data_sampled.surf[idx]
        data_sampled.batch = data_sampled.batch[idx]

        out = [torch.zeros_like(data.y)]*len(models)
        tim = np.zeros(len(models))
        for n, model in enumerate(models):
            try:
                data_sampled.edge_index = nng.radius_graph(
                    x = data_sampled.pos.to(device),
                    r = hparams[n]['r'], loop = True,
                    max_num_neighbors = int(hparams[n]['max_neighbors'])
                ).cpu()
            except KeyError:
                data_sampled.edge_index = None

            model.eval()
            data_sampled = data_sampled.to(device)
            start = time.time()
            o = model(data_sampled)
            tim[n] += time.time() - start
            out[n][idx] = o.cpu()
            outs[n] = outs[n] + out[n]

        n_out[idx] = n_out[idx] + torch.ones_like(n_out[idx])
        cond = (len(idx_points) > 0)

    for n, out in enumerate(outs):
        outs[n] = out / n_out  # moyenne

        # Détection du nb de sorties
        out_dim = outs[n].shape[1]

        # Si single-output (pression seule), ne pas tenter de forcer vx, vy, nut
        if out_dim == 1:
            continue

        # Cas legacy 4 sorties : comportement historique
        if coef_norm is not None:
            outs[n][data.surf, :2] = -torch.tensor(coef_norm[2][None, :2]) * torch.ones_like(out[data.surf, :2]) / (torch.tensor(coef_norm[3][None, :2]) + 1e-8)
            outs[n][data.surf, 3]  = -torch.tensor(coef_norm[2][3]) * torch.ones_like(out[data.surf, 3])  / (torch.tensor(coef_norm[3][3])   + 1e-8)
        else:
            outs[n][data.surf, :2] = torch.zeros_like(out[data.surf, :2])
            outs[n][data.surf, 3]  = torch.zeros_like(out[data.surf, 3])

    return outs, tim/i
def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    # 🔹 Filtrer le VTU pour ne garder que les points de surface si besoin
    if bool_surf.numel() != internal.n_points:
        # le masque est plus court : on suppose que data contient seulement la surface
        # → on fabrique un sous-maillage "surface only" cohérent avec len(bool_surf)
        surf_idx = np.where(bool_surf.cpu().numpy())[0] if bool_surf.dtype==torch.bool else np.arange(len(bool_surf))
        internal = internal.extract_points(surf_idx, include_cells=False)
        # maintenant internal.n_points == len(surf_idx)

    internals = []
    airfoils = []
    for out in outs:
        intern = internal.copy()
        aerofoil = airfoil.copy()

        # Points = uniquement ceux de la surface
        point_mesh = intern.points[:, :2]
        point_surf = aerofoil.points[:, :2]

        out_dim = out.shape[1]
        denorm = (out * (coef_norm[3][:out_dim] + 1e-8) + coef_norm[2][:out_dim]).numpy()

        if out_dim == 1:
            intern.point_data['p'] = denorm[:, 0]
        else:
            intern.point_data['U'][:, :2] = denorm[:, :2]
            intern.point_data['p']        = denorm[:, 2]
            intern.point_data['nut']      = denorm[:, 3]

        # réorganise pression surface → maillage foil
        surf_p = intern.point_data['p']
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)

        internals.append(intern)
        airfoils.append(aerofoil)

    return internals, airfoils

def Airfoil_mean(internals, airfoils):
    # Average multiple prediction over one simulation

    oi_point = np.zeros((internals[0].points.shape[0], 4))
    oi_cell = np.zeros((internals[0].cell_data['U'].shape[0], 4))
    oa_point = np.zeros((airfoils[0].points.shape[0], 4))
    oa_cell = np.zeros((airfoils[0].cell_data['U'].shape[0], 4))

    for k in range(len(internals)):
        oi_point[:, :2] += internals[k].point_data['U'][:, :2]
        oi_point[:, 2] += internals[k].point_data['p']
        oi_point[:, 3] += internals[k].point_data['nut']
        oi_cell[:, :2] += internals[k].cell_data['U'][:, :2]
        oi_cell[:, 2] += internals[k].cell_data['p']
        oi_cell[:, 3] += internals[k].cell_data['nut']

        oa_point[:, :2] += airfoils[k].point_data['U'][:, :2]
        oa_point[:, 2] += airfoils[k].point_data['p']
        oa_point[:, 3] += airfoils[k].point_data['nut']
        oa_cell[:, :2] += airfoils[k].cell_data['U'][:, :2]
        oa_cell[:, 2] += airfoils[k].cell_data['p']
        oa_cell[:, 3] += airfoils[k].cell_data['nut']
    oi_point = oi_point/len(internals)
    oi_cell = oi_cell/len(internals)
    oa_point = oa_point/len(airfoils)
    oa_cell = oa_cell/len(airfoils)
    internal_mean = internals[0].copy()
    internal_mean.point_data['U'][:, :2] = oi_point[:, :2]
    internal_mean.point_data['p'] = oi_point[:, 2]
    internal_mean.point_data['nut'] = oi_point[:, 3]
    internal_mean.cell_data['U'][:, :2] = oi_cell[:, :2]
    internal_mean.cell_data['p'] = oi_cell[:, 2]
    internal_mean.cell_data['nut'] = oi_cell[:, 3]

    airfoil_mean = airfoils[0].copy()
    airfoil_mean.point_data['U'][:, :2] = oa_point[:, :2]
    airfoil_mean.point_data['p'] = oa_point[:, 2]
    airfoil_mean.point_data['nut'] = oa_point[:, 3]
    airfoil_mean.cell_data['U'][:, :2] = oa_cell[:, :2]
    airfoil_mean.cell_data['p'] = oa_cell[:, 2]
    airfoil_mean.cell_data['nut'] = oa_cell[:, 3]

    return internal_mean, airfoil_mean

def Compute_coefficients(internals, airfoils, bool_surf, Uinf, angle, keep_vtk = False):
    # Compute force coefficients, if keet_vtk is True, also return the .vtu/.vtp with wall shear stress added over the airfoil and velocity gradient over the volume.

    coefs = []
    if keep_vtk:
        new_internals = []
        new_airfoils = []
    
    for internal, airfoil in zip(internals, airfoils):
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]

        intern = intern.compute_derivative(scalars = 'U', gradient = 'pred_grad')

        surf_grad = intern.point_data['pred_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_p = intern.point_data['p'][bool_surf]

        surf_grad = reorganize(point_mesh, point_surf, surf_grad)
        surf_p = reorganize(point_mesh, point_surf, surf_p)

        Wss_pred = WallShearStress(surf_grad, -aerofoil.point_data['Normals'])
        aerofoil.point_data['wallShearStress'] = Wss_pred
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data = True) 
        aerofoil = aerofoil.ptc(pass_point_data = True)

        WP_int = -aerofoil.cell_data['p'][:, None]*aerofoil.cell_data['Normals'][:, :2]

        Wss_int = (aerofoil.cell_data['wallShearStress']*aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        WP_int = (WP_int*aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        force = Wss_int - WP_int

        alpha = angle*np.pi/180
        basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        force_rot = basis@force
        coef = 2*force_rot/Uinf**2
        coefs.append(coef)
        if keep_vtk:
            new_internals.append(intern)
            new_airfoils.append(aerofoil)
        
    if keep_vtk:
        return coefs, new_internals, new_airfoils
    else:
        return coefs
def Results_test(device, models, hparams, coef_norm, path_in, path_out,
                 n_test = 3, criterion = 'MSE', x_bl = [.2, .4, .6, .8], s = 'full_test'):

    sns.set()
    pathlib.Path(path_out).mkdir(parents = True, exist_ok = True)

    with open(osp.join(path_in, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    test_dataset = manifest[s]
    idx = random.sample(range(len(test_dataset)), k = n_test)
    idx.sort()

    # Déterminer si single-output (pression seule) à partir des hparams
    try:
        out_dim_hint = hparams[0]['decoder'][-1]
    except Exception:
        out_dim_hint = 1  # fallback raisonnable pour notre nouveau pipeline
    single_output = (out_dim_hint == 1)

    # Dataset de test cohérent :
    # - single_output -> surface_only + pressure_only
    # - legacy        -> tout le maillage, toutes sorties
    if single_output:
        test_dataset_vtk = Dataset(test_dataset, sample=None, coef_norm=coef_norm,
                                   surface_only=True, pressure_only=True)
    else:
        test_dataset_vtk = Dataset(test_dataset, sample=None, coef_norm=coef_norm)

    test_loader = DataLoader(test_dataset_vtk, shuffle=False)

    # Choix critère
    if criterion == 'MSE':
        criterion_fn = nn.MSELoss(reduction='none')
    elif criterion == 'MAE':
        criterion_fn = nn.L1Loss(reduction='none')
    else:
        raise ValueError("criterion must be 'MSE' or 'MAE'")

    # Conteneurs
    scores_vol = []
    scores_surf = []
    scores_force = []
    scores_p = []
    scores_wss = []
    internals = []
    airfoils = []
    true_internals = []
    true_airfoils = []
    times = []
    true_coefs = []
    pred_coefs = []

    # Boucle sur les seeds/entrainements par modèle
    for i in range(len(models[0])):
        model = [models[n][i] for n in range(len(models))]

        # Dimension des sorties (depuis un batch)
        first_batch = next(iter(test_loader))
        nvars = first_batch.y.shape[1]

        avg_loss_per_var = np.zeros((len(model), nvars))
        avg_loss = np.zeros(len(model))
        avg_loss_surf_var = np.zeros((len(model), nvars))
        avg_loss_vol_var = np.zeros((len(model), nvars))
        avg_loss_surf = np.zeros(len(model))
        avg_loss_vol = np.zeros(len(model))
        avg_rel_err_force = np.zeros((len(model), 2))
        avg_loss_p = np.zeros((len(model)))
        avg_loss_wss = np.zeros((len(model), 2))

        internal = []
        airfoil = []
        pred_coef = []

        for j, data in enumerate(tqdm(test_loader)):
            Uinf, angle = float(test_dataset[j].split('_')[2]), float(test_dataset[j].split('_')[3])

            outs, tim = Infer_test(device, model, hparams, data, coef_norm=coef_norm)
            times.append(tim)

            # Charger GT VTK
            intern_vtk = pv.read(osp.join(path_in, test_dataset[j], test_dataset[j] + '_internal.vtu'))
            aero_vtk   = pv.read(osp.join(path_in, test_dataset[j], test_dataset[j] + '_aerofoil.vtp'))

            if not single_output:
                # Legacy : on peut calculer les coefficients complets
                tc, true_intern, true_airfoil = Compute_coefficients([intern_vtk], [aero_vtk], data.surf, Uinf, angle, keep_vtk=True)
                tc, true_intern, true_airfoil = tc[0], true_intern[0], true_airfoil[0]
            else:
                # Pression seule : pas de WSS/forces ; on prépare une version "true_airfoil" avec wss=0 pour les Cp (et Cf=0)
                true_intern = intern_vtk.ptc(pass_point_data=True)
                true_airfoil = aero_vtk.ptc(pass_point_data=True)
                if 'wallShearStress' not in true_airfoil.point_data:
                    true_airfoil.point_data['wallShearStress'] = np.zeros((true_airfoil.n_points, 3))
                tc = np.zeros(2, dtype=float)  # placeholder

            # Appliquer Airfoil_test (projeter la prédiction)
            intern_pred_list, aero_pred_list = Airfoil_test(intern_vtk, aero_vtk, outs, coef_norm, data.surf)

            if not single_output:
                pc, intern_list, aero_list = Compute_coefficients(intern_pred_list, aero_pred_list, data.surf, Uinf, angle, keep_vtk=True)
            else:
                # Pression seule : pas de Compute_coefficients ; s'assurer que wallShearStress existe (zeros) pour surface_coefficients()
                intern_list, aero_list = [], []
                pc = []
                for k in range(len(aero_pred_list)):
                    ap = aero_pred_list[k].ptc(pass_point_data=True)
                    if 'wallShearStress' not in ap.point_data:
                        ap.point_data['wallShearStress'] = np.zeros((ap.n_points, 3))
                    intern_list.append(intern_pred_list[k])
                    aero_list.append(ap)
                    pc.append(np.zeros(2, dtype=float))  # placeholder

            if i == 0:
                if not single_output:
                    true_coefs.append(tc)

            pred_coef.append(pc)

            # Garder des échantillons pour visu/exports
            if j in idx:
                internal.append(intern_list)
                airfoil.append(aero_list)
                if i == 0:
                    true_internals.append(true_intern)
                    true_airfoils.append(true_airfoil)

            # Pertes
            for n, out in enumerate(outs):
                # calcul sur ce batch
                lpv = criterion_fn(out, data.y).mean(dim=0)
                lmean = lpv.mean()

                if single_output:
                    # Surface only (dataset construit ainsi) : pas de volume
                    lsurf_var = lpv
                    lvol_var = np.zeros_like(lsurf_var)
                    lsurf = lmean
                    lvol = 0.0
                else:
                    lsurf_var = criterion_fn(out[data.surf, :], data.y[data.surf, :]).mean(dim=0)
                    lvol_var  = criterion_fn(out[~data.surf, :], data.y[~data.surf, :]).mean(dim=0)
                    lsurf = lsurf_var.mean()
                    lvol  = lvol_var.mean()

                avg_loss_per_var[n] += lpv.cpu().numpy()
                avg_loss[n]         += lmean.cpu().numpy()
                avg_loss_surf_var[n]+= lsurf_var.cpu().numpy()
                avg_loss_vol_var[n] += lvol_var if isinstance(lvol_var, np.ndarray) else np.array([lvol_var])
                avg_loss_surf[n]    += lsurf if isinstance(lsurf, float) else lsurf.cpu().numpy()
                avg_loss_vol[n]     += lvol  if isinstance(lvol,  float) else lvol.cpu().numpy()

                if not single_output:
                    avg_rel_err_force[n] += rel_err(tc, pc[n])
                    avg_loss_wss[n] += rel_err(true_airfoil.point_data['wallShearStress'],
                                               aero_list[n].point_data['wallShearStress']).mean(axis=0)
                    avg_loss_p[n]   += rel_err(true_airfoil.point_data['p'],
                                               aero_list[n].point_data['p']).mean(axis=0)
                else:
                    # pression seule : on ne peut pas calculer WSS/forces ; on évalue quand même l'erreur relative sur p surf
                    avg_loss_p[n] += rel_err(true_airfoil.point_data['p'],
                                             aero_list[n].point_data['p']).mean(axis=0)

        internals.append(internal)
        airfoils.append(airfoil)
        pred_coefs.append(pred_coef)

        # Agrégation des scores
        score_var     = np.array(avg_loss_per_var)/len(test_loader)
        score         = np.array(avg_loss)/len(test_loader)
        score_surf_var= np.array(avg_loss_surf_var)/len(test_loader)
        score_vol_var = np.array(avg_loss_vol_var)/len(test_loader)
        score_surf    = np.array(avg_loss_surf)/len(test_loader)
        score_vol     = np.array(avg_loss_vol)/len(test_loader)
        score_force   = np.array(avg_rel_err_force)/len(test_loader)
        score_p       = np.array(avg_loss_p)/len(test_loader)
        score_wss     = np.array(avg_loss_wss)/len(test_loader)

        score = score_surf + score_vol
        scores_vol.append(score_vol_var)
        scores_surf.append(score_surf_var)
        scores_force.append(score_force)
        scores_p.append(score_p)
        scores_wss.append(score_wss)

    scores_vol   = np.array(scores_vol)
    scores_surf  = np.array(scores_surf)
    scores_force = np.array(scores_force)
    scores_p     = np.array(scores_p)
    scores_wss   = np.array(scores_wss)
    times        = np.array(times)

    if not single_output:
        true_coefs = np.array(true_coefs)
        pred_coefs = np.array(pred_coefs)
        pred_coefs_mean = pred_coefs.mean(axis = 0)
        pred_coefs_std  = pred_coefs.std(axis = 0)

        spear_coefs = []
        for j in range(pred_coefs.shape[0]):
            spear_coef = []
            for k in range(pred_coefs.shape[2]):
                spear_drag = sc.stats.spearmanr(true_coefs[:, 0], pred_coefs[j, :, k, 0])[0]
                spear_lift = sc.stats.spearmanr(true_coefs[:, 1], pred_coefs[j, :, k, 1])[0]
                spear_coef.append([spear_drag, spear_lift])
            spear_coefs.append(spear_coef)
        spear_coefs = np.array(spear_coefs)
    else:
        # Placeholders cohérents pour la signature de retour
        true_coefs = np.zeros((0, 2))
        pred_coefs_mean = np.zeros((0, 0, 2))
        pred_coefs_std  = np.zeros((0, 0, 2))
        spear_coefs     = np.zeros((0, 0, 2))

    with open(osp.join(path_out, 'score.json'), 'w') as f:
        json.dump(
            {   
                'mean_time': times.mean(axis = 0) if len(times) else 0.0,
                'std_time':  times.std(axis = 0) if len(times) else 0.0,
                'mean_score_vol': scores_vol.mean(axis = 0) if scores_vol.size else 0.0,
                'std_score_vol':  scores_vol.std(axis = 0)  if scores_vol.size else 0.0,
                'mean_score_surf': scores_surf.mean(axis = 0) if scores_surf.size else 0.0,
                'std_score_surf':  scores_surf.std(axis = 0)  if scores_surf.size else 0.0,
                'mean_rel_p': scores_p.mean(axis = 0) if scores_p.size else 0.0,
                'std_rel_p':  scores_p.std(axis = 0)  if scores_p.size else 0.0,
                'mean_rel_wss': scores_wss.mean(axis = 0) if scores_wss.size else 0.0,
                'std_rel_wss':  scores_wss.std(axis = 0)  if scores_wss.size else 0.0,
                'mean_score_force': scores_force.mean(axis = 0) if scores_force.size else 0.0,
                'std_score_force':  scores_force.std(axis = 0)  if scores_force.size else 0.0,
                'spearman_coef_mean': spear_coefs.mean(axis = 0) if spear_coefs.size else 0.0,
                'spearman_coef_std':  spear_coefs.std(axis = 0)  if spear_coefs.size else 0.0
            }, f, indent = 4, cls = NumpyEncoder
        )

    # Préparation des sorties "surface coefficients" et "boundary layers"
    surf_coefs = []
    true_surf_coefs = []
    bls = []
    true_bls = []

    # En pression seule, on peut calculer Cp (et Cf=0) pour visu ; boundary layer nécessite vitesses -> skip
    for i in range(len(internals[0])):
        aero_name = test_dataset[idx[i]]
        true_internal = true_internals[i]
        true_airfoil  = true_airfoils[i]

        surf_coef = []
        bl = []
        for j in range(len(internals[0][0])):
            internal_mean, airfoil_mean = Airfoil_mean(
                [internals[k][i][j] for k in range(len(internals))],
                [airfoils[k][i][j]   for k in range(len(airfoils))]
            )
            # pour éviter erreurs dans surface_coefficients si wss manquant
            if 'wallShearStress' not in airfoil_mean.point_data:
                airfoil_mean.point_data['wallShearStress'] = np.zeros((airfoil_mean.n_points, 3))
            surf_coef.append(np.array(metrics_NACA.surface_coefficients(airfoil_mean, aero_name)))

            # pas de boundary layer crédible en pression seule -> on saute
        # true
        if 'wallShearStress' not in true_airfoil.point_data:
            true_airfoil.point_data['wallShearStress'] = np.zeros((true_airfoil.n_points, 3))
        true_surf_coefs.append(np.array(metrics_NACA.surface_coefficients(true_airfoil, aero_name)))

        surf_coefs.append(np.array(surf_coef))
        bls.append(np.zeros((0,)))      # placeholder
        true_bls.append(np.zeros((0,))) # placeholder

    true_bls = np.array(true_bls, dtype=object)
    bls = np.array(bls, dtype=object)

    return true_coefs, pred_coefs_mean, pred_coefs_std, true_surf_coefs, surf_coefs, true_bls, bls
