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
def Infer_test(device, models, hparams, data, coef_norm=None):
    """
    Ensemble inference on one (surface-only) graph.
    - models: list of models (or list of modules for one architecture)
    - hparams: list of dicts (one per model) or a single dict to be reused
    Returns:
      - outs: list[Tensor] length == len(models), each (N_pts, C_out)
      - mean_times: np.array len == len(models)
    """
    # Normalize inputs: flatten a possible list-of-lists
    if len(models) > 0 and isinstance(models[0], (list, tuple)):
        # e.g. models = [[seed0, seed1, ...]] for a single architecture
        models = list(models[0])

    # If hparams is a single dict, reuse it for all models
    if isinstance(hparams, dict):
        hparams = [hparams for _ in range(len(models))]
    elif len(hparams) > 0 and isinstance(hparams[0], dict) and len(hparams) != len(models):
        # If you passed [hparams] from main, broadcast it
        hparams = [hparams[0] for _ in range(len(models))]

    # One output accumulator per model
    outs   = [torch.zeros_like(data.y) for _ in range(len(models))]
    n_out  = torch.zeros_like(data.y[:, :1])  # how many times each point was written

    remaining = set(map(tuple, data.pos[:, :2].cpu().numpy()))
    call_times = np.zeros(len(models), dtype=float)

    while remaining:
        idx = torch.tensor(
            random.sample(range(data.x.size(0)),
                          min(hparams[0].get('subsampling', data.x.size(0)), data.x.size(0)))
        )
        remaining -= set(map(tuple, data.pos[idx, :2].cpu().numpy()))

        # slice
        ds = data.clone()
        ds.pos  = ds.pos[idx]
        ds.x    = ds.x[idx]
        ds.y    = ds.y[idx]
        ds.surf = ds.surf[idx]
        if hasattr(ds, "batch") and ds.batch is not None:
            ds.batch = ds.batch[idx]

        # per model forward
        for n, model in enumerate(models):
            hp = hparams[n]
            try:
                '''
                ds.edge_index = nng.radius_graph(
                    x=ds.pos.to(device),
                    r=hp['r'], loop=True,
                    max_num_neighbors=int(hp['max_neighbors'])
                ).cpu()
                '''
                ds.edge_index = nng.radius_graph(
                    x=ds.pos,
                    r=hp['r'], loop=True,
                    max_num_neighbors=int(hp['max_neighbors'])
                ).cpu()
            except KeyError:
                ds.edge_index = None

            model.eval()
            dsd = ds.to(device)
            t0 = time.time()
            o = model(dsd).detach().cpu()  # (len(idx), C)
            call_times[n] += (time.time() - t0)

            outs[n][idx] += o

        n_out[idx] += 1.0

    # average over subsamples
    for n in range(len(outs)):
        outs[n] = outs[n] / (n_out + 1e-8)

        # If your training used normalized targets and you want denorm here:
        if coef_norm is not None:
            # y = y*std_out + mean_out
            outs[n] = outs[n] * (coef_norm[3][:outs[n].shape[1]] + 1e-8) + coef_norm[2][:outs[n].shape[1]]

    return outs, (call_times / max(1, (n_out.max().item())))

'''
def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    internals = []
    airfoils = []

    # gather the correspondence mesh(surface mask) -> airfoil points
    point_mesh  = internal.points[bool_surf, :2]
    point_surf  = airfoil.points[:, :2]

    for out in outs:
        # out already denormalized in Infer_test if coef_norm is not None
        p_pred_surf_on_mesh = out.squeeze(1).cpu().numpy()  # (Ns,)

        # Copy VTKs to avoid mutation
        intern  = internal.copy()
        aero    = airfoil.copy()

        # Project surface pressure from internal mesh surface points to the airfoil polyline
        p_on_airfoil = reorganize(point_mesh, point_surf, p_pred_surf_on_mesh)
        aero.point_data['p'] = p_on_airfoil

        # (Optional) pass point data to cells for plotting
        intern = intern.ptc(pass_point_data=True)
        aero   = aero.ptc(pass_point_data=True)

        internals.append(intern)
        airfoils.append(aero)

    return internals, airfoils
'''
def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf, graph_pos):
    """
    Map predicted *surface* pressures from the GRAPH to the airfoil polyline.

    Args
    ----
    internal : pv.UnstructuredGrid
        Full VTU (not indexed with the graph mask).
    airfoil  : pv.PolyData
        Airfoil polyline (.vtp).
    outs     : list[Tensor]
        One prediction tensor per model, each of shape (N_graph, 1).
    coef_norm : unused (kept for API compatibility)
    bool_surf : 1D boolean array/tensor of shape (N_graph,)
        Surface-node mask in GRAPH index space.
    graph_pos : array/tensor (N_graph, 3)
        Node positions of the GRAPH (same indexing as bool_surf).

    Returns
    -------
    internals : list[pv.UnstructuredGrid]
    airfoils  : list[pv.PolyData]
    """
    # --- convert inputs / basic checks ---
    if torch.is_tensor(bool_surf):
        bool_surf = bool_surf.detach().cpu().numpy().astype(bool)
    else:
        bool_surf = np.asarray(bool_surf, dtype=bool)

    if torch.is_tensor(graph_pos):
        graph_pos = graph_pos.detach().cpu().numpy()
    else:
        graph_pos = np.asarray(graph_pos)

    assert bool_surf.ndim == 1, "Airfoil_test: surf mask must be 1D."
    assert graph_pos.shape[0] == bool_surf.shape[0], (
        f"Airfoil_test: graph_pos N={graph_pos.shape[0]} != mask N={bool_surf.shape[0]}"
    )

    # graph-surface coordinates (what the mask was built for)
    point_mesh = graph_pos[bool_surf, :2]            # (Ns, 2)
    point_surf = airfoil.points[:, :2]               # (Na, 2)

    internals = []
    airfoils  = []

    for out in outs:
        # out is (N_graph, 1) -> keep only surface predictions
        if torch.is_tensor(out):
            out = out.detach().cpu().numpy()
        out = np.asarray(out)

        assert out.ndim == 2 and out.shape[1] == 1, "Airfoil_test: each out must be (N_graph, 1)."
        p_pred_surf_on_mesh = out[bool_surf, 0]      # (Ns,)

        # copy VTKs to avoid mutation
        intern = internal.copy()
        aero   = airfoil.copy()

        # project surface pressures from graph-surface pts to airfoil polyline
        p_on_airfoil = reorganize(point_mesh, point_surf, p_pred_surf_on_mesh)
        aero.point_data['p'] = p_on_airfoil

        # optional: ensure both point and cell data exist for plotting/integration
        intern = intern.ptc(pass_point_data=True)
        aero   = aero.ptc(pass_point_data=True)

        internals.append(intern)
        airfoils.append(aero)

    return internals, airfoils

'''def Airfoil_mean(airfoils):
    oa_point = np.zeros((airfoils[0].points.shape[0], 1))
    oa_cell  = np.zeros((airfoils[0].cell_data['p'].shape[0], 1))

    for a in airfoils:
        oa_point[:, 0] += a.point_data['p']
        oa_cell[:, 0]  += a.cell_data['p']

    oa_point /= len(airfoils)
    oa_cell  /= len(airfoils)

    a_mean = airfoils[0].copy()
    a_mean.point_data['p'] = oa_point[:, 0]
    a_mean.cell_data['p']  = oa_cell[:, 0]
    return a_mean'''

def Airfoil_mean(airfoils):
    """
    Average several predicted airfoil VTPs (same topology),
    robust to multi-component 'p' on points/cells and to missing
    point/cell arrays (uses ctp/ptc when needed).

    Returns a single VTP with averaged 'p' on points and cells.
    """
    if not airfoils:
        raise ValueError("Airfoil_mean: received an empty list.")

    # --- get reference shapes from the first mesh (after scalar reduction) ---
    a0 = airfoils[0].copy()

    # ensure 'p' on points (scalar)
    if 'p' not in a0.point_data:
        a0 = a0.ctp(pass_cell_data=True)  # move cell->point if needed
    p_pt0 = np.asarray(a0.point_data['p'])
    if p_pt0.ndim == 2:                   # e.g. (Np, k)
        p_pt0 = p_pt0.mean(axis=1)        # -> (Np,)
    if p_pt0.ndim != 1:
        raise ValueError("Airfoil_mean: could not reduce point 'p' to 1D.")

    # ensure 'p' on cells (scalar)
    if 'p' not in a0.cell_data:
        a0 = a0.ptc(pass_point_data=True) # move point->cell if needed
    p_cell0 = np.asarray(a0.cell_data['p'])
    if p_cell0.ndim == 2:                 # e.g. (Nc, k)
        p_cell0 = p_cell0.mean(axis=1)    # -> (Nc,)
    if p_cell0.ndim != 1:
        raise ValueError("Airfoil_mean: could not reduce cell 'p' to 1D.")

    Np = p_pt0.shape[0]
    Nc = p_cell0.shape[0]
    sum_pt   = np.zeros(Np, dtype=float)
    sum_cell = np.zeros(Nc, dtype=float)

    # --- accumulate over all inputs ---
    for a in airfoils:
        ac = a.copy()

        # points -> ensure scalar 'p'
        if 'p' not in ac.point_data:
            ac = ac.ctp(pass_cell_data=True)
        p_pt = np.asarray(ac.point_data['p'])
        if p_pt.ndim == 2:
            p_pt = p_pt.mean(axis=1)
        if p_pt.ndim != 1:
            raise ValueError("Airfoil_mean: could not reduce point 'p' to 1D during accumulation.")
        if p_pt.shape[0] != Np:
            raise ValueError("Airfoil_mean: inconsistent number of points across inputs.")
        sum_pt += p_pt

        # cells -> ensure scalar 'p'
        if 'p' not in ac.cell_data:
            ac = ac.ptc(pass_point_data=True)
        p_cell = np.asarray(ac.cell_data['p'])
        if p_cell.ndim == 2:
            p_cell = p_cell.mean(axis=1)
        if p_cell.ndim != 1:
            raise ValueError("Airfoil_mean: could not reduce cell 'p' to 1D during accumulation.")
        if p_cell.shape[0] != Nc:
            raise ValueError("Airfoil_mean: inconsistent number of cells across inputs.")
        sum_cell += p_cell

    # --- finalize mean ---
    mean_pt   = (sum_pt   / len(airfoils)).astype(float)
    mean_cell = (sum_cell / len(airfoils)).astype(float)

    out = airfoils[0].copy()
    out.point_data['p'] = mean_pt
    out.cell_data['p']  = mean_cell
    return out

def Compute_coefficients(airfoils, Uinf, angle_deg, include_viscous=False):
    """
    Compute force coefficients from surface fields (pressure-only if include_viscous=False).
    Handles cases where point->cell conversion produced multi-component cell 'p'
    (e.g., one value per point of the cell).
    """
    coefs = []
    alpha = np.deg2rad(angle_deg)
    R = np.array([[ np.cos(alpha),  np.sin(alpha)],
                  [-np.sin(alpha),  np.cos(alpha)]])  # global->body (x:drag, y:lift aligned with freestream)

    for aero in airfoils:
        # Work on a copy; make sure required geometric arrays exist
        a = aero.copy()

        # Ensure cell lengths and normals are available
        a = a.compute_cell_sizes(length=True)           # provides 'Length' in cell_data
        a = a.compute_normals(cell_normals=True)        # provides 'Normals' in cell_data

        # Ensure pressure is on cells
        if 'p' in a.cell_data:
            p_cell = a.cell_data['p']
        else:
            # Convert point->cell if only point data exists
            a = a.ptc(pass_point_data=True)
            p_cell = a.cell_data['p']

        p_cell = np.asarray(p_cell)

        # If 'p' became multi-component per cell (e.g., num points per cell),
        # reduce to a single scalar (mean is a reasonable choice for panel integration).
        if p_cell.ndim == 2:
            p = p_cell.mean(axis=1)   # (Nc,)
        else:
            p = p_cell                # (Nc,)

        # Geometry
        n = np.asarray(a.cell_data['Normals'])[:, :2]              # (Nc,2)
        L = np.asarray(a.cell_data['Length']).reshape(-1, 1)       # (Nc,1)

        # Pressure force (2D): integrate -p * n * ds
        Fp_local = -p[:, None] * n * L     # (Nc,2)
        Fp = Fp_local.sum(axis=0)          # (2,)

        F_total = Fp.copy()

        # Optional viscous term if available/requested
        if include_viscous and ('wallShearStress' in a.cell_data):
            tau = np.asarray(a.cell_data['wallShearStress'])       # (Nc,2) in global coords
            Ft = (tau * L).sum(axis=0)                             # (2,)
            F_total = Fp + Ft

        # Rotate to drag/lift frame
        F_rot = R @ F_total
        C = 2.0 * F_rot / (Uinf**2)  # rho=1 convention (same as earlier code)
        coefs.append(C)

    return coefs

def Results_test(device, models, hparams, coef_norm, path_in, path_out,
                 n_test=3, criterion='MSE', s='full_test'):
    """
    Surface-only evaluation:
    - field loss: pressure on surface points
    - coefficients: pressure-only drag/lift
    - saves score.json
    """
    sns.set()
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    with open(osp.join(path_in, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    test_dataset = manifest[s]
    # choose n_test random indices deterministically for reproducibility if needed
    pick = random.sample(range(len(test_dataset)), k=n_test)
    pick.sort()

    # load graphs (your Dataset now surface-only; coef_norm applied inside if you passed it)
    # test_dataset_vtk = Dataset(test_dataset, sample=None, coef_norm=coef_norm)
    test_dataset_vtk = Dataset(test_dataset, sample = 'uniform', coef_norm = coef_norm, surf_ratio = 1)
    test_loader = DataLoader(test_dataset_vtk, shuffle=False)

    if criterion == 'MSE':
        crit = nn.MSELoss(reduction='none')
    elif criterion == 'MAE':
        crit = nn.L1Loss(reduction='none')
    else:
        raise ValueError("criterion must be 'MSE' or 'MAE'")

    times = []
    # per-architecture accumulators
    field_losses = []            # shape: [n_arch,]
    rel_p_errors = []            # per-arch mean relative |p_true - p_pred| / |p_true|
    all_true_coefs = []
    all_pred_coefs = []

    # we will also collect VTKs for a few picked cases to write out
    airfoils_true = []
    airfoils_pred_all_arch = []  # [arch][case] -> list of vtps per run to later average

    for arch_i in range(len(models[0])):  # loop over training seeds or variants
        # select the i-th model from each arch bucket
        model_group = [models[n][arch_i] for n in range(len(models))]

        loss_sum = np.zeros(len(model_group))
        relp_sum = np.zeros(len(model_group))
        n_batches = 0

        pred_coefs_this_seed = []

        for j, data in enumerate(tqdm(test_loader)):
            case_name = test_dataset[j]
            Uinf  = float(case_name.split('_')[2])
            angle = float(case_name.split('_')[3])

            # predict
            outs, tim = Infer_test(device, model_group, hparams, data, coef_norm=coef_norm)
            times.append(tim)

            # load reference meshes
            intern_ref = pv.read(osp.join(path_in, case_name, case_name + '_internal.vtu'))
            aero_ref   = pv.read(osp.join(path_in, case_name, case_name + '_aerofoil.vtp'))

            # true pressure on surface polyline
            # pass to cells for integration
            aero_true = aero_ref.copy().ptc(pass_point_data=True)

            # map ground-truth surface pressure from graph to airfoil (they match already if graph was surface-only, but
            # we use reorganization through internal surface mask when needed)
            # Here, simplest: the file already contains 'p' on points/cells:
            true_coef = Compute_coefficients([aero_true], Uinf, angle, include_viscous=False)[0]

            if arch_i == 0:
                all_true_coefs.append(true_coef)

            # write predicted pressure to airfoil (one per model in the group)
            # intern_pred_list, aero_pred_list = Airfoil_test(intern_ref, aero_ref, outs, coef_norm, data.surf)
            intern_pred_list, aero_pred_list = Airfoil_test(
                internal=intern_ref,
                airfoil=aero_ref,
                outs=outs,
                coef_norm=coef_norm,
                bool_surf=data.surf,
                graph_pos=data.pos,
            )
            # compute coeffs per model (pressure-only)
            pred_coef_models = Compute_coefficients(aero_pred_list, Uinf, angle, include_viscous=False)
            pred_coefs_this_seed.append(pred_coef_models)

            # collect a few VTKs to save later
            if j in pick:
                if arch_i == 0:
                    airfoils_true.append(aero_true)
                # store per-arch per-case list to average later
                # initialize the container at first case
                if len(airfoils_pred_all_arch) <= arch_i:
                    airfoils_pred_all_arch.append([])
                airfoils_pred_all_arch[arch_i].append(aero_pred_list)

            # losses on pressure only
            for n, out in enumerate(outs):
                # data.y may be normalized or not depending on Dataset; we want *denormalized* field loss for interpretability
                y_true = data.y
                if coef_norm is not None:
                    y_true = y_true*(coef_norm[3][:1] + 1e-8) + coef_norm[2][:1]

                lpv = crit(out, y_true).mean(dim=0)      # (1,)
                loss_sum[n] += lpv.cpu().numpy()[0]

                relp = torch.mean(torch.abs((y_true - out) / (y_true + 1e-8)))
                relp_sum[n] += relp.cpu().numpy()

            n_batches += 1

        field_losses.append(loss_sum / max(n_batches, 1))
        rel_p_errors.append(relp_sum / max(n_batches, 1))
        all_pred_coefs.append(np.array(pred_coefs_this_seed))  # shape: [n_cases, n_models, 2]

    # aggregate over seeds
    field_losses  = np.array(field_losses)      # [n_seeds, n_models]
    rel_p_errors  = np.array(rel_p_errors)      # [n_seeds, n_models]
    times         = np.array(times)             # if you filled it
    true_coefs    = np.array(all_true_coefs)    # [n_cases, 2]
    pred_coefs    = np.array(all_pred_coefs)    # [n_seeds, n_cases, n_models, 2]
    pred_mean     = pred_coefs.mean(axis=0)     # [n_cases, n_models, 2]
    pred_std      = pred_coefs.std(axis=0)      # [n_cases, n_models, 2]

    # Spearman (per model) against truth, separately for CD and CL
    spear = []
    for m in range(pred_mean.shape[1]):
        sd = sc.stats.spearmanr(true_coefs[:, 0], pred_mean[:, m, 0])[0]
        sl = sc.stats.spearmanr(true_coefs[:, 1], pred_mean[:, m, 1])[0]
        spear.append([sd, sl])
    spear = np.array(spear)

    # Save metrics
    with open(osp.join(path_out, 'score.json'), 'w') as f:
        json.dump({
            'mean_time': getattr(times, 'mean', lambda: 0)() if hasattr(times, 'mean') else 0,
            'std_time':  getattr(times, 'std',  lambda: 0)() if hasattr(times, 'std')  else 0,
            'mean_field_loss': field_losses.mean(axis=0),
            'std_field_loss':  field_losses.std(axis=0),
            'mean_rel_p': rel_p_errors.mean(axis=0),
            'std_rel_p':  rel_p_errors.std(axis=0),
            'spearman_coef_mean': spear.mean(axis=0),
            'spearman_coef_std':  spear.std(axis=0),
        }, f, indent=4, cls=NumpyEncoder)

    # Optionally save a few averaged predicted airfoils
    for ix_case, j in enumerate(pick):
        case_name = test_dataset[j]
        # average across models/seeds
        # flatten list over seeds -> list of lists (per seed) of VTPs (per model)
        vtps_all = []
        for seed_idx in range(len(airfoils_pred_all_arch)):
            vtps_all.extend(airfoils_pred_all_arch[seed_idx][ix_case])
        # average them
        # simple average on 'p' (use Airfoil_mean on a flat list)
        a_mean = Airfoil_mean(vtps_all)
        a_mean.save(osp.join(path_out, f"{case_name}_pred_surface.vtp"))

    return true_coefs, pred_mean, pred_std
