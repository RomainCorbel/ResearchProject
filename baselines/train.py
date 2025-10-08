import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time, json

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

import metrics

from tqdm import tqdm

from pathlib import Path
import os.path as osp

def get_nb_trainable_params(model):
   '''
   Return the number of trainable parameters
   '''
   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
   return sum([np.prod(p.size()) for p in model_parameters])

def train(device, model, train_loader, optimizer, scheduler, criterion='MSE'):
    model.train()
    avg_loss = torch.tensor(0.0, device=device)
    iters = 0

    if criterion == 'MSE':
        loss_criterion = nn.MSELoss(reduction='none')
    elif criterion == 'MAE':
        loss_criterion = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    for data in train_loader:
        data = data.clone().to(device)
        optimizer.zero_grad()

        pred = model(data)        # expect shape (N,1)
        y = data.y                # expect shape (N,1)

        if pred.dim() == 1: pred = pred.unsqueeze(1)
        if y.dim() == 1:    y    = y.unsqueeze(1)
        assert pred.shape == y.shape == (y.size(0), 1), f"{pred.shape} vs {y.shape}"

        m_surf = data.surf
        # strictly surface-only loss
        if not m_surf.any():
            continue  # nothing to learn from
        loss = loss_criterion(pred[m_surf], y[m_surf]).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_loss += loss
        iters += 1

    # return scalars (floats)
    return float((avg_loss / max(iters,1)).detach().cpu())


@torch.no_grad()
def test(device, model, test_loader, criterion='MSE'):
    model.eval()
    avg_loss = 0.0
    iters = 0

    if criterion == 'MSE':
        loss_criterion = nn.MSELoss(reduction='none')
    elif criterion == 'MAE':
        loss_criterion = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    for data in test_loader:
        data = data.clone().to(device)
        pred = model(data)
        y = data.y

        if pred.dim() == 1: pred = pred.unsqueeze(1)
        if y.dim() == 1:    y    = y.unsqueeze(1)
        assert pred.shape == y.shape == (y.size(0), 1), f"{pred.shape} vs {y.shape}"

        m_surf = data.surf
        if not m_surf.any():
            continue
        loss = loss_criterion(pred[m_surf], y[m_surf]).mean()

        avg_loss += float(loss.detach().cpu())
        iters += 1

    return avg_loss / max(iters,1)


"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
"""
# ---  new NumpyEncoder because of the new env  ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np, torch
        if isinstance(obj, np.generic):       # np.float32, np.int64, etc.
            return obj.item()
        if isinstance(obj, np.ndarray):       # arrays
            return obj.tolist()
        if isinstance(obj, torch.Tensor):     # tensors
            return obj.detach().cpu().item() if obj.dim() == 0 else obj.detach().cpu().tolist()
        return super().default(obj)

def main(device, train_dataset, val_dataset, Net, hparams, path,
         criterion='MSE', reg=1, val_iter=10, name_mod='GraphSAGE', val_sample=True):
    """
    Surface-only, pressure-only training loop.
    - Trains and validates using only surface nodes (data.surf == True)
    - Targets are expected to be shape (N,1) with pressure only
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
    )

    start = time.time()

    train_loss_list = []
    val_loss_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    last_val_loss = float('nan')  # for display when we skip validation this epoch

    for epoch in pbar_train:
        # ---- subsample training graphs each epoch ----
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            # subsample nodes (uniform) on THIS graph
            n = data_sampled.x.size(0)
            k = min(hparams['subsampling'], n)
            idx = torch.tensor(random.sample(range(n), k))

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]

            # build edges if needed (graph models)
            if name_mod not in ('PointNet', 'MLP'):
                
                data_sampled.edge_index = nng.radius_graph(
                    x=data_sampled.pos.to(device),
                    r=hparams['r'],
                    loop=True,
                    max_num_neighbors=int(hparams['max_neighbors'])
                ).cpu()
                '''
                data_sampled.edge_index = nng.radius_graph(
                    x=data_sampled.pos,
                    r=hparams['r'],
                    loop=True,
                    max_num_neighbors=int(hparams['max_neighbors'])
                ).cpu()
                '''
            train_dataset_sampled.append(data_sampled)

        train_loader = DataLoader(
            train_dataset_sampled,
            batch_size=hparams['batch_size'],
            shuffle=True
        )
        del train_dataset_sampled

        # ---- train one epoch (surface-only loss) ----
        train_loss = train(device, model, train_loader, optimizer, lr_scheduler, criterion)
        del train_loader
        train_loss_list.append(train_loss)

        # ---- validation every val_iter epochs (surface-only) ----
        if val_iter is not None and (epoch % val_iter == val_iter - 1 or epoch == 0):
            if val_sample:
                # resample validation multiple times and average
                val_losses = []
                for _ in range(20):
                    val_dataset_sampled = []
                    for data in val_dataset:
                        data_sampled = data.clone()
                        n = data_sampled.x.size(0)
                        k = min(hparams['subsampling'], n)
                        idx = torch.tensor(random.sample(range(n), k))

                        data_sampled.pos = data_sampled.pos[idx]
                        data_sampled.x = data_sampled.x[idx]
                        data_sampled.y = data_sampled.y[idx]
                        data_sampled.surf = data_sampled.surf[idx]

                        if name_mod not in ('PointNet', 'MLP'):
                            
                            data_sampled.edge_index = nng.radius_graph(
                                x=data_sampled.pos.to(device),
                                r=hparams['r'],
                                loop=True,
                                max_num_neighbors=int(hparams['max_neighbors'])
                            ).cpu()
                            '''
                            data_sampled.edge_index = nng.radius_graph(
                                x=data_sampled.pos,
                                r=hparams['r'],
                                loop=True,
                                max_num_neighbors=int(hparams['max_neighbors'])
                            ).cpu()
                            '''
                        val_dataset_sampled.append(data_sampled)

                    val_loader = DataLoader(val_dataset_sampled, batch_size=1, shuffle=True)
                    val_loss = test(device, model, val_loader, criterion)
                    del val_loader, val_dataset_sampled
                    val_losses.append(val_loss)

                val_loss = float(np.mean(val_losses))
            else:
                # no resampling: build a single val loader
                # (optionally precompute edge_index once outside the loop if desired)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                val_loss = test(device, model, val_loader, criterion)
                del val_loader

            val_loss_list.append(val_loss)
            last_val_loss = val_loss
            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss, val_loss=last_val_loss)

    # ---- save model ----
    time_elapsed = time.time() - start
    params_model = float(get_nb_trainable_params(model))
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, osp.join(path, 'model'))

    # ---- plots: single curve for train, single for val ----
    sns.set()
    fig_train, ax_train = plt.subplots(figsize=(20, 5))
    ax_train.plot(train_loss_list, label='Train loss (surface, p)')
    ax_train.set_xlabel('epochs')
    ax_train.set_yscale('log')
    ax_train.set_title('Train loss (surface-only, pressure)')
    ax_train.legend(loc='best')
    fig_train.savefig(osp.join(path, 'train_loss.png'), dpi=150, bbox_inches='tight')

    if val_iter is not None and len(val_loss_list) > 0:
        fig_val, ax_val = plt.subplots(figsize=(20, 5))
        ax_val.plot(val_loss_list, label='Val loss (surface, p)')
        ax_val.set_xlabel('validation checkpoints')
        ax_val.set_yscale('log')
        ax_val.set_title('Validation loss (surface-only, pressure)')
        ax_val.legend(loc='best')
        fig_val.savefig(osp.join(path, 'val_loss.png'), dpi=150, bbox_inches='tight')

    # ---- minimal JSON log ----
    with open(osp.join(path, name_mod + '_log.json'), 'a') as f:
        json.dump(
            {
                'task': 'pressure_surface_only',
                'loss': criterion,
                'nb_parameters': params_model,
                'time_elapsed': time_elapsed,
                'hparams': hparams,
                'train_loss': train_loss_list[-1],
                'val_loss': (val_loss_list[-1] if len(val_loss_list) else None),
            },
            f, indent=2, cls=NumpyEncoder
        )

    return model
