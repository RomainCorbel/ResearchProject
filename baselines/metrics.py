import torch
import numpy as np
import os.path as osp
from torch_geometric.loader import DataLoader
import random
import torch_geometric.nn as nng
import json
from dataset import Dataset

def Results_test(device, models_list, hparams_list, coef_norm, path_in='Dataset', path_out='scores', 
                 n_test=3, criterion='MSE', s='full_test'):
    """
    Compute test results for surface-only pressure prediction models.
    
    Args:
        device: torch device
        models_list: list of trained models (or list of model lists)
        hparams_list: list of hyperparameter dictionaries
        coef_norm: normalization coefficients
        path_in: input data path
        path_out: output scores path
        n_test: number of test samples to evaluate
        criterion: loss criterion ('MSE' or 'MAE')
        s: test dataset key
    
    Returns:
        true_coefs: true pressure values
        pred_mean: mean predicted pressure values  
        pred_std: standard deviation of predictions
    """
    
    # Load test dataset
    with open(osp.join(path_in, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    test_files = manifest[s]
    
    test_dataset = Dataset(test_files, sample='uniform', coef_norm=coef_norm, surf_ratio=1)
    
    # Initialize storage
    all_true = []
    all_preds = []
    
    # Evaluate each model
    for model_idx, (models, hparams) in enumerate(zip(models_list, hparams_list)):
        model_preds = []
        
        for model in models:
            model.eval()
            model_preds_batch = []
            true_batch = []
            
            # Create test loader with sampling (similar to validation)
            for _ in range(n_test):
                test_dataset_sampled = []
                
                for data in test_dataset:
                    data_sampled = data.clone()
                    n = data_sampled.x.size(0)
                    k = min(hparams['subsampling'], n)
                    idx = torch.tensor(random.sample(range(n), k))
                    
                    data_sampled.pos = data_sampled.pos[idx]
                    data_sampled.x = data_sampled.x[idx] 
                    data_sampled.y = data_sampled.y[idx]
                    data_sampled.surf = data_sampled.surf[idx]
                    
                    # Build edges for graph models only (not for MLP or PointNet)
                    if 'r' in hparams and hparams['r'] is not None:
                        '''
                        data_sampled.edge_index = nng.radius_graph(
                            x=data_sampled.pos,
                            r=hparams['r'],
                            loop=True,
                            max_num_neighbors=int(hparams['max_neighbors'])
                        ).cpu()
                        '''
                        data_sampled.edge_index = nng.radius_graph(
                            x=data_sampled.pos.to(device),
                            r=hparams['r'],
                            loop=True,
                            max_num_neighbors=int(hparams['max_neighbors'])
                        ).cpu()
                    
                    test_dataset_sampled.append(data_sampled)
            
            test_loader = DataLoader(test_dataset_sampled, batch_size=1, shuffle=False)
            
            # Run inference
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    pred = model(data)
                    y = data.y
                    
                    if pred.dim() == 1: 
                        pred = pred.unsqueeze(1)
                    if y.dim() == 1:    
                        y = y.unsqueeze(1)
                    
                    # Store only surface node predictions
                    m_surf = data.surf
                    if m_surf.any():
                        model_preds_batch.append(pred[m_surf].cpu().numpy())
                        if model_idx == 0:  # Store true values only once
                            true_batch.append(y[m_surf].cpu().numpy())
            
            model_preds.append(np.concatenate(model_preds_batch) if model_preds_batch else np.array([]))
            
        if model_idx == 0 and true_batch:  # Store true values only once
            all_true = np.concatenate(true_batch)
        
        all_preds.append(model_preds)
    
    # Convert to numpy arrays
    true_coefs = all_true if isinstance(all_true, np.ndarray) else np.array([])
    
    # Compute mean and std across models
    if all_preds and len(all_preds[0]) > 0:
        # Stack predictions from all models
        pred_arrays = []
        for model_preds in all_preds:
            for pred in model_preds:
                if len(pred) > 0:
                    pred_arrays.append(pred)
        
        if pred_arrays:
            # Ensure all arrays have the same length by taking the minimum length
            min_length = min(len(arr) for arr in pred_arrays)
            pred_arrays_truncated = [arr[:min_length] for arr in pred_arrays]
            
            pred_stack = np.stack(pred_arrays_truncated, axis=0)
            pred_mean = np.mean(pred_stack, axis=0)
            pred_std = np.std(pred_stack, axis=0)
        else:
            pred_mean = np.array([])
            pred_std = np.array([])
    else:
        pred_mean = np.array([])
        pred_std = np.array([])
    
    return true_coefs, pred_mean, pred_std