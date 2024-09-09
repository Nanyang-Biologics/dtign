import os
import re
import argparse
import time
import json
import shutil
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import zip_longest, cycle
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import AverageMeter, flatten, standardize, delta_weight
from DTIGN import DTIGN
from config.config_dict import Config
from aws import S3DataFetcher
from log.train_logger import TrainLogger

# Set CUDA device and multiprocessing strategy
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.multiprocessing.set_sharing_strategy('file_system')

def reduction_by_pocket(tensor, pocket_list, device, semi_supervise=False):
    output_tensor = torch.empty(0).to(device)
    current_index = 0
    
    for pockets in pocket_list:
        additional_num = sum(1 for pocket in pockets if pocket.startswith('G'))
        pocket_num = len(pockets)
        reducted_tensor = tensor[current_index:current_index + pocket_num].mean(dim=0, keepdim=True)
        
        output_tensor = torch.cat((output_tensor, reducted_tensor), dim=0)
        
        if semi_supervise:
            output_tensor = torch.cat((output_tensor, reducted_tensor.repeat(additional_num, 1)), dim=0)
        
        current_index += pocket_num
    
    return output_tensor

def val(model, dataloader, device, epoch, attention_dict, save_attention=False, val_rate=1):
    model.eval()
    predict_dict, label_dict = {}, {}
    
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            label, pocket, idxes = data.y, data.pocket, data.idx
            pred, outputs = model(data=data, pocket_list=pocket, self_attention=True, 
                                  semi_supervise=semi_supervise, save_attention=save_attention, 
                                  graph_type=graph_type)
            
            if save_attention:
                attention_weights = outputs[1]
                for i, idx in enumerate(idxes):
                    attention_dict.setdefault(idx, []).append(attention_weights[i])
            
            pred = standardize(pred, weighted_mean, weighted_std, device, reverse=True)
            label = reduction_by_pocket(label, pocket, device, semi_supervise=semi_supervise)
            idx_list = flatten(idxes)
            
            for i, idx in enumerate(idx_list):
                predict_dict.setdefault(idx, []).append(pred[i].detach().cpu())
                label_dict.setdefault(idx, []).append(label[i].detach().cpu())
    
    pred, label = map(np.array, ([sum(vals) / len(vals) for vals in predict_dict.values()], 
                                 [sum(vals) / len(vals) for vals in label_dict.values()]))
    
    random_indices = np.random.choice(len(pred), int(len(pred) * val_rate), replace=False)
    pred, label = pred[random_indices], label[random_indices]
    
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    tau, p_value = stats.kendalltau(pred, label)

    model.train()
    
    return rmse, coff, tau, attention_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model with specified settings.')
    parser.add_argument('--setting', type=str, required=True, help='The setting for the training session')
    return parser.parse_args()

def configure_task(args):
    task_settings = {
        'I1': (2, [3], 4, 40, 1, 0, 1e-4, 128, 100, 5, 10, 0.95, 100, 32, 8),
        'I2': (1, [], 5, 40, 1, 0, 1e-4, 128, 1, 5, 10, 0.95, 100, 32, 8),
        'I3': (1, [], 1, 40, 0.9, 2, 8e-5, 128, 100, 5, 10, 0.975, 100, 32),
        'I4': (3, [], 3, 0, 1, 4, 8e-5, 128, 100, 5, 10, 0.95, 100, 32, 4),
        'I5': (3, [], 3, 0, 1, 4, 8e-5, 128, 100, 5, 10, 0.95, 100, 32, 4),
        'E1': (1, [], 1, 40, 0.9, 2, 8e-5, 128, 50, 5, 10, 0.975, 100, 32, 4),
        'E2': (1, [], 5, 40, 1, 0, 1e-4, 128, 1, 5, 10, 0.95, 100, 64, 4),
        'E3': (3, [], 3, 0, 1, 4, 8e-5, 128, 100, 5, 10, 0.95, 100, 32, 4)
    }
    return task_settings.get(args.setting)

def load_task_config(task_id):
    task_dict = {
        'I1': ('CHEMBL202', 'pIC50', '1boz', 7, 1),
        'I2': ('CHEMBL3976', 'pIC50', '4ebb', 2, 4),
        'I3': ('CHEMBL333', 'pIC50', '1ck7', 6, 3),
        'I4': ('CHEMBL2971', 'pIC50', '3ugc', 3, 4),
        'I5': ('CHEMBL279', 'pIC50', '1ywn', 3, 4),
        'E1': ('CHEMBL3820', 'pEC50', '3f9m', 6, 3),
        'E2': ('CHEMBL4422', 'pEC50', '5tzr', 3, 3),
        'E3': ('CHEMBL235', 'pEC50', '1zgy', 4, 3)
    }
    return task_dict[task_id]

def fetch_data(protein_name):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    local_dir = './data/'
    data_fetcher = S3DataFetcher(aws_access_key_id, aws_secret_access_key, f's3://dtign-benchmark/{protein_name}.zip')
    data_fetcher.fetch_and_extract(local_dir, protein_name)
    return local_dir

def setup_training_environment(args, graph_type, task_id, num_pose, seed):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args['save_dir'] = f"{args.get('save_dir')}/{timestamp}_{args['mark']}/"
    torch.manual_seed(seed)
    np.random.seed(seed)
    args['mark'] = f"{task_id}_{graph_type}_Pose_{num_pose}_Seed_{seed}_hidden_dim={args['hidden_dim']}_lr={args['learning_rate']}_val_num={args['val_num']}-val_rate={args['val_rate']}-patience={args['early_stop_epoch']}-D_count={args['D_count']}"
    
    logger = TrainLogger(args, cfg='TrainConfig_DTIGN', create=True)
    logger.info(__file__)
    
    return logger

def prepare_data_loaders(args, graph_type, task_id, data_root, ground_root, num_pose, create, semi_supervise):
    train_set_list_all, train_loader_list_all = [], []
    
    for fold_temp in range(args['folds']):
        set_name = f'train_{fold_temp+1}'
        train_dir = os.path.join(data_root, set_name)
        ground_dir = os.path.join(ground_root, set_name)
        train_df = pd.read_csv(os.path.join(data_root, f'{args["protein_name"]}_{args["assay_type"]}_{set_name}.csv'))
        
        if fold_temp == args['fold'] - 1:
            val_to_train_dir = train_dir
            val_to_train_df = train_df.sample(n=len(train_df) - args['val_num'], random_state=args['seed'])
            train_df = train_df.drop(val_to_train_df.index).reset_index(drop=True)
            val_to_train_df = val_to_train_df.reset_index(drop=True)
        
        train_set = GraphDataset(train_dir, ground_dir, train_df, run_datafold=args['DTIGN_datafold'], num_pose=num_pose, graph_type=graph_type, assay_type=args['assay_type'], create=create)
        train_loader = PLIDataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
        train_set_list_all.append(train_set)
        train_loader_list_all.append(train_loader)
    
    val_to_train_set = GraphDataset(val_to_train_dir, ground_dir, val_to_train_df, run_datafold=args['DTIGN_datafold'], num_pose=num_pose, graph_type=graph_type, assay_type=args['assay_type'], create=create)
    val_to_train_loader = PLIDataLoader(val_to_train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    train_set_list_all.append(val_to_train_set)
    train_loader_list_all.append(val_to_train_loader)
    
    return train_set_list_all, train_loader_list_all

def setup_model_and_training(args, device):
    model = DTIGN(
        node_dim=35,
        bond_dim=10,
        hidden_dim=args['hidden_dim'],
        num_pose=args['num_pose'],
        dropout=args['dropout'],
        self_attention=True,
        graph_type=args['graph_type'],
        D_count=args['D_count']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    
    return model, optimizer, scheduler

if __name__ == '__main__':
    args = parse_arguments()
    print(f"Training model with setting: {args.setting}")
    
    task_id = args.setting
    config = Config('TrainConfig_DTIGN')
   
   
    task_params = configure_task(args)

    args = config.get_config()
    
    args.update({
        'start_fold': task_params[0],
        'skip_fold': task_params[1],
        'stop_fold': task_params[2],
        'warmup_epoch': task_params[3],
        'val_rate': task_params[4],
        'seed': task_params[5],
        'learning_rate': task_params[6],
        'hidden_dim': task_params[7],
        'val_num': task_params[8],
        'subset_num': task_params[9],
        'step_size': task_params[10],
        'gamma': task_params[11],
        'early_stop_epoch': task_params[12],
        'D_count': task_params[13],
        'num_heads': task_params[14] if len(task_params) > 14 else None,
        'semi_supervise': False,
        'save_attention': False,
        'create': False
    })

    args['protein_name'], args['assay_type'], args['pdb_name'], args['pocket_num'], args['num_pose'] = load_task_config(task_id)
    data_root = fetch_data(args['protein_name'])
    
    logger = setup_training_environment(args, args['graph_type'], task_id, args['num_pose'], args['seed'])
    
    train_set_list_all, train_loader_list_all = prepare_data_loaders(args, args['graph_type'], task_id, data_root, ground_root, args['num_pose'], args['create'], args['semi_supervise'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, scheduler = setup_model_and_training(args, device)

    logger.info("Starting training...")
    trainer = DTIGNTrainer(
        model, optimizer, scheduler, train_loader_list_all,
        epochs=args['epochs'], logger=logger,
        batch_size=args['batch_size'], early_stop_epoch=args['early_stop_epoch']
    )
    trainer.train()
    logger.info("Training complete.")
