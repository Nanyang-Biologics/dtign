
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from itertools import zip_longest, cycle
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Batch
from tqdm import tqdm
import scipy, math, re
# %%

def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def val(model, dataloader, device):
    model.eval()
    predict_list, label_list = [], []
    predict_dict, label_dict = {}, {}
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            pred = standardize(pred, mean, std, device, reverse=True)
            label = data.y
            pocket, idxes = data.pocket, data.idx
            idx_list = flatten(idxes)
            for i, idx in enumerate(idx_list):
                p= pocket[i] - 1
                if idx not in predict_dict:
                    predict_dict[idx] = []  
                else:
                    predict_dict[idx].append(pred[i].detach().gpu().item())
                if idx not in label_dict:
                    label_dict[idx] = []  
                else:
                    label_dict[idx].append(label[i].detach().gpu().item())
                    
    for key, pred in predict_dict.items():
        mean_pred = sum(pred) / len(pred)
        predict_list.append(mean_pred)
        label_item = label_dict[key]
        mean_label = sum(label_item) / len(label_item)
        label_list.append(mean_label)
        
    pred = np.array(predict_list)
    label = np.array(label_list)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    tau, p_value = scipy.stats.kendalltau(pred, label)

    model.train()

    return rmse, coff, tau

def standardize(x, mean, std, device, reverse=True):
    mean, std = torch.tensor(mean).to(device), torch.tensor(std).to(device)
    return x * (std+1e-9) + mean if reverse else (x - mean)/(std+1e-9) 

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    folds = args.get("fold")
    valid_metric = args.get("valid_metric")
    args['mark'] = 'eval'
    
    args['start_checkpoint'] = './model/20230614_160105_GIGN_fold_2_MCDO/model/Current, epoch-53, mean-5.7254, std-1.2432, train_loss-34.4721, train_rmse-5.8713, valid_rmse-4.6876, valid_pr--0.1050, valid_tau--0.0900.pt'
    match = re.search(r'fold_(\d+)', args['start_checkpoint'])
    args['fold'] = int(match.group(1))
    match = re.search(r'mean-(\d+\.\d+)', args['start_checkpoint'])
    mean = float(match.group(1))
    match = re.search(r'std-(\d+\.\d+)', args['start_checkpoint'])
    std = float(match.group(1))
    logger = TrainLogger(args, cfg, create=True)
    logger.info(__file__)
    logger.info(f"model path: {args['start_checkpoint']}")
    logger.info(f"fold: {args['fold']}")
    
    valid_dir = os.path.join(data_root, '1boz', f'train_{args["fold"]}')
    valid_df = pd.read_csv(os.path.join(data_root, f'CHEMBL202_pIC50_train_{args["fold"]}.csv'))
    valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dir = os.path.join(data_root, '1boz', 'test')
    test_df = pd.read_csv(os.path.join(data_root, 'CHEMBL202_pIC50_test.csv'))
    test_set = GraphDataset(test_dir, test_df, graph_type=graph_type, create=False)
    test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"valid data: {len(valid_set)}")
    logger.info(f"test data: {len(test_set)}")

    device = torch.device('cuda:0')
    model = GIGN(35, 256).to(device)
    
    # final testing
    
    load_model_dict(model, args['start_checkpoint'])
    model.eval()
    valid_rmse, valid_pr, valid_tau = val(model, valid_loader, device)
    test_rmse, test_pr, test_tau = val(model, test_loader, device)
    msg = "valid_rmse-%.4f, valid_pr-%.4f, test_rmse-%.4f, test_pr-%.4f, test_tau-%.4f"%(valid_rmse, valid_pr, test_rmse, test_pr, test_tau)

    logger.info(msg)
        

# %%