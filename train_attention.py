
# %%
import os, re
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from aws import S3DataFetcher
from sklearn.metrics import mean_squared_error
from itertools import zip_longest, cycle
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Batch
from tqdm import tqdm
import scipy, math, json, shutil, pickle, time
import torch.nn.functional as F
import scipy.stats as stats
from torch.optim.lr_scheduler import StepLR

def reduction_by_pocket(tensor, pocket_list, device, semi_supervise=False):
    output_tensor, current_index, tensor_shape = torch.tensor([]).to(device), 0, tensor.shape[0]
    for pockets in pocket_list:
        additional_num = 0
        for pocket in pockets:
            if pocket.startswith('G'):
                additional_num += 1
        pocket_num = len(pockets)
        reducted_tensor = tensor[current_index:current_index+pocket_num].mean(dim=0, keepdim=True)
        output_tensor = torch.cat((output_tensor, reducted_tensor), dim=0)
        if semi_supervise:
            for i in range(additional_num):
                output_tensor = torch.cat((output_tensor, reducted_tensor), dim=0)
        current_index += pocket_num
    
    return output_tensor

def val(model, dataloader, device, epoch, attention_dict, save_attention=False, val_rate=1):
    model.eval()
    predict_list, label_list = [], []
    predict_dict, label_dict = {}, {}
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            label, pocket, idxes = data.y, data.pocket, data.idx
            pred, outputs = model(data=data, pocket_list=pocket, self_attention=True, semi_supervise=semi_supervise, save_attention=save_attention, graph_type=graph_type)
            pocket_supervised_loss = outputs[0]
            if save_attention:
                attention_weights = outputs[1]
                for i, idx in enumerate(idxes):
                    if epoch == 0:
                        attention_dict[idx] = [attention_weights[i]]
                    else:
                        attention_dict[idx].append(attention_weights[i])
            pred = standardize(pred, weighted_mean, weighted_std, device, reverse=True)
            label = reduction_by_pocket(label, pocket, device, semi_supervise=semi_supervise)
            idx_list = flatten(idxes)
            for i, idx in enumerate(idx_list):
                predict_dict[idx] = [] if idx not in predict_dict else predict_dict[idx]
                label_dict[idx] = [] if idx not in label_dict else label_dict[idx]
                predict_dict[idx].append(pred[i].detach().cpu())
                label_dict[idx].append(label[i].detach().cpu())
           
    for key, pred in predict_dict.items():
        mean_pred = sum(pred) / len(pred)
        predict_list.append(mean_pred)
        label_item = label_dict[key]
        mean_label = sum(label_item) / len(label_item)
        label_list.append(mean_label)
        
    pred = np.array(predict_list)
    label = np.array(label_list)

    # 计算要抽取的样本数量（50%的索引数）
    num_samples_to_extract = int(len(pred) * val_rate)
    # 使用 numpy.random.choice 函数随机抽取索引
    random_indices = np.random.choice(len(pred), num_samples_to_extract, replace=False)
    # 根据随机抽取的索引，从 pred 数组中取出对应元素组成一个新的 NumPy 数组
    pred = pred[random_indices]
    label = label[random_indices]
    
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    tau, p_value = scipy.stats.kendalltau(pred, label)

    model.train()

    return rmse, coff, tau, attention_dict

def delta_weight(x, c):
    if x == 0:
        return 1
    elif x < c:
        return 1 - (1 - ((x-c)/c)**2 )**0.5
    elif x == c: 
        return 0
    elif x < 2*c: 
        return (1 - ((x-c)/c)**2 )**0.5 - 1
    elif x >= 2*c: 
        return -1

def standardize(x, mean, std, device, reverse=True):
    mean, std = torch.tensor(mean).to(device), torch.tensor(std).to(device)
    return x * (std+1e-9) + mean if reverse else (x - mean)/(std+1e-9) 

def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified settings.')
    parser.add_argument('--setting', type=str, required=True, help='The setting for the training session')
    # Parse the arguments
    arguments = parser.parse_args()
    # Print the received setting
    print(f"Training model with setting: {arguments.setting}")
    cfg = 'TrainConfig_GIGN'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    total_batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    ground_root = args.get('ground_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    folds = args.get("fold")
    dropout = args.get("dropout")
    valid_metric = args.get("valid_metric")
    ### Experimental settings ###
    task_id = f"{arguments.setting}"
    if task_id == 'I1':
        start_fold, skip_fold, stop_fold, warmup_epoch, val_rate, seed, learning_rate = 1, [], 5, 40, 1, 0, 1e-4
    if task_id == 'I2':
        start_fold, skip_fold, stop_fold, warmup_epoch, val_rate, seed, learning_rate = 1, [], 5, 40, 1, 0, 1e-4
    if task_id == 'I5':
        start_fold, skip_fold, stop_fold, warmup_epoch, val_rate, seed, learning_rate = 1, [4], 5, 40, 1, 0, 1e-4
    if task_id == 'E2':
        start_fold, skip_fold, stop_fold, warmup_epoch, val_rate, seed, learning_rate = 1, [], 5, 40, 1, 0, 1e-4
    if task_id == 'E3':
        start_fold, skip_fold, stop_fold, warmup_epoch, val_rate, seed, learning_rate = 1, [], 1, 40, 1, 0, 1e-4
    args['start_checkpoint'] = None
    semi_supervise = False
    save_attention = False
    create = False
    task_dict = {'I1': ('CHEMBL202', 'pIC50', '1boz', 7, 1), 'E3': ('CHEMBL235', 'pEC50', '1zgy', 4, 6), 
                 'I5': ('CHEMBL279', 'pIC50', '1ywn',3, 4), 'I4': ('CHEMBL2971', 'pIC50', '3ugc', 3, 5), 
                 'I3': ('CHEMBL333', 'pIC50', '1ck7', 6, 1), 'E1': ('CHEMBL3820', 'pEC50', '3f9m', 6 ,2), 
                 'I2': ('CHEMBL3976', 'pIC50', '4ebb', 2, 4), 'E2': ('CHEMBL4422', 'pEC50', '5tzr', 3, 2)}
    protein_name, assay_type, pdb_name, pocket_num, num_pose = task_dict[task_id]
    graph_type = ['Graph_GIGN', 'Graph_Bond'][1]
    args['mark'] = f'{task_id}_{graph_type}_Pose_{num_pose}_Seed_{seed}_Attention_physics_D_count=64_rbf_power_reverse'
    GIGN_datafold = '/home/yueming/Drug_Discovery/Baselines/GIGN-main/GIGN'
    #############################
    batch_size = round(0.25 * total_batch_size/(pocket_num * num_pose)) # 4 training subsets
    if semi_supervise:
        args['mark'] += '_semi_sum'
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args['save_dir'] = args.get('save_dir') + '/' + timestamp + '_' + args['mark'] + '/'
#     args['save_dir'] = '/home/yueming/Drug_Discovery/Baselines/GIGN-main/GIGN/model/20230727_200655_I1_Graph_Bond_Pose_1_Seed_0_Attention_linear_physics*/'
    data_root = f'{data_root}/{protein_name}'
    target_pdb = f'{pdb_name}/vina'
    torch.manual_seed(seed)
    np.random.seed(seed)
#     args['start_checkpoint'] = "./model/20230613_102821_GIGN_fold_2_reweight/model/Best, epoch-41, mean-5.7254, std-1.2432, train_loss-0.2112, train_rmse-0.4596, valid_rmse-1.0741, valid_pr-0.3733, valid_tau-0.2560.pt"
#     args['best_checkpoint'] = "./model/20230613_102821_GIGN_fold_2_reweight/model/Best, epoch-41, mean-5.7254, std-1.2432, train_loss-0.2112, train_rmse-0.4596, valid_rmse-1.0741, valid_pr-0.3733, valid_tau-0.2560.pt"
    if graph_type == 'Graph_Bond':
        if semi_supervise:
            from dataset_GIGN_semi_supervised_bond import GraphDataset, PLIDataLoader
        else:
            from dataset_GIGN_packaged_bond import GraphDataset, PLIDataLoader
    elif semi_supervise:
        from dataset_GIGN_semi_supervised import GraphDataset, PLIDataLoader
    else:
        from dataset_GIGN_packaged import GraphDataset, PLIDataLoader
    
    train_set_list_all, train_loader_list_all = [], []
    print("len folds: ", folds)
    for fold in range(folds):
        train_dir = os.path.join(data_root, target_pdb, f'train_{fold+1}')
        ground_dir = os.path.join(ground_root, f'train_{fold+1}')
        train_df = pd.read_csv(os.path.join(data_root, target_pdb, f'{protein_name}_{assay_type}_train_{fold+1}.csv'))
        train_set = GraphDataset(train_dir, ground_dir, train_df, GIGN_datafold=GIGN_datafold, num_pose=num_pose, graph_type=graph_type, assay_type=assay_type, create=create) if semi_supervise else GraphDataset(train_dir, train_df, GIGN_datafold=GIGN_datafold, num_pose=num_pose, graph_type=graph_type, assay_type=assay_type, create=create)
        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        train_set_list_all.append(train_set)
        train_loader_list_all.append(train_loader)
       
    test_dir = os.path.join(data_root, target_pdb, 'test')
    ground_dir = os.path.join(ground_root, 'test')
    test_df = pd.read_csv(os.path.join(data_root, target_pdb, f'{protein_name}_{assay_type}_test.csv'))
    test_set = GraphDataset(test_dir, ground_dir, test_df, GIGN_datafold=GIGN_datafold, num_pose=num_pose, graph_type=graph_type, assay_type=assay_type, create=create) if semi_supervise else GraphDataset(test_dir, test_df, GIGN_datafold=GIGN_datafold, num_pose=num_pose, graph_type=graph_type, assay_type=assay_type, create=create)
    test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.MSELoss(reduction='none')
    
    for fold in range(folds):
        args['fold'] = fold + start_fold
        if args['fold'] > stop_fold:
            break
        if args['fold'] in skip_fold:
            continue
        train_set_list, train_loader_list = train_set_list_all.copy(), train_loader_list_all.copy()
        valid_set, valid_loader = train_set_list[args['fold']-1], train_loader_list[args['fold']-1]
        train_set_list.pop(args['fold']-1)
        train_loader_list.pop(args['fold']-1)
        train_set1, train_set2, train_set3, train_set4 = tuple(train_set_list)
        train_loader1, train_loader2, train_loader3, train_loader4 = tuple(train_loader_list)
        
        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"fold: {args['fold']}")
        logger.info(f"train data: {len(train_set1)+len(train_set2)+len(train_set3)+len(train_set4)}")
        logger.info(f"train data subsets: {len(train_set1)}, {len(train_set2)}, {len(train_set3)}, {len(train_set4)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test data: {len(test_set)}")
        logger.info(f"batch ligands: {4*batch_size}") # 4 training subsets
        logger.info(f"pockets/ligand: {pocket_num}")
        logger.info(f"poses/pocket: {num_pose}")
        logger.info(f"batch samples: {4*batch_size*pocket_num*num_pose}")
        shutil.copy(__file__, logger.get_log_dir())
        result_path = os.path.join(logger.get_model_dir()[:-5], 'result')
        # 获取源文件名
        source_file_list = ["GIGN.py", "HIL.py"]
        for source_file in source_file_list:
            # 构建源文件的完整路径
            source_path = os.path.join(os.path.dirname(__file__), source_file)
            # 复制文件
            shutil.copy(source_path, logger.get_log_dir())
        
        weight_path = os.path.join(logger.get_model_dir()[:-5], 'result')
        best_model_list, current_model_list = [None], [None]
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        best_type = "min" if valid_metric == "rmse" else "max"
        running_best_metric = BestMeter(best_type)
        running_best_metric.reset
        
        device = torch.device('cuda:0')
        model = GIGN(node_dim=35, bond_dim=10, hidden_dim=256, num_pose=num_pose, dropout=dropout, self_attention=True, graph_type=graph_type).to(device)
        if args['start_checkpoint'] is not None:
            best_model_list.append(args['best_checkpoint'])
            load_model_dict(model, args['start_checkpoint'])
            match = re.search(r'epoch-(\d+)', args['start_checkpoint'])
            ckpt_epoch = int(match.group(1))
            start_epoch = ckpt_epoch + 1
            match = re.search(fr'valid_{valid_metric}-(\d+\.\d+)', args['best_checkpoint'])
            valid_metric_value = float(match.group(1))
            running_best_metric.update(valid_metric_value)
            match = re.search(r'epoch-(\d+)', args['best_checkpoint'])
            best_epoch = int(match.group(1))
            if ckpt_epoch != best_epoch:
                current_model_list.append(args['start_checkpoint'])
            count = running_best_metric.counter(interval = ckpt_epoch - best_epoch)
        else:
            start_epoch = 0
            responsible_dict, weight_dict, uncertainty_dict = {}, {}, {}
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10**(-4.3))
        scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
        model.train()
        train_loaders = [train_loader1, train_loader2, train_loader3, train_loader4]
        
        sample_counts = [len(dataset) for dataset in [train_set1, train_set2, train_set3, train_set4]]
        sample_means = [dataset.mean for dataset in [train_set1, train_set2, train_set3, train_set4]]
        sample_stds = [dataset.std for dataset in [train_set1, train_set2, train_set3, train_set4]]
        total_count = sum(sample_counts)
        # 计算加权平均的均值
        weighted_mean = sum(count * mean for count, mean in zip(sample_counts, sample_means)) / total_count
        # 计算加权平均的标准差
        weighted_variance = sum(count * (std ** 2) for count, std in zip(sample_counts, sample_stds)) / total_count
        weighted_std = math.sqrt(weighted_variance)
        logger.info(f"train mean: {'{:.4f}'.format(weighted_mean)}")
        logger.info(f"train std: {'{:.4f}'.format(weighted_std)}")
        logger.info(f"valid_metric: {valid_metric}")
        
        loader_list = [train_loader1, train_loader2, train_loader3, train_loader4]
        train_attention_dict, val_attention_dict, test_attention_dict = {}, {}, {}
        for ep in range(epochs - start_epoch):
            epoch = ep + start_epoch
            for data1, data2, data3, data4 in tqdm(zip_longest(train_loader1, train_loader2, train_loader3, train_loader4)):
                all_data_list = [data1, data2, data3, data4]
                data_list = []
                for i, data_ in enumerate(all_data_list):
                    if not data_:
                        data_iter = iter(loader_list[i])
                        data_ = next(data_iter)
                    data_list.append(data_.to(device))
                merged_data = Batch.from_data_list(data_list)
                label, pockets, idxes = merged_data.y, merged_data.pocket, merged_data.idx
#                 print(pockets)
                idxes = flatten(idxes)
                pocket = []
                for i in range(len(pockets)):
                    pocket += pockets[i]
                label = reduction_by_pocket(label, pocket, device, semi_supervise=semi_supervise)
                pred, outputs = model(data=merged_data, pocket_list=pocket, self_attention=True, semi_supervise=semi_supervise, save_attention=save_attention, graph_type=graph_type)
                pocket_supervised_loss = outputs[0]
                if save_attention:
                    attention_weights = outputs[1]
                    for i, idx in enumerate(idxes):
                        if epoch == 0:
                            train_attention_dict[idx] = [attention_weights[i]]
                        else:
                            train_attention_dict[idx].append(attention_weights[i])
                pred = standardize(pred, weighted_mean, weighted_std, device, reverse=True)
#                 print('pocket_supervised_loss:', pocket_supervised_loss)
                loss = criterion(pred, label).mean() + pocket_supervised_loss if semi_supervise else criterion(pred, label).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 
                
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr, valid_tau, val_attention_dict = val(model, valid_loader, device, epoch, val_attention_dict, save_attention=save_attention, val_rate=val_rate)
            test_rmse, test_pr, test_tau, test_attention_dict = val(model, test_loader, device, epoch, test_attention_dict, save_attention=save_attention)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f, valid_tau-%.4f, test_rmse-%.4f, test_pr-%.4f, test_tau-%.4f, lr-%.4e" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr, valid_tau, test_rmse, test_pr, test_tau, current_lr)
            logger.info(msg)
            
            if save_attention:
                with open(result_path + '/train_attentions.pickle', 'wb') as file:
                    pickle.dump(train_attention_dict, file)
                with open(result_path + '/val_attentions.pickle', 'wb') as file:
                    pickle.dump(val_attention_dict, file)
                with open(result_path + '/test_attentions.pickle', 'wb') as file:
                    pickle.dump(test_attention_dict, file)
            
            valid_metric_value = valid_rmse if best_type == "min" else valid_pr
            best_condition = (valid_metric_value < running_best_metric.get_best()) if best_type == "min" else (valid_metric_value > running_best_metric.get_best())
            msg = "epoch-%d, mean-%.4f, std-%.4f, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f, valid_tau-%.4f, test_rmse-%.4f, test_pr-%.4f, test_tau-%.4f" \
            % (epoch, weighted_mean, weighted_std, epoch_loss, epoch_rmse, valid_rmse, valid_pr, valid_tau, test_rmse, test_pr, test_tau)
            if best_condition and epoch >= warmup_epoch:
                running_best_metric.update(valid_metric_value)
                if save_model:
                    model_path = os.path.join(logger.get_model_dir(), 'Best, ' + msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), 'Best, ' + msg)
                    if best_model_list[-2] is not None:
                        os.remove(best_model_list[-2])
            else:
                current_model_list.append(os.path.join(logger.get_model_dir(), 'Current, ' + msg + '.pt'))
                save_model_dict(model, logger.get_model_dir(), 'Current, ' + msg)
                if current_model_list[-2] is not None:
                    os.remove(current_model_list[-2])
                count = running_best_metric.counter()
                if count > early_stop_epoch:
                    best_metric = running_best_metric.get_best()
                    msg = f"best_{valid_metric}: %.4f" % best_metric
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr, valid_tau, val_attention_dict = val(model, valid_loader, device, epoch, val_attention_dict, save_attention=save_attention)
        test_rmse, test_pr, test_tau, test_attention_dict = val(model, test_loader, device, epoch, test_attention_dict, save_attention=save_attention)
        if save_attention:
            with open(result_path + '/val_attentions.pickle', 'wb') as file:
                pickle.dump(val_attention_dict, file)
            with open(result_path + '/test_attentions.pickle', 'wb') as file:
                pickle.dump(test_attention_dict, file)
        msg = "valid_rmse-%.4f, valid_pr-%.4f, test_rmse-%.4f, test_pr-%.4f, test_tau-%.4f"%(valid_rmse, valid_pr, test_rmse, test_pr, test_tau)

        logger.info(msg)
# %%