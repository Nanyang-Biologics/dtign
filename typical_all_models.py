import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from copy import deepcopy
from dgllife.utils import Meter, EarlyStopping
from hyperopt import fmin, tpe
from shutil import copyfile
from torch.optim import Adam
from torch.utils.data import DataLoader

# from hyper import init_hyper_space
from utils_unified import get_configure, get_configure_from_results, mkdir_p, init_trial_path, \
    split_dataset, collate_molgraphs, load_model, load_model_from_config, load_GCN_L_model, predict, init_featurizer, load_dataset
    
import scipy
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter

tasks = ['value']
def topk_acc2(df, predict, k, active_num):
    df['predict'] = predict
    df2 = df.sort_values(by='predict',ascending=False) # 拼接预测值后对预测值进行排序
#     print('df2:\n',df2)
    
    df3 = df2[:k]  #取按预测值排完序后的前k个
    
    true_sort = df.sort_values(by=tasks[0],ascending=False) #返回一个新的按真实值排序列表
    k_true = true_sort[tasks[0]].values[k-1]  # 真实排第k个的活性值
#     print('df3:\n',df3['predict'])
#     print('k_true: ',type(k_true),k_true)
#     print('k_true: ',k_true,'min_predict: ',df3['predict'].values[-1],'index: ',df3['predict'].values>=k_true,'acc_num: ',len(df3[df3['predict'].values>=k_true]),
#           'fp_num: ',len(df3[df3['predict'].values>=-4.1]),'k: ',k)
    acc = len(df3[df3[tasks[0]].values>=k_true])/k #预测值前k个中真实排在前k个的个数/k
    fp = len(df3[df3[tasks[0]].values==-4.1])/k  #预测值前k个中为-4.1的个数/k
    if k>active_num:
        min_active = true_sort[tasks[0]].values[active_num-1]
        acc = len(df3[df3[tasks[0]].values>=min_active])/k
    
    return acc,fp
    
def topk_recall(df, predict, k,test_active):
    df['predict'] = predict
    df2 = df.sort_values(by='predict',ascending=False) # 拼接预测值后对预测值进行排序
#     print('df2:\n',df2)
        
    df3 = df2[:k]  #取按预测值排完序后的前active_num个，因为后面的全是-4.1
    
    true_sort = df.sort_values(by='value',ascending=False) #返回一个新的按真实值排序列表
    min_active = true_sort['value'].values[test_active-1]  # 真实排第k个的活性值
#     print('df3:\n',df3['predict'])
#     print('min_active: ',type(min_active),min_active)
#     print('min_active: ',min_active,'min_predict: ',df3['predict'].values[-1],'index: ',df3['predict'].values>=min_active,'acc_num: ',len(df3[df3['predict'].values>=min_active]),
#           'fp_num: ',len(df3[df3['predict'].values>=-4.1]),'k: ',k,'test_active: ',test_active)
    acc = len(df3[df3['value'].values>=min_active])/test_active #预测值前k个中真实排在前test_active个的个数/test_active
    fp = len(df3[df3['value'].values==-4.1])/k  #预测值前k个中为-4.1的个数/test_active
    
    #if(show_flag):
        #进来的是按实际活性值排好序的
    #sorted_show_pik(true_sort,true_sort['predict'],k,k_predict,acc)
    return acc,fp

    
def topk_acc_recall(df, predict, k, test_active):
    if k>test_active:
        return topk_recall(df, predict, k,test_active)
    return topk_acc2(df,predict,k,test_active)

#    print("acc=".format(acc))
#    print("num/k=".format(num / k))

def AP(df, predict, active_num):
    prec = []
    rec = []
    for k in np.arange(1,len(df)+1,1):
        prec_k, fp1 = topk_acc2(df,predict,k, active_num)
        rec_k, fp2 = topk_recall(df, predict, k, active_num)
        prec.append(prec_k)
        rec.append(rec_k)

    # 取所有不同的recall对应的点处的精度值做平均
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # 计算包络线，从后往前取最大保证precise非减
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 找出所有检测结果中recall不同的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 用recall的间隔对精度作加权平均
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
   
def standardize(exp_config, x, reverse=True):
    return x * (exp_config['std']+1e-9) + exp_config['mean'] if reverse else (x - exp_config['mean'])/(exp_config['std']+1e-9) 

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    batch_step = len(data_loader)
    global_step = epoch * batch_step
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        # print(len(smiles), '/', len(data_loader.dataset))
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            print("Please change the batch size to avoid {}%({}//4) = 1.".format(len(data_loader.dataset), exp_config['batch_size']))
            continue

        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = predict(args, model, bg)
        prediction = standardize(exp_config, prediction, reverse=True)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        train_meter.update(prediction, labels, masks)
        if log:
            logger.add_scalar('loss/label_loss', loss, global_step)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    total_score = np.mean(train_meter.compute_metric(args['metric']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], total_score))


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = predict(args, model, bg)
            prediction = standardize(exp_config, prediction, reverse=True)
            eval_meter.update(prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric(args['metric']))
    return total_score
    
def caculate_r2(y,predict):
#     print(y)
#     print(predict)
#    y = torch.FloatTensor(y).reshape(-1,1)
#    predict = torch.FloatTensor(predict).reshape(-1,1)
    y_mean = torch.mean(y)
    predict_mean = torch.mean(predict)
    
    y1 = torch.pow(torch.mm((y-y_mean).t(),(predict-predict_mean)),2)
    y2 = torch.mm((y-y_mean).t(),(y-y_mean))*torch.mm((predict-predict_mean).t(),(predict-predict_mean))
#     print(y1,y2)
    return (y1/y2).item()

def flatten(x_list):
    res = []
    for x in x_list:
        res.extend(x)
    return res

def AUC(labels,preds,n_bins=10000):
    m = sum(labels)
    n = len(labels) - m
    total_case = m * n
    
    pos = [0 for _ in range(n_bins)]
    neg = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int((preds[i]-1e-6)/bin_width)
        if labels[i]==1:
            pos[nth_bin] += 1
        else:
            neg[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos[i]*accumulated_neg + pos[i]*neg[i]*0.5)
        accumulated_neg += neg[i]
    return satisfied_pair / total_case
    
def run_an_test_epoch(args, model, data_loader,test_df):
    model.eval()
    eval_meter = Meter()
    total_predict = []
    total_smiles = []
    total_label = []
   
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = predict(args, model, bg)
            prediction = standardize(exp_config, prediction, reverse=True)
            total_predict.extend(prediction.cpu().numpy())
            total_label.extend(labels.cpu().numpy())
            total_smiles.extend(smiles)
            #print(type(prediction),prediction.shape)
            eval_meter.update(prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric(args['metric']))  # np.mean()是因为源代码有多任务的
        y_pred = torch.stack(flatten(eval_meter.y_pred),0)
        y_true = torch.stack(flatten(eval_meter.y_true),0)
        df = pd.DataFrame([])
        df['smiles'] = total_smiles
        df['label'] = flatten(total_label)
        df['prediction'] = flatten(total_predict)
        # print(y_pred.shape,y_true.shape,df.shape)
        if not os.path.exists(args['result_path']):
            os.makedirs(args['result_path'])
        df.to_csv(args['result_path'] + '/prediction.csv',index=False)
        
        tau,p_value = scipy.stats.kendalltau(y_pred,y_true)
        # print("tau:",tau)
        active_num = len(df)
        active_y_pred = y_pred[:active_num]
        active_y_true = y_true[:active_num]
        # print(active_y_pred.shape,active_y_true.shape)
        active_r2 = caculate_r2(active_y_pred,active_y_true)
        active_rmse = torch.sqrt(F.mse_loss(active_y_pred, active_y_true).cpu()).item()
        
        labels = np.append(np.ones(active_num),np.zeros(y_pred.shape[0]-active_num))
        # binary_y_pred = np.ones(y_pred.shape)
        # binary_y_pred[y_pred<-4]=0
        # auc = AUC(labels,binary_y_pred,n_bins=10000)
        auc = 0
        
    topk = []
    false_positive_rate = []
    for k in [int(len(total_predict)*0.01),int(len(total_predict)*0.03),int(len(total_predict)*0.05),int(len(total_predict)*0.1),int(len(total_predict)*0.25),int(len(total_predict)*0.5),1,3,5,10,25,100]:
        if k > len(test_df):
            topk.append(1)
            false_positive_rate.append(0)
            continue
        if k < 1:
            topk.append(None)
            false_positive_rate.append(None)
            continue
        a,b = topk_acc_recall(test_df, total_predict, k, len(test_df))
        topk.append(round(a,4))
        false_positive_rate.append(round(b,4))
        #ap = AP(test_df, total_predict, args['test_activte_num'])
    return topk, false_positive_rate, active_r2, active_rmse, tau, auc
    
def main(args, exp_config, train_set, val_set, test_set, model_size_level):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    # Set up directory for saving results
    # args = init_trial_path(args)

    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = load_model(exp_config, model_size_level).to(args['device'])
    model_pare_num = sum([param.nelement() for param in model.parameters()])
    ######
    loss_criterion = nn.MSELoss(reduction='none')
    #loss_criterion = nn.MSELoss()
    #loss_criterion = nn.SmoothL1Loss(reduction='none')###
    #loss_criterion = nn.SmoothL1Loss()

    #loss_criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])
    stopper = EarlyStopping(mode=args['early_stop_mode'],
                            patience=exp_config['patience'],
                            filename= args['result_path'] + '/model.pth')
    batch_step = len(train_loader)
    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        global_step = (epoch+1) * batch_step
        if log:
            logger.add_scalar('val/{}'.format(args['metric']), val_score, global_step)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'], val_score,
            args['metric'], stopper.best_score))

        if early_stop:
            break
    
    stopper.load_checkpoint(model)
    args['best_val_score_in_k'] = stopper.best_score
    topk, false_positive_rate, active_r2, active_rmse, tau, auc= run_an_test_epoch(args, model, test_loader,test_df)
    result_name_list = ['Val_stop_{}'.format(args['metric']),'Top-1%','Top-3%','Top-5%','Top-10%','Top-25%','Top-50%','Top-1','Top-3','Top-5','Top-10','Top-25','Top-100','R','RMSE','Tau-B','Test_num','Model_Para','Train_mean','Train_std']
    result_value_list = [stopper.best_score]
    result_value_list.extend(topk)
    result_value_list.append(round(active_r2**0.5,4))
    result_value_list.append(round(active_rmse,4))
    result_value_list.append(round(tau,4))
    result_value_list.append(len(test_df))
    result_value_list.append(model_pare_num)
    result_value_list.append(exp_config['mean'])
    result_value_list.append(exp_config['std'])
    print("----------------")
    print("Model saving path:",args['result_path'])
    print(args['model'])
#    print('test {} {:.4f}'.format(args['metric'], test_score))
#    print("ap:",ap)
    result_df = pd.DataFrame([])
    for i in range(len(result_name_list)):
        print(result_name_list[i]+': '+f'{result_value_list[i]}')
        result_df[result_name_list[i]] = [result_value_list[i]]
        if log and result_value_list[i] is not None:
            logger.add_scalar(f'test/{result_name_list[i]}', result_value_list[i], global_step)
    
    result_df.to_csv(path_or_buf = args['result_path'] + '/metrics.csv', index=False)
    with open(args['result_path'] + '/configure.json', 'w') as f:
        json.dump(exp_config, f, indent=2)
        
    print('Results are successfully saved at ' + args['result_path'] + '/metrics.csv')

    return args['result_path'], stopper.best_score

def bayesian_optimization(args, train_set, val_set, test_set):
    # Run grid search
    results = []
    candidate_hypers = init_hyper_space(args['model'])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams, train_set, val_set, test_set)

        if args['metric'] in ['r2']:
            # Maximize R2 is equivalent to minimize the negative of it
            val_metric_to_minimize = -1 * val_metric
        else:
            val_metric_to_minimize = val_metric

        results.append((trial_path, val_metric_to_minimize))

        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_trial_path



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('(Multitask) Regression')
    parser.add_argument('-train', '--train-csv-path', type=str, required=False,
                        help='Path to a csv file for loading a train dataset')
    parser.add_argument('-test', '--test-csv-path', type=str, required=False,
                        help='Path to a csv file for loading a test dataset')
    parser.add_argument('-active', '--test_activte_num', type=int, required=False,default=200,
                        help='activte number of test dataset')
    parser.add_argument('-sc', '--smiles-column', type=str, required=False,default='smiles',
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-lv', '--log-values', action='store_true', default=False,
                        help='Whether to take logarithm of the labels for modeling')
    parser.add_argument('-t', '--task-names', default='value', type=str,
                        help='Header for the tasks to model. If None, we will model '
                             'all the columns except for the smiles_column in the CSV file. '
                             '(default: None)')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='2/3,1/6,1/6', type=str,
                        help='Proportion of the dataset used for training, validation and test '
                             '(default: 2/3,1/6,1/6)')
    parser.add_argument('-me', '--metric', choices=['r2', 'mae', 'rmse'], default='rmse',
                        help='Metric for evaluation (default: r2)')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
                                                   'gin_supervised_contextpred',
                                                   'gin_supervised_infomax',
                                                   'gin_supervised_edgepred',
                                                   'gin_supervised_masking','NF'],
                        default='GCN', help='Model to use (default: GCN)')
    parser.add_argument('-a', '--atom-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for atoms (default: canonical)')
    parser.add_argument('-b', '--bond-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for bonds (default: canonical)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs allowed for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 300)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-p', '--result-prefix', type=str, default='../Results',
                        help='Path to save training results (default: Results)')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='Number of trials for hyperparameter search (default: None)')
    parser.add_argument('-gpu', '--gpu', type=int, default=0,
                        help='select gpu (default: 0)')
    parser.add_argument('-tr', '--trial-num', type=str, default='1',
                        help='Trial num (default: 1)')                   
    args = parser.parse_args().__dict__
    if torch.cuda.is_available():
        if args['gpu']==0:
            args['device'] = torch.device('cuda:0')
        else:
            args['device'] = torch.device('cuda:1')
    else:
        args['device'] = torch.device('cpu')


    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',')

    if args['metric'] == 'r2':
        args['early_stop_mode'] = 'higher'
    else:
        args['early_stop_mode'] = 'lower'
    print('Early_stop_mode:', args['early_stop_mode'])
    
    # model_list = ['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
    #                'gin_supervised_contextpred', 'NF'] # python typical_all_models.py
    
    model_list = ['GCN', 'AttentiveFP']
    # model_size_dict = {'GCN': 8, 'GAT': 3, 'Weave': 2, 'MPNN': 2, 'AttentiveFP': 5, 'NF': 6, 'gin_supervised_contextpred': 1}
    model_size_dict = {'GCN': 2, 'AttentiveFP': 1}
    dataset_perfix = './data/'
    args['result_prefix'] = './baseline/'
    # task_dict = {0: ('CHEMBL202', 'pIC50'), 1: ('CHEMBL235', 'pEC50'), 2: ('CHEMBL279', 'pIC50'), 3: ('CHEMBL2971', 'pIC50'), 4: ('CHEMBL333', 'pIC50'), 5: ('CHEMBL3820', 'pEC50'), 6: ('CHEMBL3976', 'pIC50'), 7: ('CHEMBL4422', 'pEC50')}
    task_dict = {0: ('CHEMBL202', 'pIC50'), 1: ('CHEMBL279', 'pIC50'), 2: ('CHEMBL2971', 'pIC50'), 3: ('CHEMBL333', 'pIC50'), 4: ('CHEMBL3820', 'pEC50'), 5: ('CHEMBL3976', 'pIC50'), 6: ('CHEMBL4422', 'pEC50')}
    new_column_names =new_column_names = {'pIC50': 'value', 'SMILES': 'smiles'}
    log = True
    # TODO: change random_seed between 0,1,2,3,4, before 999
    random_seed = 0
    max_val_num = 200
    for model in model_list:
        args['model'] = model
        if args['model'] == 'AttentiveFP':
            args['atom_featurizer_type'] = 'attentivefp'
            args['bond_featurizer_type'] = 'attentivefp'
            print('Using attentivefp atom and bond features.')
        model_size_level = model_size_dict[model]
        args['trial_num'] = 'S' + str(model_size_level) # model size
        trial_num = args['trial_num']
        print(args['atom_featurizer_type'])
        args = init_featurizer(args)
        print('model:', model, 'model_size_level:', model_size_level, 'trial_num:', trial_num)
        for task_id in tqdm(range(len(task_dict))):
            target_id, assay_type = task_dict[task_id]
            args['target_id'], args['assay_type'] = target_id, assay_type
            dataset_path = dataset_perfix + target_id +'/'
            args['result_num'] = 0
            print('Reading: ' + dataset_path + assay_type + '/' + target_id + '_' + assay_type + '_test.csv')
            test_df = pd.read_csv(dataset_path + assay_type + '/' + target_id + '_' + assay_type + '_test.csv')
            test_df = test_df.rename(columns=new_column_names)
            args['test_activte_num'] = len(test_df)
            if args['early_stop_mode'] == 'higher':
                args['best_val_score_across_k'] = 0
            if args['early_stop_mode'] == 'lower':
                args['best_val_score_across_k'] = 1e6
            k_fold = 5 #if len(test_df) < max_val_num else 1
            for k in range(k_fold):
                args['result_path'] = args['result_prefix'] + '/' + target_id + '/' + assay_type + '/' + args['model'] + '/' + str(trial_num) + '/' + str(k+1)
                if not os.path.exists(args['result_path']):
                    os.makedirs(args['result_path'])
                if os.path.exists(args['result_path'] + '/metrics.csv'):
                    print('Completed: '+args['result_path'] + '/metrics.csv')
                    continue
                print('Starting the {}/{} cross-validation for {}/{} ...'.format(k+1, k_fold, target_id, assay_type))
                args['result_num'] += 1
                train_df = pd.DataFrame([])
                for k_trian in range(5):
                    if k_trian != k:
                        k_trian_df = pd.read_csv(dataset_path + assay_type + '/' + target_id + '_' + assay_type + f'_train_{k_trian+1}.csv')
                        k_trian_df = k_trian_df.rename(columns=new_column_names)
                        train_df = pd.concat([train_df, k_trian_df],  ignore_index=True)
                sample_threshold = 10000
                # if len(train_df) > sample_threshold:
                #     print(f'Skip large dataset (>{sample_threshold}): ' + dataset_path + assay_type + '/' + target_id + '_' + assay_type)
                #     continue
                val_df = pd.read_csv(dataset_path + assay_type + '/' + target_id + '_' + assay_type + f'_train_{k+1}.csv')
                val_df = val_df.rename(columns=new_column_names)
#                 if len(val_df) > max_val_num:
#                     val_df_initial = val_df
#                     val_df = val_df_initial.sample(max_val_num, random_state=random_seed)
#                     val_df_rest = val_df_initial.drop(val_df.index)
#                     val_df = val_df.reset_index(drop=True)
#                     train_df = pd.concat([train_df, val_df_rest],  ignore_index=True)
                train_set = load_dataset(args, train_df)
                val_set = load_dataset(args, val_df)
                test_set = load_dataset(args, test_df)
                # Whether to take the logarithm of labels for narrowing the range of values
                if args['log_values']:
                    train_set.labels = train_set.labels.log()
                args['n_tasks'] = train_set.n_tasks
                #train_set, val_set, test_set = split_dataset(args, dataset)

                if args['num_evals'] is not None:
                    assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                                  'be greater than 0, got {:d}'.format(args['num_evals'])
                    print('Start hyperparameter search with Bayesian '
                          'optimization for {:d} trials'.format(args['num_evals']))
                    trial_path = bayesian_optimization(args, train_set, val_set, test_set)
                else:
                    print('Use the manually specified hyperparameters')

                exp_config = get_configure(args['model'])
                exp_config['mean'] = np.mean(train_df.value.values)
                exp_config['std'] = np.std(train_df.value.values)
                if log:
                    log_dir = args['result_path'] + '/log/'
                    # if os.path.isdir(log_dir):
                    #     for files in os.listdir(log_dir):
                    #         os.remove(log_dir+files)
                    #     os.rmdir(log_dir)
                    logger = SummaryWriter(log_dir)
#                 main(args, exp_config, train_set, val_set, test_set, model_size_level)
                try:
                    main(args, exp_config, train_set, val_set, test_set, model_size_level)
                except Exception as e:
                    with open(args['result_path'] + '/Error.txt', 'w') as f:
                        f.write(str(e))
                    continue
                if args['early_stop_mode'] == 'higher':
                    if args['best_val_score_in_k'] < args['best_val_score_across_k']:
                        os.remove(args['result_path'] + '/model.pth')
                    elif k == 0:
                        args['best_val_score_across_k'] = args['best_val_score_in_k']
                    else:
                        args['best_val_score_across_k'] = args['best_val_score_in_k']
                        for k_ in range(k):
                            if os.path.exists(args['result_prefix'] + '/' + target_id + '/' + assay_type + '/' + args['model'] + '/' + str(trial_num) + f'/{k_+1}/model.pth'):
                                os.remove(args['result_prefix'] + '/' + target_id + '/' + assay_type + '/' + args['model'] + '/' + str(trial_num) + f'/{k_+1}/model.pth')

                if args['early_stop_mode'] == 'lower':
                    print(args['best_val_score_in_k'], ':', args['best_val_score_across_k'])
                    if args['best_val_score_in_k'] > args['best_val_score_across_k']:
                        print('Remove ' + args['result_path'] + '/model.pth')
                        os.remove(args['result_path'] + '/model.pth')
                    elif k == 0:
                        args['best_val_score_across_k'] = args['best_val_score_in_k']
                    else:
                        args['best_val_score_across_k'] = args['best_val_score_in_k']
                        for k_ in range(k):
                            if os.path.exists(args['result_prefix'] + '/' + target_id + '/' + assay_type + '/' + args['model'] + '/' + str(trial_num) + f'/{k_+1}/model.pth'):
                                os.remove(args['result_prefix'] + '/' + target_id + '/' + assay_type + '/' + args['model'] + '/' + str(trial_num) + f'/{k_+1}/model.pth')

                if os.path.exists(args['result_path'] + '/graph.bin'):
                    os.remove(args['result_path'] + '/graph.bin')
