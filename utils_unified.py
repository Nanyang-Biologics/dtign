# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import errno
import json
import os
import torch
import torch.nn.functional as F

from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, ScaffoldSplitter, RandomSplitter, mol_to_bigraph
from functools import partial

def init_featurizer(args):
    """Initialize node/edge featurizer

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
        args['atom_featurizer_type'] = 'pre_train'
        args['bond_featurizer_type'] = 'pre_train'
        args['node_featurizer'] = PretrainAtomFeaturizer()
        args['edge_featurizer'] = PretrainBondFeaturizer()
        return args
    elif args['model'] == 'AttentiveFP':
        args['atom_featurizer_type'] = 'attentivefp'
        args['bond_featurizer_type'] = 'attentivefp'
    else:
        args['atom_featurizer_type'] = 'canonical'
        args['bond_featurizer_type'] = 'canonical'

    if args['atom_featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['atom_featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['bond_featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        elif args['bond_featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        args['edge_featurizer'] = None

    return args

def load_dataset(args, df):
    print("df.col: ", df.columns)
    if 'value' not in df.columns:
        df['value'] = df['pEC50']
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'],
                                 smiles_column=args['smiles_column'],
                                 cache_file_path=args['result_path'] + '/graph.bin',
                                 task_names=args['task_names'],
                                 n_jobs=args['num_workers'])

    return dataset

def load_dataset_multi_task(args, df):
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'],
                                 smiles_column=args['smiles_column'],
                                 cache_file_path=args['result_path'] + '/graph.bin',
                                 task_names=args['task_names'],
                                 target_column=args['target_column'],
                                 assay_column=args['assay_column'],
                                 n_jobs=args['num_workers'])

    return dataset

def get_configure(model):
    """Query for the manually specified configuration

    Parameters
    ----------
    model : str
        Model type

    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    with open('model_configures/{}.json'.format(model), 'r') as f:
        config = json.load(f)
    return config

def get_configure_from_results(result_path):
    """Query for the manually specified configuration

    Parameters
    ----------
    model : str
        Model type

    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    with open('{}/configure.json'.format(result_path), 'r') as f:
        config = json.load(f)
    return config

def mkdir_p(path):
    """Create a folder for the given path.

    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def init_trial_path(args):
    """Initialize the path for a hyperparameter setting

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    if path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id) + '/{:d}'.format(args['result_num'])
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args

def split_dataset(args, dataset):
    """Split the dataset

    Parameters
    ----------
    args : dict
        Settings
    dataset
        Dataset instance

    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def collate_molgraphs_multi_task(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks, target, assay = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    target = torch.LongTensor(target)
    assay = torch.LongTensor(assay)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks, target, assay

def collate_molgraphs_unlabeled(data):
    """Batching a list of datapoints without labels

    Parameters
    ----------
    data : list of 2-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES and a DGLGraph.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    """
    smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg

def load_model(exp_configure, model_size_level):
    if exp_configure['model'] == 'GCN':
        from dgllife.model.model_zoo import GCNPredictor
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['gnn_hidden_feats'] = 8
            exp_configure['predictor_hidden_feats'] = 8
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 2:
            exp_configure['gnn_hidden_feats'] = 16
            exp_configure['predictor_hidden_feats'] = 16
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 3:
            exp_configure['gnn_hidden_feats'] = 32
            exp_configure['predictor_hidden_feats'] = 32
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 4:
            exp_configure['gnn_hidden_feats'] = 64
            exp_configure['predictor_hidden_feats'] = 64
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 5:
            exp_configure['gnn_hidden_feats'] = 128
            exp_configure['predictor_hidden_feats'] = 128
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 6:
            exp_configure['gnn_hidden_feats'] = 256
            exp_configure['predictor_hidden_feats'] = 256
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 7:
            exp_configure['gnn_hidden_feats'] = 512
            exp_configure['predictor_hidden_feats'] = 512
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 8:
            exp_configure['gnn_hidden_feats'] = 512
            exp_configure['predictor_hidden_feats'] = 512
            exp_configure['num_gnn_layers'] = 5
        if model_size_level == 9:
            exp_configure['gnn_hidden_feats'] = 1024
            exp_configure['predictor_hidden_feats'] = 1024
            exp_configure['num_gnn_layers'] = 7
        model = GCNPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'GAT':
        from dgllife.model.model_zoo import GATPredictor
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['gnn_hidden_feats'] = 8
            exp_configure['predictor_hidden_feats'] = 8 # initial: 16
            exp_configure['num_gnn_layers'] = 1
            exp_configure['num_heads'] = 1
        if model_size_level == 2:
            exp_configure['gnn_hidden_feats'] = 16
            exp_configure['predictor_hidden_feats'] = 16
            exp_configure['num_gnn_layers'] = 1
            exp_configure['num_heads'] = 2
        if model_size_level == 3:
            exp_configure['gnn_hidden_feats'] = 32
            exp_configure['predictor_hidden_feats'] = 32
            exp_configure['num_gnn_layers'] = 2
            exp_configure['num_heads'] = 4
        if model_size_level == 4:
            exp_configure['gnn_hidden_feats'] = 32
            exp_configure['predictor_hidden_feats'] = 32
            exp_configure['num_gnn_layers'] = 3
            exp_configure['num_heads'] = 6
        if model_size_level == 5:
            exp_configure['gnn_hidden_feats'] = 64
            exp_configure['predictor_hidden_feats'] = 64
            exp_configure['num_gnn_layers'] = 5
            exp_configure['num_heads'] = 8
        if model_size_level == 6:
            exp_configure['gnn_hidden_feats'] = 128
            exp_configure['predictor_hidden_feats'] = 128
            exp_configure['num_gnn_layers'] = 7
            exp_configure['num_heads'] = 12
        if model_size_level == 7:
            exp_configure['gnn_hidden_feats'] = 256
            exp_configure['predictor_hidden_feats'] = 256
            exp_configure['num_gnn_layers'] = 9
            exp_configure['num_heads'] = 16
        if model_size_level == 8:
            exp_configure['gnn_hidden_feats'] = 512
            exp_configure['predictor_hidden_feats'] = 512
            exp_configure['num_gnn_layers'] = 11
            exp_configure['num_heads'] = 20
        model = GATPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
            feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
            residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'Weave':
        from dgllife.model.model_zoo import WeavePredictor
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['gnn_hidden_feats'] = 10
            exp_configure['predictor_hidden_feats'] = 5
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 2:
            exp_configure['gnn_hidden_feats'] = 20
            exp_configure['predictor_hidden_feats'] = 10
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 3:
            exp_configure['gnn_hidden_feats'] = 50
            exp_configure['predictor_hidden_feats'] = 25
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 4:
            exp_configure['gnn_hidden_feats'] = 100
            exp_configure['predictor_hidden_feats'] = 50
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 5:
            exp_configure['gnn_hidden_feats'] = 200
            exp_configure['predictor_hidden_feats'] = 100
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 6:
            exp_configure['gnn_hidden_feats'] = 400
            exp_configure['predictor_hidden_feats'] = 200
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 7:
            exp_configure['gnn_hidden_feats'] = 600
            exp_configure['predictor_hidden_feats'] = 300
            exp_configure['num_gnn_layers'] = 4
        if model_size_level == 8:
            exp_configure['gnn_hidden_feats'] = 800
            exp_configure['predictor_hidden_feats'] = 400
            exp_configure['num_gnn_layers'] = 5
        model = WeavePredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            num_gnn_layers=exp_configure['num_gnn_layers'],
            gnn_hidden_feats=exp_configure['gnn_hidden_feats'],
            graph_feats=exp_configure['graph_feats'],
            gaussian_expand=exp_configure['gaussian_expand'],
            n_tasks=exp_configure['n_tasks'],
        )
    elif exp_configure['model'] == 'MPNN':
        from dgllife.model.model_zoo import MPNNPredictor
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['node_out_feats'] = 8
            exp_configure['num_step_message_passing'] = 1
        if model_size_level == 2:
            exp_configure['node_out_feats'] = 16
            exp_configure['num_step_message_passing'] = 1
        if model_size_level == 3:
            exp_configure['node_out_feats'] = 32
            exp_configure['num_step_message_passing'] = 2
        if model_size_level == 4:
            exp_configure['node_out_feats'] = 64
            exp_configure['num_step_message_passing'] = 4
        if model_size_level == 5:
            exp_configure['node_out_feats'] = 128
            exp_configure['num_step_message_passing'] = 6
        if model_size_level == 6:
            exp_configure['node_out_feats'] = 256
            exp_configure['num_step_message_passing'] = 8
        if model_size_level == 7:
            exp_configure['node_out_feats'] = 256
            exp_configure['num_step_message_passing'] = 10
        if model_size_level == 8:
            exp_configure['node_out_feats'] = 512
            exp_configure['num_step_message_passing'] = 10
        exp_configure['edge_hidden_feats'] = 2 * exp_configure['node_out_feats']
        exp_configure['num_step_set2set'] = exp_configure['num_step_message_passing']
        exp_configure['num_layer_set2set'] = max(int(exp_configure['num_step_set2set']/2), 1)
        model = MPNNPredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            node_out_feats=exp_configure['node_out_feats'],
            edge_hidden_feats=exp_configure['edge_hidden_feats'],
            num_step_message_passing=exp_configure['num_step_message_passing'],
            num_step_set2set=exp_configure['num_step_set2set'],
            num_layer_set2set=exp_configure['num_layer_set2set'],
            n_tasks=exp_configure['n_tasks'],
        )
    elif exp_configure['model'] == 'AttentiveFP':
        from dgllife.model.model_zoo import AttentiveFPPredictor
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['graph_feat_size'] = 10
            exp_configure['num_layers'] = 1
        if model_size_level == 2:
            exp_configure['graph_feat_size'] = 20
            exp_configure['num_layers'] = 1
        if model_size_level == 3:
            exp_configure['graph_feat_size'] = 50
            exp_configure['num_layers'] = 2
        if model_size_level == 4:
            exp_configure['graph_feat_size'] = 100
            exp_configure['num_layers'] = 2
        if model_size_level == 5:
            exp_configure['graph_feat_size'] = 200
            exp_configure['num_layers'] = 2
        if model_size_level == 6:
            exp_configure['graph_feat_size'] = 400
            exp_configure['num_layers'] = 3
        if model_size_level == 7:
            exp_configure['graph_feat_size'] = 600
            exp_configure['num_layers'] = 4
        if model_size_level == 8:
            exp_configure['graph_feat_size'] = 800
            exp_configure['num_layers'] = 5
        exp_configure['num_timesteps'] = exp_configure['num_layers']
        model = AttentiveFPPredictor(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'],
        )
    
    elif exp_configure['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                                    'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.model.model_zoo import GINPredictor
        from dgllife.model.pretrain import load_pretrained
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=exp_configure['jk'],
            dropout=0.5,
            readout=exp_configure['readout'],
            n_tasks=exp_configure['n_tasks']
        )
        model.gnn = load_pretrained(exp_configure['model'])
        model.gnn.JK = exp_configure['jk']
    elif exp_configure['model'] == 'NF':
        from dgllife.model.model_zoo import NFPredictor
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['gnn_hidden_feats'] = 8
            exp_configure['predictor_hidden_feats'] = 8
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 2:
            exp_configure['gnn_hidden_feats'] = 16
            exp_configure['predictor_hidden_feats'] = 16
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 3:
            exp_configure['gnn_hidden_feats'] = 32
            exp_configure['predictor_hidden_feats'] = 32
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 4:
            exp_configure['gnn_hidden_feats'] = 64
            exp_configure['predictor_hidden_feats'] = 64
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 5:
            exp_configure['gnn_hidden_feats'] = 128
            exp_configure['predictor_hidden_feats'] = 128
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 6:
            exp_configure['gnn_hidden_feats'] = 256
            exp_configure['predictor_hidden_feats'] = 256
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 7:
            exp_configure['gnn_hidden_feats'] = 256
            exp_configure['predictor_hidden_feats'] = 256
            exp_configure['num_gnn_layers'] = 4
        if model_size_level == 8:
            exp_configure['gnn_hidden_feats'] = 512
            exp_configure['predictor_hidden_feats'] = 512
            exp_configure['num_gnn_layers'] = 5
        model = NFPredictor(
            in_feats=exp_configure['in_node_feats'],
            n_tasks=exp_configure['n_tasks'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_size=exp_configure['predictor_hidden_feats'],
            predictor_batchnorm=exp_configure['batchnorm'],
            predictor_dropout=exp_configure['dropout']
        )
    else:
        return ValueError("Expect model to be from ['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP', "
                          "'gin_supervised_contextpred', 'gin_supervised_infomax', "
                          "'gin_supervised_edgepred', 'gin_supervised_masking'], "
                          "got {}".format(exp_configure['model']))

    return model


def load_GCN_L_model(exp_configure, model_size_level):
    if exp_configure['model'] == 'GCN':
        from dgllife.model.model_zoo import GCNPredictor_MLP
        exp_configure['model_size_level'] = model_size_level
        if model_size_level == 1:
            exp_configure['gnn_hidden_feats'] = 8
            exp_configure['predictor_hidden_feats'] = 8
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 2:
            exp_configure['gnn_hidden_feats'] = 16
            exp_configure['predictor_hidden_feats'] = 16
            exp_configure['num_gnn_layers'] = 1
        if model_size_level == 3:
            exp_configure['gnn_hidden_feats'] = 32
            exp_configure['predictor_hidden_feats'] = 32
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 4:
            exp_configure['gnn_hidden_feats'] = 64
            exp_configure['predictor_hidden_feats'] = 64
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 5:
            exp_configure['gnn_hidden_feats'] = 128
            exp_configure['predictor_hidden_feats'] = 128
            exp_configure['num_gnn_layers'] = 2
        if model_size_level == 6:
            exp_configure['gnn_hidden_feats'] = 256
            exp_configure['predictor_hidden_feats'] = 256
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 7:
            exp_configure['gnn_hidden_feats'] = 512
            exp_configure['predictor_hidden_feats'] = 512
            exp_configure['num_gnn_layers'] = 3
        if model_size_level == 8:
            exp_configure['gnn_hidden_feats'] = 512
            exp_configure['predictor_hidden_feats'] = 512
            exp_configure['num_gnn_layers'] = 5
        if model_size_level == 9:
            exp_configure['gnn_hidden_feats'] = 1024
            exp_configure['predictor_hidden_feats'] = 1024
            exp_configure['num_gnn_layers'] = 7
        model = GCNPredictor_MLP(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])

def load_model_from_config(exp_configure):
    if exp_configure['model'] == 'GCN':
        from dgllife.model import GCNPredictor
        model = GCNPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'GAT':
        from dgllife.model import GATPredictor
        model = GATPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
            feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
            residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'Weave':
        from dgllife.model import WeavePredictor
        model = WeavePredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            num_gnn_layers=exp_configure['num_gnn_layers'],
            gnn_hidden_feats=exp_configure['gnn_hidden_feats'],
            graph_feats=exp_configure['graph_feats'],
            gaussian_expand=exp_configure['gaussian_expand'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'MPNN':
        from dgllife.model import MPNNPredictor
        model = MPNNPredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            node_out_feats=exp_configure['node_out_feats'],
            edge_hidden_feats=exp_configure['edge_hidden_feats'],
            num_step_message_passing=exp_configure['num_step_message_passing'],
            num_step_set2set=exp_configure['num_step_set2set'],
            num_layer_set2set=exp_configure['num_layer_set2set'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'AttentiveFP':
        from dgllife.model import AttentiveFPPredictor
        model = AttentiveFPPredictor(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                                    'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.model import GINPredictor
        from dgllife.model import load_pretrained
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=exp_configure['jk'],
            dropout=0.5,
            readout=exp_configure['readout'],
            n_tasks=exp_configure['n_tasks']
        )
        model.gnn = load_pretrained(exp_configure['model'])
        model.gnn.JK = exp_configure['jk']
    elif exp_configure['model'] == 'NF':
        from dgllife.model import NFPredictor
        model = NFPredictor(
            in_feats=exp_configure['in_node_feats'],
            n_tasks=exp_configure['n_tasks'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_size=exp_configure['predictor_hidden_feats'],
            predictor_batchnorm=exp_configure['batchnorm'],
            predictor_dropout=exp_configure['dropout']
        )
    else:
        return ValueError("Expect model to be from ['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP', "
                          "'gin_supervised_contextpred', 'gin_supervised_infomax', "
                          "'gin_supervised_edgepred', 'gin_supervised_masking', 'NF'], "
                          "got {}".format(exp_configure['model']))

    return model

#def predict(args, model, bg, feature_flag=False,sim_flag=False):#sim_flag=sim_flag
#    bg = bg.to(args['device'])
#    if args['edge_featurizer'] is None:
#        node_feats = bg.ndata.pop('h').to(args['device'])
#        return model(bg, node_feats,feature_flag=feature_flag,sim_flag=sim_flag)
#    elif args['bond_featurizer_type'] == 'pre_train':
#        node_feats = [
#            bg.ndata.pop('atomic_number').to(args['device']),
#            bg.ndata.pop('chirality_type').to(args['device'])
#        ]
#        edge_feats = [
#            bg.edata.pop('bond_type').to(args['device']),
#            bg.edata.pop('bond_direction_type').to(args['device'])
#        ]
#        return model(bg, node_feats, edge_feats,feature_flag=feature_flag,sim_flag=sim_flag)
#    else:
#        node_feats = bg.ndata.pop('h').to(args['device'])
#        edge_feats = bg.edata.pop('e').to(args['device'])
#        return model(bg, node_feats, edge_feats,feature_flag=feature_flag,sim_flag=sim_flag)
#    
#def predict(args, model, bg):
#    bg = bg.to(args['device'])
#    if args['edge_featurizer'] is None:
#        node_feats = bg.ndata.pop('h').to(args['device'])
#        return model(bg, node_feats)
#    elif args['bond_featurizer_type'] == 'pre_train':
#        node_feats = [
#            bg.ndata.pop('atomic_number').to(args['device']),
#            bg.ndata.pop('chirality_type').to(args['device'])
#        ]
#        edge_feats = [
#            bg.edata.pop('bond_type').to(args['device']),
#            bg.edata.pop('bond_direction_type').to(args['device'])
#        ]
#        return model(bg, node_feats, edge_feats)
#    else:
#        node_feats = bg.ndata.pop('h').to(args['device'])
#        edge_feats = bg.edata.pop('e').to(args['device'])
#        return model(bg, node_feats, edge_feats)


       
def predict(args, model, bg, feature_flag=False,last_layer=False,use_classifier=False):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats)
    elif args['bond_featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(args['device']),
            bg.ndata.pop('chirality_type').to(args['device'])
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(args['device']),
            bg.edata.pop('bond_direction_type').to(args['device'])
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop('h').to(args['device'])
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats)


