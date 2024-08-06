# %%
import os, re
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
from tqdm import tqdm
import warnings
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    if not graph.edges(data=False):
        return [], [], True
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index, False

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    
    if os.path.exists(complex_path):
        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)
    else:
        print('Complex file not found:', complex_path)
        return complex_path

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l, fail_l = mol2graph(ligand)
    x_p, edge_index_p, fail_p = mol2graph(pocket)
    if fail_l or fail_p:
        print('Failed to read complex file:', complex_path)
        return complex_path
    
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    
    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

    torch.save(data, save_path)
    return False

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self.mean, self.std = 0, 1
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        pocket_num = len(os.listdir(data_dir))
        dis_thresholds = repeat(self.dis_threshold, len(data_df)*pocket_num)

        complex_path_list = []
        complex_id_list = []
        pIC50_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pIC50 = row['ChEMBL_Compound_ID'], float(row['pIC50'])
            for pocket in os.listdir(data_dir):
                pocket_idx = pocket.split('_')[-1]
                vina = float(row[f'Pocket_{pocket_idx}_Vina_Score'])
                complex_dir = os.path.join(data_dir, pocket)
                graph_path = os.path.join(complex_dir, f"{cid}_{graph_type}_{self.dis_threshold}A.pyg")
                complex_path = os.path.join(complex_dir, f"{cid}_Complex_{self.dis_threshold}A.rdkit")

                complex_path_list.append(complex_path)
                complex_id_list.append(cid)
                pIC50_list.append(pIC50)
                graph_path_list.append(graph_path)
                
        self.mean, self.std = np.mean(pIC50_list), np.std(pIC50_list)
        not_found_list = []
        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            not_found_path = pool.starmap(mols2graphs, zip(complex_path_list, pIC50_list, graph_path_list, dis_thresholds))
            for flag in not_found_path:
                if flag:
                    exclude_graph = flag.replace('.rdkit', '.pyg').replace('Complex', graph_type)
                    not_found_list.append(exclude_graph)
            with open(data_dir + '_not_found_list.pkl', 'wb') as f:
                pickle.dump(not_found_list, f)
            pool.close()
            pool.join()
        
        with open(data_dir + '_not_found_list.pkl', 'rb') as f:
            not_found_list = pickle.load(f)
        self.complex_ids = complex_id_list
        self.graph_paths = [x for x in graph_path_list if x not in not_found_list]

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx])
        match = re.search(r'pocket_(\d+)', self.graph_paths[idx])
        pocket_idx = int(match.group(1))
        data['pocket'] = pocket_idx
        match = re.search(r'CHEMBL(\d+)_', self.graph_paths[idx])
        chembl_id = match.group(0)
        data['idx'] = chembl_id[:-1]
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.graph_paths)


if __name__ == '__main__':
    data_root = './data/CHEMBL202/'
    set_list = ['test', 'train_1', 'train_2', 'train_3', 'train_4', 'train_5']
    for dataset in tqdm(set_list):
        data_df = pd.read_csv(os.path.join(data_root, f'CHEMBL202_pIC50_{dataset}.csv'))
        dataset_path = data_root + '1boz/' + dataset + '/'
        data_set = GraphDataset(dataset_path, data_df, graph_type='Graph_GIGN', dis_threshold=5, create=True)
        print(len(data_set))
        data_loader = PLIDataLoader(data_set, batch_size=8, shuffle=True, num_workers=4)
        for data in data_loader:
            # print(data) --> DataBatch(x=[2481, 35], y=[8], pos=[2481, 3], edge_index_intra=[2, 4884], edge_index_inter=[2, 4456], split=[2481], pocket=[8], batch=[2481], ptr=[9])
            data, pocket, idx = data, data.pocket, data.idx
            print(f'Loading {len(pocket)} data with the pocket idx:', pocket, 'and idx:', idx)
            
# %%
