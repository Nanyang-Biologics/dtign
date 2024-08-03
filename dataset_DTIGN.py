# conda activate base
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
from torch_geometric.data import Batch

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

def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)
        
def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond)
        graph.add_edge(i, j, weight=bond_feats)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    x_bond = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges()], dtype=torch.float32)
    if not graph.edges(data=False):
        return [], [], [], True
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, x_bond, edge_index, False

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
def mols2graphs(complex_path_list, label, vina_score_list, save_path, dis_threshold):
    data_list = []
    fail_path = []
    for i, complex_path in enumerate(complex_path_list):
        if os.path.exists(complex_path):
            with open(complex_path, 'rb') as f:
                ligand, pocket = pickle.load(f)
        else:
            print('Complex file not found:', complex_path)
            fail_path.append(complex_path)
            continue

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        x_l, x_l_bond, edge_index_l, fail_l = mol2graph(ligand)
        x_p, x_p_bond, edge_index_p, fail_p = mol2graph(pocket)
        if fail_l or fail_p:
            print('Failed to read complex file:', complex_path)
            fail_path.append(complex_path)
            continue

        x = torch.cat([x_l, x_p], dim=0)
        x_bond = torch.cat([x_l_bond, x_p_bond], dim=0)
        edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
        try:
            edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        except:
            print('Failed to read complex edges:', complex_path)
            fail_path.append(complex_path)
            continue
            
        y = torch.FloatTensor([label])
        vina_score = torch.FloatTensor([vina_score_list[i]])
        pos = torch.concat([pos_l, pos_p], dim=0)
        split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
        pocket = complex_path.split('/')[-1].split('_')[2]
        
        data = Data(x=x, x_bond=x_bond, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, vina_score=vina_score, pos=pos, pocket=pocket, split=split)
        data_list.append(data)
        
    if len(fail_path) == len(complex_path_list):
        return complex_path_list
    else:
        merged_data = Batch.from_data_list(data_list)
        torch.save(merged_data, save_path)
        return fail_path

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, run_datafold=None, dis_threshold=5, num_pose=1, graph_type='Graph_GIGN', assay_type='pIC50', num_process=8, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.num_pose = num_pose
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.assay_type = assay_type
        self.num_process = num_process
        self.mean, self.std = 0, 1
        self.GIGN_datafold = run_datafold
        self._generate_graph_not_found_list()
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        pocket_num = len(os.listdir(data_dir))

        complex_path_list, complex_id_list, pIC50_list, score_list, graph_path_list, dis_threshold_list = [], [], [], [], [], []
        file_list = os.listdir(data_dir)
        pocket_list_all = [x for x in file_list if x.split('_')[-1] == f'{self.dis_threshold}A.rdkit']
        not_found_list = []
        for i, row in data_df.iterrows():
            cid, pIC50 = row['ChEMBL_Compound_ID'], float(row[self.assay_type])
            complex_path_list_cid, complex_id_list_cid, score_list_cid = [], [], []
            graph_path = os.path.join(data_dir, f"{cid}_{graph_type}_{self.dis_threshold}A.pyg")
            pocket_list = [x for x in pocket_list_all if x.split('_')[0] == cid]
            if pocket_list:
                for pocket in pocket_list:
                    pocket_idx = pocket.split('_')[2]
                    score = float(row[f'Pocket_{pocket_idx}_Vina_Score'])
                    complex_path = os.path.join(data_dir, f"{cid}_Complex_{pocket_idx}_{self.dis_threshold}A.rdkit")
                    if len(pocket_idx.split('-')) > 1 and int(pocket_idx.split('-')[1]) <= self.num_pose or len(pocket_idx.split('-')) == 1:
                        complex_path_list_cid.append(complex_path)
                        score_list_cid.append(score)
                complex_path_list.append(complex_path_list_cid)
                score_list.append(score_list_cid)
                complex_id_list.append(cid)
                pIC50_list.append(pIC50)
                graph_path_list.append(graph_path)
                dis_threshold_list.append(self.dis_threshold)
            else:  
                not_found_list.append(graph_path)

        self.mean, self.std = np.mean(pIC50_list), np.std(pIC50_list)
        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            for complex_path, pIC50, vina_score ,graph_path, dis_threshold in zip(complex_path_list, pIC50_list, score_list, graph_path_list, dis_threshold_list):
                not_found_path = mols2graphs(complex_path, pIC50, vina_score ,graph_path, dis_threshold)
                if len(not_found_path) == len(complex_path):
                    not_found_list.append(graph_path)
            with open(data_dir + f'/{self.graph_type}_not_found_list.pkl', 'wb') as f:
                pickle.dump(not_found_list, f)
            pool.close()
            pool.join()
        
        with open(data_dir + f'/{self.graph_type}_not_found_list.pkl', 'rb') as f:
            not_found_list = pickle.load(f)
            # print("not_found_list: ", not_found_list)
        not_found_list = [path.replace(self.GIGN_datafold, '.') for path in not_found_list]
        self.complex_ids = complex_id_list
        self.graph_paths = [x for x in graph_path_list if x not in not_found_list]
    
    def _generate_graph_not_found_list(self):
        train_dir = self.data_dir
        print("train_dir: ", train_dir)
        files = os.listdir(train_dir)
        for file in files:
            if file.startswith('CHEMBL') and file.endswith('Graph_Bond_5A.pyg'):
                new_name = file.replace('Graph_Bond_5A', 'Graph_DTIGN_5A')
                old_file_path = os.path.join(train_dir, file)
                new_file_path = os.path.join(train_dir, new_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {file} to {new_name}")
        suffix = train_dir.split('/')[-1]
        chembl_id = train_dir.split('/')[-4]

        # Construct the CSV file path from train_dir
        csv_file = train_dir.replace(suffix, f'{chembl_id}_{self.assay_type}_{suffix}.csv')

        # Read the CSV file if it exists
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as file:
                df = pd.read_csv(file)
                csv_ids = df['ChEMBL_Compound_ID'].tolist()
                file_ids = [file.split('_')[0] for file in files if file.startswith('CHEMBL') and file.endswith('Graph_DTIGN_5A.pyg')]
                missing_ids = [chembl_id for chembl_id in csv_ids if chembl_id not in file_ids]
                output_pickle_path = os.path.join(train_dir, 'Graph_DTIGN_not_found_list.pkl')
                if os.path.exists(output_pickle_path):
                    print(f"The file {output_pickle_path} already exists. No new file created.")
                    return
                with open(output_pickle_path, 'wb') as f:
                    missing_ids = [f"{train_dir}/{chembl_id}_Graph_DTIGN_5A.pyg" for chembl_id in missing_ids]
                    pickle.dump(missing_ids, f)
        else:
            print(f'{csv_file} does not exist.')
        return []

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx])
        match = re.search(r'CHEMBL(\d+)_', self.graph_paths[idx])
        chembl_id = match.group(0)
        data['idx'] = chembl_id[:-1]
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.graph_paths)

    
if __name__ == '__main__':
    skip_exist, graph_type = False, 'Graph_Bond'
    task_dict = {'I1': ('CHEMBL202', 'pIC50', '1boz', 7, 1), 'E3': ('CHEMBL235', 'pEC50', '1zgy'), 
                 'I5': ('CHEMBL279', 'pIC50', '1ywn'), 'I4': ('CHEMBL2971', 'pIC50', '3ugc'), 
                 'I3': ('CHEMBL333', 'pIC50', '1ck7'), 'E1': ('CHEMBL3820', 'pEC50', '3f9m'), 
                 'I2': ('CHEMBL3976', 'pIC50', '4ebb', 2, 4), 'E2': ('CHEMBL4422', 'pEC50', '5tzr', 3, 3)}
    protein_name, assay_type, pdb_name, pocket_num, pose_num = task_dict['I2']
    data_root = f'/home/yueming/Drug_Discovery/Baselines/GIGN-main/GIGN/data/{protein_name}/{pdb_name}/vina/'
    set_list = ['test', 'train_1', 'train_2', 'train_3', 'train_4', 'train_5']
    for set_name in set_list:
        data_dir = data_root + set_name + '/'
        data_df = pd.read_csv(os.path.join(data_root, f'{protein_name}_{assay_type}_{set_name}.csv'))
        dataset = GraphDataset(data_dir, data_df, graph_type='Graph_GIGN', assay_type=assay_type, dis_threshold=5, create=True)
        print('Dataset size:', len(dataset))
        data_loader = PLIDataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
        for data in data_loader:
            # print(data) --> DataBatch(x=[2481, 35], y=[8], pos=[2481, 3], edge_index_intra=[2, 4884], edge_index_inter=[2, 4456], split=[2481], pocket=[8], batch=[2481], ptr=[9])
            data, pocket, idx, label, socre = data, data.pocket, data.idx, data.y, data.vina_score
            print(f'Loading {len(pocket)} data successfully')