import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# %%
def generate_pocket(data_dir, distance=5):
    complex_id = os.listdir(data_dir)
    for file in complex_id:
        if file[-6:] == '.pdbqt':
            cid = file[:-6]
            lig_native_path = os.path.join(data_dir, file)
            protein_path= './data/CHEMBL202/protein/pdb1boz.pdbqt'

#             if os.path.exists(os.path.join(data_dir, f'{cid}_Pocket_{distance}A.pdb')):
#                 continue

            pymol.cmd.load(protein_path)
            pymol.cmd.remove('resn HOH')
            pymol.cmd.load(lig_native_path)
            pymol.cmd.remove('hydrogens')
            pymol.cmd.select('Pocket', f'byres pdb1boz within {distance} of {cid}')
            pymol.cmd.save(os.path.join(data_dir, f'{cid}_Pocket_{distance}A.pdb'), 'Pocket')
            pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pIC50 = row['ChEMBL_Compound_ID'], float(row['pIC50'])
        pocket_path = os.path.join(data_dir, f'{cid}_Pocket_{distance}A.pdb')
        ligand_input_path = os.path.join(data_dir, f'{cid}.{input_ligand_format}')
        ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
        if not os.path.exists(ligand_input_path):
            if os.path.exists(ligand_path):
                os.remove(ligand_path)
            continue
        if input_ligand_format != 'pdb' and not os.path.exists(ligand_path):
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(data_dir, f'{cid}.pdb')

        save_path = os.path.join(data_dir, f"{cid}_Complex_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue
            
        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'pdbqt'
    data_root = './data/CHEMBL202/'
    set_list = ['test', 'train_1', 'train_2', 'train_3', 'train_4', 'train_5']
    for dataset in tqdm(set_list):
        data_df = pd.read_csv(os.path.join(data_root, f'CHEMBL202_pIC50_{dataset}.csv'))
        dataset_path = data_root + '1boz/' + dataset + '/'
        for pocket in tqdm(os.listdir(dataset_path)):
            data_dir = dataset_path + pocket + '/'

            ## generate pocket within 5 Ångström around ligand 
            generate_pocket(data_dir=data_dir, distance=distance)
            generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)


# %%
