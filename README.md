# DTIGN
Reproduction of the paper entitled "Advancing Bioactivity Prediction through Molecular Docking and Self-Attention: The Drug-Target Interaction Graph Neural Network (DTIGN)". This paper has been submitted to IEEE Journal of Biomedical and Health Informatics after a minor revision.

- This work is accorded Singapore provisional patent application number 10202400124Q. All rights reserved by Kong Wai-Kin Adams (adamskong@ntu.edu.sg), Yin Yueming (yueming.yin@ntu.edu.sg), Mu Yuguang (ygmu@ntu.edu.sg) and Li Hoi Yeung (hyli@ntu.edu.sg).

# Requirements
- Anaconda
- Pytorch
- Jupyter
- GPU with CUDA

# Get Started With A New Conda Virtual Environment
## Anaconda Installation
For installing anaconda on your OS, please refer to [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

Once the anaconda is properly installed, one can create a new virtual environment by entering the following command in terminal:
```bash
conda create rdkit python=3.6 -c rdkit -n rdkit
conda activate rdkit
```
- Note that `python=3.6` is adopted due to its good compatibility with a wide range of packages in our projects. One can try other versions if no compatibility issues occur.

## Pytorch Installation
For installing pytorch to a certain version, please refer to [Pytorch Installation](https://pytorch.org/get-started/locally/) or [Early Versions](https://pytorch.org/get-started/previous-versions/).

Here, we provide our command of install `pytorch-cuda=11.6` with anaconda in terminal:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
- Note that the version of `pytorch-cuda=11.6` should be consistent with your CUDA version, choose suitable commond according to your CUDA version from [here](https://pytorch.org/get-started/previous-versions/).

## Jupyter Installation
For installing and configuring jupyter, please refer to [Jupyter Installation](https://jupyter.org/install) and [Jupyter Configuration](https://jupyter-server.readthedocs.io/en/latest/users/configuration.html).

Here, we provide our commands of installing and configuring jupyter with conda:
```bash
conda install jupyter notebook
jupyter notebook --generate-config
ipython
>>> from notebook.auth import passwd
>>> passwd()
(enter your passwords)
(conform your passwords)
Output: [your password key]
```

Then, open your generated jupyter_notebook_config.py file (usually in the '/home/.jupyter' directory) to add the following commands:
```python
c.NotebookApp.ip = '*'
c.NotebookApp.port = [port id]
c.NotebookApp.open_browser = False
c.NotebookApp.allow_remote_access = True
c.NotebookApp.password = [your password key]
c.NotebookApp.notebook_dir = [your default workdir]
```

To start jupyter notebook, please enter this commond in the terminal:
```bash
jupyter notebook
```
Then, one can access jupyter notebook at [your server's ip]:[port id].

## Package Installation
One can install environmental packages as the following commands in terminal:
```bash
conda install matplotlib tensorboardx tensorboard seaborn
conda install -c anaconda scikit-learn
pip install hyperopt
conda install pyg -c pyg
conda install -c conda-forge multicore-tsne
pip install retry
conda install -c schrodinger pymol
conda install -c conda-forge openbabel
pip install torch-geometric, torch-sparse, torch-scatter
```
For the installation of additional deep graph learning (dgl) package, please refer to [DGL Installation](https://www.dgl.ai/pages/start.html).
In our case, the installation command in terminal is:
```bash
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
```

# Data Preparation and Processing
## Benchmark Dataset Construction
We start from constructing DTIGN benchmark datasets within the CHEMBL benchmark datasets. For the instruction of constructing the CHEMBL benchmark datasets, please refer to `../CHEMBL_Experiments/README.md`.

Once the CHEMBL benchmark datasets are constructed, one can construct DTIGN benchmark datasets by following the instructions and codes in `./Data_Processing.ipynb`. Here, we provide the recommended order. Follow the markdown of `./Data_Processing.ipynb`, run section named
```
[1] Search suitable datasets
>>> running cells
[2] Standardize datasets
>>> running cells
```
- For constructing PARP1 dataset, please refer to the `Preprocessing PARP1 datasets` Section in `./Data_Processing.ipynb`.

## Docking Data Generation
The benchmark datasets constructed in the above subsection contain SMILES strings of molecules and their bioactivities against certain drug targets (one dataset for one target). This subsection introduces the measure to generate docking data according to benchmark datasets for DITGN training and test.

Following the instructions and codes in `./Prepare_Docking.ipynb`, one can generate docking data used in DTIGN by the following order:
```
[1] Save SDF for Compounds in the benchmark datasets
>>> running cells
[2] Prepare ligands in pdbqt format for molecular docking
>>> running cells
[3] Do AutoDock Vina on a batch of ligands and one protein
>>> running cells
[4] Save Vina scores into the datasets
>>> running cells
[5] Preprocessing docking results to mols
>>> running cells
[6] Save GIGN graphs for each ligand
>>> running cells
[7] Save GIGN graphs for each ligand with bond features (DTIGN)
>>> running cells
```
- Note that one can change the parameters in the line `protein_name, pdb_name, pocket_num = 'CHEMBL3976', '4ebb', 2` of step [3] to do multiprocess of docking on multiple proteins and datasets. Here, Autodock Vina is used to generate docking data due to its stable performance.
- For generating docking data in the PARP1 dataset, a similar pipline is provided in `./Prepare_Docking_PARP1.ipynb`.

# Reproduction of Results
## Reproduction of DTIGN Results on Benchmark Datasets
For reproducting the training process of DTIGN on benchmark datasets, please run the following command in terminal:
```bash
python train_DTIGN.py
```

For reproducting the training process of DTIGN on PARP1 datasets, please run the following command in terminal:
```bash
python train_DTIGN_PARP1.py
```

To change the experiment settings, please modify hyperparameters in `/train_DTIGN.py` as follows:
```python
[Line 136] task_id = [task id, choice=['I1', 'I2', 'I3', 'I4', 'I5', 'E1', 'E2', 'E3']]
[Line 137] start_fold, skip_fold, stop_fold, warmup_epoch, val_rate, seed, learning_rate, hidden_dim, val_num, subset_num, step_size, gamma, early_stop_epoch, D_count = [start cross-validation fold, choice={1,2,3,4,5}], [skipped cross-validation fold, choice={2,3,4}], [end cross-validation fold, choice={1,2,3,4,5}], [warmup_epoch, default=40], [validation rate, choice={1.0, 0.9}, default=1.0], [random seed, defult=0, choice={0,1,2,3,4,5}], [learning rate, default=1e-4, choice={1e-4, 5e-4}], [hidden dimension of DTIGN, default=128], [numbere of validation samples, default=100], [cross-validation folds, default=5], [step size of learning rate scheduler, default=5], [decay rate of learning rate scheduler, default=0.99], [the epochs of stopping model training after achieving the global best validation metric, default=100], [the number of sample points between minimum and maximum atomic distance, default=32]
[Line 139] semi_supervise = [if using semi-supervised training strategy, default=False, cchoice={True, False}]
[Line 140] save_attention = [if saving model attentions in the multi-head self-attention mechanism, default=False, cchoice={True, False}]
```
- Note that if `semi_supervise = True` is set in line 139 of `./train_DTIGN.py`, please make sure the corresponding `./data/[target id]/pdb_with_activity/[subset]` folder includes collected crystall structures for semi-supervised learning.
- Experimental settings are similar in `./train_DTIGN_PARP1.py` (Line 140-148).

The training and test results will be saved in the path defined as `save_dir` in `./config/Train_Config_DTIGN.json`. Our results are saved in `./results/benchmark_test.tar.xz` for constructed benchmark datasets and `./results/PARP1.tar.xz` for the PARP1 dataset. The results of each experiment consists five major part:
```
[Experimental tag]/log/train.log: training and validation losses in epochs
[Experimental tag]/log/[other files]: the codes used to do this experiment
[Experimental tag]/model/Best, [tag]: the best model obtained after training
[Experimental tag]/model/Current, [tag]: the current model during training
(Optional)[Experimental tag]/result/[subset]_attention.pickle: the model attentions saved for further analysis
```
- To reproduce figures included in our paper, please refer to `./Data_Analysis.ipynb`.

## Reproduction of Baseline Results on Benchmark Datasets
Our results on baseline models are saved in `./results/baselines.tar.xz` and `./results/Baselines_Uni-Mol_SGCN.tar.xz` with the following architecture:
```
[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/metric.csv: performance on various metrics
[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/configure.json: configuration of model training
[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/prediction.csv: predictions on test set using best model obtained in training
(Optional)[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/model.pth: best model during cross-validation training (only the best model parameters are saved in the corresponding cross-validation folder)
```

### Reproduction of GIGN Results on Benchmark Datasets
For reproducing the training process of GIGN on benchmark datasets, please run the following command in terminal:
```bash
python train_GIGN.py
```

To change the experiment settings, please modify hyperparameters in `/train_GIGN.py` as follows:
```python
[Line 127] task_id = [task id, choice=['I1', 'I2', 'I3', 'I4', 'I5', 'E1', 'E2', 'E3']]
[Line 134] start_fold, skip_fold, end_fold, warmup_epoch, val_rate, seed = [start cross-validation fold, choice={1,2,3,4,5}], [skipped cross-validation fold, choice={2,3,4}], [end cross-validation fold, choice={1,2,3,4,5}], [warmup_epoch, default=40], [validation rate, choice={1.0, 0.9}, default=1.0], [random seed, defult=0, choice={0,1,2,3,4,5}]
```
- The training and test results will be saved in the path defined as `save_dir` in `./config/Train_Config_GIGN.json` with the same architecture with DTIGN. 

### Reproduction of Uni-Mol Results on Benchmark Datasets
For reproducing the training process of Uni-Mol on benchmark datasets, please re-run `../CHEMBL_Experiments/Uni-Mol/notebooks/standardized_test.ipynb`. To change the running experiments, please modify the line 14 of `Batch Runing` Section in `../CHEMBL_Experiments/Uni-Mol/notebooks/standardized_test.ipynb` as follows:
```
[Cell 2][Line 14] for task_id in tqdm([task id to run, a list of choice={0,1,2,3,4,5,6,7}]):
```
- Results can be directly read out from the cell output.

### Reproduction of SGCN Results on Benchmark Datasets
For reproducing the training process of SGCN on benchmark datasets, please run the following commands in terminal:
```bash
cd ../CHEMBL_Experiments/SGCN
python typical_SGCN_standardized.py
```

To change the running experiments, please modify the Line 427 in `./SGCN/typical_SGCN_standardized.py` as follows:
```
[Line 427] for task_id in tqdm([task id to run, a list of choice={0,1,2,3,4,5,6,7}]):
```
- Results are saved to `args['result_path']` defined in Line 441 of `./SGCN/typical_SGCN_standardized.py`, where the performance is recorded in `[experimental tag]/metric.csv`.

### Reproduction of DGL Results on Benchmark Datasets
Deep graph learning (DGL) models used in this study include: GCN, GAT, Weave, MPNN, AttentiveFP, Neural FP, GIN.
To reproduce results on these DGL models, please run the following commands in your terminal:
```bash
cd ../CHEMBL_Experiments
python typical_all_models.py
``` 

To change the running experiments, please modify the Line 427 in `../CHEMBL_Experiments/typical_all_models.py` as follows:
```
[Line 457] for model in [the list of training model in sequence, choose from {'GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP', 'gin_supervised_contextpred', 'NF'}]:
[Line 469] for task_id in tqdm([index of training task, a list choose from {0,1,2,3,4,5,6,7}]):
```
- Results are saved to `args['result_path']` defined in Line 487 of `../CHEMBL_Experiments/typical_all_models.py`, where the performance is recorded in `[experimental tag]/metric.csv`.