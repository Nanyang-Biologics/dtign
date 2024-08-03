# CHEMBL Experiments
Reproduction of the CHEMBL experiments during Yin Yueming's (yueming.yin@ntu.edu.sg) Research Fellow period under the management of Kong Wai-Kin Adams (adamskong@ntu.edu.sg).

- This work involves private data. All rights reserved by Li Hoi Yeung (hyli@ntu.edu.sg), Kong Wai-Kin Adams (adamskong@ntu.edu.sg), Yin Yueming (yueming.yin@ntu.edu.sg) and Chong Kim San Allen (kimsanallen.chong@ntu.edu.sg).

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
We start from constructing CHEMBL benchmark datasets within the CHEMBL data collected by Chong Kim San Allen (kimsanallen.chong@ntu.edu.sg).

One can construct DTIGN benchmark datasets by re-runing `./Construct_Datasets.ipynb`. Here, we provide the main steps in `./Construct_Datasets.ipynb`:
```
[1] Extract information from raw data according to target ids and assay types
>>> running cells
[2] Split all data into subsets for cross-validation according to molecular scaffolds
>>> running cells
```

# Reproduction of Results
Our results on CHEMBL experiments are saved in `./results/Results.tar.xz` with the following architecture:
```
[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/metric.csv: performance on various metrics
[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/configure.json: configuration of model training
[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/prediction.csv: predictions on test set using best model obtained in training
(Optional)[target id, e.g. CHEMBL202]/[assay type, e.g. IC50]/[model, e.g. GCN]/[model setting, e.g. S8]/[cross-validation fold, e.g. 5]/model.pth: best model during cross-validation training (only the best model parameters are saved in the corresponding cross-validation folder)
```

## Reproduction of Uni-Mol Results on Benchmark Datasets
For reproducing the training process of Uni-Mol on benchmark datasets, please re-run `../CHEMBL_Experiments/Uni-Mol/notebooks/benchmark_test.ipynb.ipynb`. It will train and test Uni-Mol models on all benchmark datasets in sequence.
- Results can be directly read out from the cell output, and can be also found in the `args['result_path']`.

## Reproduction of SGCN Results on Benchmark Datasets
For reproducing the training process of SGCN on benchmark datasets, please run the following commands in terminal:
```bash
cd ./SGCN
python typical_all_SGCN.py.py
```
- The codes will train and test SGCN models on all benchmark datasets in sequence.
- Results are saved to `args['result_path']` defined in Line 445 of `./SGCN/typical_all_SGCN.py`, where the performance is recorded in `[experimental tag]/metric.csv`.

For reproducing the training process of AFSE+SGCN on benchmark datasets, please run the following commands in terminal:
```bash
cd ./SGCN
python typical_all_AFSE_SGCN.py.py.py
```
- The codes will train and test SGCN models on all benchmark datasets in sequence.
- Results are saved to `args['result_path']` defined in Line 489 of `./SGCN/typical_all_AFSE_SGCN.py.py`, where the performance is recorded in `[experimental tag]/metric.csv`.

## Reproduction of DGL Results on Benchmark Datasets
Deep graph learning (DGL) models used in this study include: GCN, GAT, Weave, MPNN, AttentiveFP, Neural FP, GIN.
To reproduce results on these DGL models, please run the following commands in your terminal:
```bash
python train_DGLs.py
``` 
- The codes will train and test DGL models on all benchmark datasets in sequence.
- Results are saved to `args['result_path']` defined in Line 471 of `./train_DGLs.py`, where the performance is recorded in `[experimental tag]/metric.csv`.

To reproduce results on AFSE + these DGL models, please run the following commands in your terminal:
```bash
python train_DGL_with_AFSE.py
``` 
- The codes will train and test DGL models on all benchmark datasets in sequence.
- Results are saved to `args['result_path']` defined in Line 600 of `./train_DGL_with_AFSE.py`, where the performance is recorded in `[experimental tag]/metric.csv`.

# Reverse Virtual Screening (RVS)
The aim of RVS is to find potential targets that a query set of compounds would bind. In our experiments, compounds from Drugbank are used as an example of the query set. To do the RVS of Drugbank compounds on all CHEMBL benchmark datasets, please re-run `./Find_Potential_Targets.ipynb`. It will predict Drugbank compounds's bioactivity on all CHEMBL target we collected using our best model stored in `./results/Results.tar.xz`.
- Make sure that `./results/Results.tar.xz` has been extract to `../Results` before running `./Find_Potential_Targets.ipynb`.