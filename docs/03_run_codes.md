# Reproducibility


<br>

## 1. Usage


<br>

### 1.1 In terminal

```
# Run the main file (at the root of the project)
python main_molecules_graph_regression.py --dataset ZINC --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' # for CPU
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC.json' # for GPU
```
The training and network parameters for each dataset and network is stored in a json file in the [`configs/`](../configs) directory.












<br>

### 1.2 In jupyter notebook
```
# Run the notebook file (at the root of the project)
conda activate benchmark_gnn 
jupyter notebook
```
Use [`main_molecules_graph_regression.ipynb`](../main_molecules_graph_regression.ipynb) notebook to explore the code and do the training interactively.




<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_graph_regression_GatedGCN_ZINC.json`](../configs/molecules_graph_regression_GatedGCN_ZINC.json) file).  

If `out_dir = 'out/molecules_graph_regression/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/molecules_graph_regression/results` to view all result text files.
2. Directory `out/molecules_graph_regression/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`
2. Run the command `tensorboard --logdir='./'`
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006) appears on the terminal immediately after running Step 2.




<br>

## 3. Reproduce results (4 runs on all, except CSL and TUs)


```
# At the root of the project 
bash scripts/Performance/script_main_superpixels_graph_classification_MNIST_shallow.sh # run MNIST dataset for 100k params
bash scripts/Performance/script_main_superpixels_graph_classification_CIFAR10_shallow.sh # run CIFAR10 dataset for 100k params

bash scripts/Performance/script_main_molecules_graph_regression_ZINC_small.sh # run ZINC dataset for 100k params
bash scripts/Performance/script_main_molecules_graph_regression_ZINC_large.sh # run ZINC dataset for 500k params

bash scripts/Performance/script_main_SBMs_node_classification_PATTERN_small.sh # run PATTERN dataset for 100k params
bash scripts/Performance/script_main_SBMs_node_classification_PATTERN_large.sh # run PATTERN dataset for 500k params
bash scripts/Performance/script_main_SBMs_node_classification_CLUSTER_small.sh # run CLUSTER dataset for 100k params
bash scripts/Performance/script_main_SBMs_node_classification_CLUSTER_large.sh # run CLUSTER dataset for 500k params

bash scripts/Performance/script_main_TSP_edge_classification.sh # run TSP dataset for 100k params
bash scripts/Performance/script_main_TSP_edge_classification_edge_feature_analysis.sh # run TSP dataset for edge feature analysis 

bash scripts/Performance/script_main_COLLAB_edge_classification.sh # run COLLAB dataset for 40k params
bash scripts/Performance/script_main_COLLAB_edge_classification_edge_feature_analysis.sh # run COLLAB dataset for edge feature analysis 

bash scripts/CSL/script_main_CSL_graph_classification.sh # run CSL dataset for on 1 seed
bash scripts/CSL/script_main_CSL_graph_classification_20_seeds.sh # run CSL dataset for on 20 seeds

bash scripts/TU/script_main_TUs_graph_classification_seed1.sh # run TU datasets for 100k params on seed1
bash scripts/TU/script_main_TUs_graph_classification_seed2.sh # run TU datasets for 100k params on seed2
```

Scripts are [located](../scripts/) at the `scripts/` directory of the repository.

 


















<br><br><br>