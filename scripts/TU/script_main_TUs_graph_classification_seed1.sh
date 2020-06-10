#!/bin/bash



# bash script_main_TUs_graph_classification_seed1.sh


############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT
#DiffPool





############
# ENZYMES & DD & PROTEINS_full
############
seed=41
code=main_TUs_graph_classification.py 
tmux new -s benchmark_TUs_graph_classification -d
tmux send-keys "source activate benchmark_gnn" C-m
dataset=ENZYMES
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GatedGCN_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_GCN_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GraphSage_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_MLP_ENZYMES.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GIN_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_MoNet_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GAT_ENZYMES.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_3WLGNN_ENZYMES.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_RingGNN_ENZYMES.json' &
wait" C-m
dataset=DD
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GatedGCN_DD.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_GCN_DD.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GraphSage_DD.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_MLP_DD.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GIN_DD.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_MoNet_DD.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GAT_DD.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_3WLGNN_DD.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_RingGNN_DD.json' &
wait" C-m
dataset=PROTEINS_full
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GatedGCN_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_GCN_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GraphSage_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_MLP_PROTEINS_full.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_GIN_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed --config 'configs/TUs_graph_classification_MoNet_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed --config 'configs/TUs_graph_classification_GAT_PROTEINS_full.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed --config 'configs/TUs_graph_classification_3WLGNN_PROTEINS_full.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed --config 'configs/TUs_graph_classification_RingGNN_PROTEINS_full.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_TUs_graph_classification" C-m






# ############
# # ENZYMES
# ############

# code=main_TUs_graph_classification.py 
# tmux new -s benchmark_TUs_graph_classification -d
# tmux send-keys "source activate benchmark_gnn" C-m
# dataset=ENZYMES
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_GatedGCN_ENZYMES.json' &
# python $code --dataset $dataset --gpu_id 1 --config 'configs/TUs_graph_classification_GCN_ENZYMES.json' &
# python $code --dataset $dataset --gpu_id 2 --config 'configs/TUs_graph_classification_GraphSage_ENZYMES.json' &
# python $code --dataset $dataset --gpu_id 3 --config 'configs/TUs_graph_classification_MLP_ENZYMES.json' &
# wait" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_GIN_ENZYMES.json' &
# python $code --dataset $dataset --gpu_id 1 --config 'configs/TUs_graph_classification_MoNet_ENZYMES.json' &
# python $code --dataset $dataset --gpu_id 2 --config 'configs/TUs_graph_classification_GAT_ENZYMES.json' &
# python $code --dataset $dataset --gpu_id 3 --config 'configs/TUs_graph_classification_3WLGNN_ENZYMES.json' &
# wait" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_RingGNN_ENZYMES.json' &
# wait" C-m
# tmux send-keys "tmux kill-session -t benchmark_TUs_graph_classification" C-m






# ############
# # DD 
# ############

# code=main_TUs_graph_classification.py 
# tmux new -s benchmark_TUs_graph_classification -d
# tmux send-keys "source activate benchmark_gnn" C-m
# dataset=DD
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_GatedGCN_DD.json' &
# python $code --dataset $dataset --gpu_id 1 --config 'configs/TUs_graph_classification_GCN_DD.json' &
# python $code --dataset $dataset --gpu_id 2 --config 'configs/TUs_graph_classification_GraphSage_DD.json' &
# python $code --dataset $dataset --gpu_id 3 --config 'configs/TUs_graph_classification_MLP_DD.json' &
# wait" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_GIN_DD.json' &
# python $code --dataset $dataset --gpu_id 1 --config 'configs/TUs_graph_classification_MoNet_DD.json' &
# python $code --dataset $dataset --gpu_id 2 --config 'configs/TUs_graph_classification_GAT_DD.json' &
# python $code --dataset $dataset --gpu_id 3 --config 'configs/TUs_graph_classification_3WLGNN_DD.json' &
# wait" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_RingGNN_DD.json' &
# wait" C-m
# tmux send-keys "tmux kill-session -t benchmark_TUs_graph_classification" C-m







# ############
# # PROTEINS_full
# ############

# code=main_TUs_graph_classification.py 
# tmux new -s benchmark_TUs_graph_classification -d
# tmux send-keys "source activate benchmark_gnn" C-m
# dataset=PROTEINS_full
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_GatedGCN_PROTEINS_full.json' &
# python $code --dataset $dataset --gpu_id 1 --config 'configs/TUs_graph_classification_GCN_PROTEINS_full.json' &
# python $code --dataset $dataset --gpu_id 2 --config 'configs/TUs_graph_classification_GraphSage_PROTEINS_full.json' &
# python $code --dataset $dataset --gpu_id 3 --config 'configs/TUs_graph_classification_MLP_PROTEINS_full.json' &
# wait" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_GIN_PROTEINS_full.json' &
# python $code --dataset $dataset --gpu_id 1 --config 'configs/TUs_graph_classification_MoNet_PROTEINS_full.json' &
# python $code --dataset $dataset --gpu_id 2 --config 'configs/TUs_graph_classification_GAT_PROTEINS_full.json' &
# python $code --dataset $dataset --gpu_id 3 --config 'configs/TUs_graph_classification_3WLGNN_PROTEINS_full.json' &
# wait" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --config 'configs/TUs_graph_classification_RingGNN_PROTEINS_full.json' &
# wait" C-m
# tmux send-keys "tmux kill-session -t benchmark_TUs_graph_classification" C-m

