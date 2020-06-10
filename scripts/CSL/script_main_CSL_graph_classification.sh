#!/bin/bash


############
# Usage
############

# bash script_main_CSL_graph_classification.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GatedGCN
#GAT
#MoNet
#GIN
#3WLGNN
#RingGNN



############
# CSL 
############

seed0=41
code=main_CSL_graph_classification.py 
dataset=CSL
tmux new -s benchmark -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/CSL_graph_classification_MLP_CSL.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/CSL_graph_classification_GCN_CSL.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/CSL_graph_classification_GraphSage_CSL.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed0 --config 'configs/CSL_graph_classification_GatedGCN_CSL.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/CSL_graph_classification_GAT_CSL.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/CSL_graph_classification_MoNet_CSL.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/CSL_graph_classification_GIN_CSL.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/CSL_graph_classification_3WLGNN_CSL.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/CSL_graph_classification_RingGNN_CSL.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m











