# Partition a graph for a link prediction task

Partition a graph for a semantic matching task

filepath is the graph path
Specify num_trainers_per_machine larger than 1
num_parts is the number of machines number of partitions
output is the name of the partioned output folder
```
python3 partition_graph_lp.py --dataset query_asin_match --filepath qa_data_graph_v_1/ --num_parts 8 --num_trainers_per_machine 8 --output qa_train_v01_8p
```

# Partition paper100m
```
python3 /fsx-dev/xiangsx/home/workspace/graph-storm/tools/partition_graph.py --dataset ogbn-papers100m --filepath ./paper100m-processed-512/ --num_parts 8 --predict_ntypes "node" --balance_train --num_trainers_per_machine 8 --output /fsx-dev/xiangsx/home/workspace/graph-storm/training_scripts/gsgnn_nc/ogbn_papers100m_nc_8p_8t/
```
