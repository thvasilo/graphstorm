GSF_HOME=/graph-storm

mkdir /regression-tests-data
mkdir /regression-tests-data/ogbn-products-data
REG_DATA_PATH=/regression-tests-data/ogbn-products-data
export PYTHONPATH=${GSF_HOME}/python/

# Construct the graph with original features
python3 ${GSF_HOME}/tools/gen_ogb_dataset.py --savepath ${REG_DATA_PATH}/ogbn-products/ \
                                             --edge_pct 0.8 \
                                             --dataset ogbn-products

# Partition the graph
python3 -u $GSF_HOME/tools/partition_graph_lp.py --dataset ogbn-products \
                                                 --filepath ${REG_DATA_PATH}/ogbn-products/ \
                                                 --num_parts 1 \
                                                 --num_trainers_per_machine 4 \
                                                 --output ${REG_DATA_PATH}/ogb_products_lp_train_val_1p_4t