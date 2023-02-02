date

mkdir -p /regression-tests-data
REG_DATA_PATH=/regression-tests-data

# create data for ogb_products nc regression tests
# 1. Construct the graph directly downloading from the OGB site.
python3 tools/gen_ogb_dataset.py --savepath ${REG_DATA_PATH}/ogbn-products-nc/ \
                                 --dataset ogbn-products \
                                 --retain_original_features true
# 2. partition the graph for test
python3 tools/partition_graph.py --dataset ogbn-products \
                                 --filepath ${REG_DATA_PATH}/ogbn-products-nc/ \
                                 --num_parts 1 \
                                 --num_trainers_per_machine 4 \
                                 --output ${REG_DATA_PATH}/ogb_products_nc_train_val_1p_4t

date
