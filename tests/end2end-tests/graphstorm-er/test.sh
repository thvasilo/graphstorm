#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_ep
echo "127.0.0.1" > ip_list.txt

cd $GS_HOME/inference_scripts/ep_infer
echo "127.0.0.1" > ip_list.txt

cat ip_list.txt

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

echo "Test GraphStorm edge regression"

date

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, with shrinkage loss"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --num-epochs 1 --regression-loss-func shrinkage

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, no test"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_no_test_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml  --num-epochs 1 --logging-file /tmp/train_log.txt

error_and_exit $?

bst_cnt=$(grep "Best Test rmse: N/A" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "Test set is empty we should have Best Test rmse: N/A"
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mse"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mse --num-epochs 1 --decoder-norm batch

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mae"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mae --num-epochs 1 --decoder-norm layer

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model and emb"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --save-model-path ./model/er_model/ --topk-model-to-save 3 --save-embed-path ./model/ml-emb/ --num-epochs 1 --save-model-frequency 1000

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, Backend nccl"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --num-epochs 1 --node-feat-name movie:title user:feat --backend nccl

error_and_exit $?

echo "**************dataset: MovieLens: ER, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, edge feat: user,rating,movie:feat inference: mini-batch"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --batch-size 64 --save-model-path /data/gsgnn_er_ml_ef/model/ --save-model-frequency 5 --eval-frequency 3  --num-epochs 1 --logging-file /tmp/train_log.txt

error_and_exit $?

python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/inference_scripts/ep_infer/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er_infer.yaml --use-mini-batch-infer true --restore-model-path /data/gsgnn_er_ml_ef/model/epoch-0/ --save-prediction-path /data/gsgnn_er_ml_ef/prediction/ --logging-file /tmp/log.txt --preserve-input True --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat

error_and_exit $?

## Emb Gen
python3 -m graphstorm.run.gs_gen_node_embedding --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --use-mini-batch-infer true --restore-model-path /data/gsgnn_er_ml_ef/model/epoch-0/ --save-embed-path /data/gsgnn_er_ml_ef/save-emb/ --logging-file /tmp/log.txt --logging-level debug

error_and_exit $?

cnt=$(ls -l /data/gsgnn_er_ml_ef/ | wc -l)
if test $cnt != 4
then
    echo "We save models, predictions, and embeddings."
    exit -1
fi

if [ -f "/data/gsgnn_er_ml_ef/save-emb/relation2id_map.json" ]; then
    echo "relation2id_map.json should not exist. It is saved when the model is trained with link prediction."
    exit -1
fi

rm -R /data/gsgnn_er_ml_ef/

rm -fr /tmp/*

echo "**************dataset: MovieLens: ER, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, edge feat: user,rating,movie:feat inference: mini-batch"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --batch-size 64 --save-model-path /data/gsgnn_er_ml_ef/model/ --save-model-frequency 5 --eval-frequency 3  --num-epochs 1 --logging-file /tmp/train_log.txt --model-encoder-type rgat

error_and_exit $?

python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/inference_scripts/ep_infer/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er_infer.yaml --use-mini-batch-infer true --restore-model-path /data/gsgnn_er_ml_ef/model/epoch-0/ --save-prediction-path /data/gsgnn_er_ml_ef/prediction/ --logging-file /tmp/log.txt --preserve-input True --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --model-encoder-type rgat

error_and_exit $?

## Emb Gen
python3 -m graphstorm.run.gs_gen_node_embedding --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --use-mini-batch-infer true --restore-model-path /data/gsgnn_er_ml_ef/model/epoch-0/ --save-embed-path /data/gsgnn_er_ml_ef/save-emb/ --logging-file /tmp/log.txt --logging-level debug --model-encoder-type rgat

error_and_exit $?

cnt=$(ls -l /data/gsgnn_er_ml_ef/ | wc -l)
if test $cnt != 4
then
    echo "We save models, predictions, and embeddings."
    exit -1
fi

if [ -f "/data/gsgnn_er_ml_ef/save-emb/relation2id_map.json" ]; then
    echo "relation2id_map.json should not exist. It is saved when the model is trained with link prediction."
    exit -1
fi

rm -R /data/gsgnn_er_ml_ef/

rm -fr /tmp/*

echo "**************dataset: MovieLens: ER, HGT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, edge feat: user,rating,movie:feat inference: mini-batch"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --batch-size 64 --save-model-path /data/gsgnn_er_ml_ef/model/ --save-model-frequency 5 --eval-frequency 3  --num-epochs 1 --logging-file /tmp/train_log.txt --model-encoder-type hgt

error_and_exit $?

python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/inference_scripts/ep_infer/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er_infer.yaml --use-mini-batch-infer true --restore-model-path /data/gsgnn_er_ml_ef/model/epoch-0/ --save-prediction-path /data/gsgnn_er_ml_ef/prediction/ --logging-file /tmp/log.txt --preserve-input True --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --model-encoder-type hgt

error_and_exit $?

## Emb Gen
python3 -m graphstorm.run.gs_gen_node_embedding --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ef_er_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --node-feat-name movie:title user:feat --edge-feat-name user,rating,movie:feat --use-mini-batch-infer true --restore-model-path /data/gsgnn_er_ml_ef/model/epoch-0/ --save-embed-path /data/gsgnn_er_ml_ef/save-emb/ --logging-file /tmp/log.txt --logging-level debug --model-encoder-type hgt

error_and_exit $?

cnt=$(ls -l /data/gsgnn_er_ml_ef/ | wc -l)
if test $cnt != 4
then
    echo "We save models, predictions, and embeddings."
    exit -1
fi

if [ -f "/data/gsgnn_er_ml_ef/save-emb/relation2id_map.json" ]; then
    echo "relation2id_map.json should not exist. It is saved when the model is trained with link prediction."
    exit -1
fi

rm -R /data/gsgnn_er_ml_ef/

rm -fr /tmp/*

date

echo 'Done'
