#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_lp

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

echo "Test GraphStorm link prediction"

date

echo "**************standalone"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --cf ml_lp.yaml

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --num-epochs 1 --eval-frequency 300

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT & construct, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --num-epochs 1 --eval-frequency 300 --node-feat-name movie:title --construct-feat-ntype user --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false, no test"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_no_test_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --num-epochs 1 --eval-frequency 300 --logging-file /tmp/train_log.txt

error_and_exit $?

bst_cnt=$(grep "Best Test mrr: N/A" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "Test set is empty we should have Best Test mrr: N/A"
    exit -1
fi

rm /tmp/train_log.txt

mkdir -p /tmp/ML_lp_profile

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false, with profiling"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --num-epochs 1 --eval-frequency 300 --profile-path /tmp/ML_lp_profile

error_and_exit $?

cnt=$(ls /tmp/ML_lp_profile/*.csv | wc -l)
if test $cnt -lt 1
then
    echo "Cannot find the profiling files."
    exit -1
fi

rm -R /tmp/ML_lp_profile

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false, no train mask"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_no_edata_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --num-epochs 1 --eval-frequency 300 --no-validation true

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: uniform, exclude_training_targets: true"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --train-negative-sampler uniform --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-epochs 1 --eval-frequency 300

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: uniform, exclude_training_targets: true, gradient clip: 0.1, gradient norm type: 1"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --train-negative-sampler uniform --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-epochs 1 --eval-frequency 300 --max-grad-norm 0.1 --grad-norm-type 1

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: uniform, exclude_training_targets: true, norm: layer"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --train-negative-sampler uniform --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-epochs 1 --eval-frequency 300 --gnn-norm layer

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --use-node-embeddings true --use-mini-batch-infer false --num-epochs 1 --eval-frequency 300

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false, mlp layer between GNN layer: 1"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --use-node-embeddings true --use-mini-batch-infer false --num-epochs 1 --eval-frequency 300 --num-ffn-layers-in-gnn 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model --lp-embed-normalizer l2_norm --save-embed-path /data/gsgnn_lp_ml/emb/

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_l2_norm_emb.py --embout /data/gsgnn_lp_ml/emb/

error_and_exit $?

python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --use-node-embeddings true --use-mini-batch-infer false --num-epochs 1 --eval-frequency 300 --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-0/ --lp-embed-normalizer l2_norm --save-embed-path /data/gsgnn_lp_ml/full-infer-emb/

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_l2_norm_emb.py --embout /data/gsgnn_lp_ml/full-infer-emb/

error_and_exit $?

python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --use-node-embeddings true --use-mini-batch-infer true --num-epochs 1 --eval-frequency 300 --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-0/ --lp-embed-normalizer l2_norm --save-embed-path /data/gsgnn_lp_ml/mini-infer-emb/

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_l2_norm_emb.py --embout /data/gsgnn_lp_ml/mini-infer-emb/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false, mlp layer in input layer: 1"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --use-node-embeddings true --use-mini-batch-infer false --num-epochs 1 --eval-frequency 300 --num-ffn-layers-in-input 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --use-node-embeddings true --use-mini-batch-infer false --num-epochs 1 --eval-frequency 300 --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-0/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --num-epochs 1 --eval-frequency 300

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --num-epochs 1 --eval-frequency 300

error_and_exit $?


echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false, contrastive loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --num-epochs 1 --eval-frequency 300 --contrastive-loss-temperature 0.01 --lp-loss-func contrastive

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: false, contrastive loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type distmult --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --contrastive-loss-temperature 0.01 --lp-loss-func contrastive

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: RotatE, exclude_training_targets: false, contrastive loss, with report-eval-per-type"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type rotate --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --contrastive-loss-temperature 0.1 --lp-loss-func contrastive --report-eval-per-type True

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: TransE_l2, exclude_training_targets: false, contrastive loss, with report-eval-per-type, with early-stop consecutive_increase"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type transe_l2 --train-etype user,rating,movie movie,rating-rev,user --num-epochs 10 --eval-frequency 300 --contrastive-loss-temperature 1 --lp-loss-func contrastive --report-eval-per-type True --use-early-stop True --early-stop-burnin-rounds 3 --early-stop-rounds 2 --early-stop-strategy consecutive_increase

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: TransE_l1, exclude_training_targets: false, contrastive loss, with report-eval-per-type, with early-stop average_increase"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type transe_l1 --train-etype user,rating,movie movie,rating-rev,user --num-epochs 10 --eval-frequency 300 --contrastive-loss-temperature 1 --lp-loss-func contrastive --report-eval-per-type True --use-early-stop True --early-stop-burnin-rounds 3 --early-stop-rounds 2 --early-stop-strategy average_increase

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: false, adversarial cross entropy loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type distmult --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --lp-loss-func cross_entropy --adversarial-temperature 0.1

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: RotatE, exclude_training_targets: false, lp-embed-normalizer: l2_norm, adversarial cross entropy loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type rotate --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --lp-loss-func cross_entropy --adversarial-temperature 0.1 --lp-embed-normalizer l2_norm

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: TransE_L2, exclude_training_targets: false, adversarial cross entropy loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type transe_l2 --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --lp-loss-func cross_entropy --adversarial-temperature 0.1

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: TransE_L1, exclude_training_targets: false, lp-embed-normalizer: l2_norm, adversarial cross entropy loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type transe_l1 --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --lp-loss-func cross_entropy --adversarial-temperature 0.1 --lp-embed-normalizer l2_norm

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: false, bayesian personalized ranking loss"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --lp-decoder-type distmult --train-etype user,rating,movie movie,rating-rev,user --num-epochs 1 --eval-frequency 300 --lp-loss-func bpr

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false, lp-embed-normalizer: l2_norm, bayesian personalized ranking loss with edge weight"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --train-etype user,rating,movie --num-epochs 1 --eval-frequency 300 --lp-loss-func bpr --lp-embed-normalizer l2_norm --lp-edge-weight-for-loss rate

error_and_exit $?

date

echo 'Done'
