# Training script examples for Link Prediction
This folder provides example yaml configurations for link prediction training tasks.
The configurations include:

  * ``ml_lp.yaml`` defines a link prediction task on the ``(user, rating, movie)`` edges. It uses a single-layer RGCN model as its graph encoder, and uses a dot product decoder.

  * ``ml_lp_text.yaml`` defines a link prediction task on the ``(user, rating, movie)`` edges. It uses a single-layer RGCN model as its graph encoder, and uses a dot product decoder. In addition, the training task will do **LM-GNN co-training**. A BERT model, i.e., ``bert-base-uncased``, is used to compute the text embeddings of ``movie`` nodes and ``user`` nodes on the fly. During training, GraphStorm will randomly select 2 nodes for each mini-batch to participate the gradient computation to tune the BERT models. For more detials, please refer to https://graphstorm.readthedocs.io/en/latest/advanced/language-models.html#fine-tune-lms-on-graph-data.

  * ``ml_lp_text_multiple_lm_models.yaml`` defines a link prediction task on the ``(user, rating, movie)`` edges. It uses a single-layer RGCN model as its graph encoder, and uses a dot product decoder. Similar to ``ml_lp_text.yaml``, it also defines a **LM-GNN co-training** task. But it uses different LM models for different node types. It uses the ``bert-base-uncased`` BERT model to compute the text embeddings of ``movie`` nodes and the ``albert-large-v1`` ALBERT model to compute the text embeddings of ``user`` nodes.

  * ``ml_lp_none_train_etype.yaml`` defines a link prediction task on edges of all edge types. It uses a single-layer RGCN model as its graph encoder, and uses a dot product decoder.

  * ``ml_lm_lp.yaml`` defines a link prediction task on the ``(user, rating, movie)`` edges. It uses a language model, i.e., the BERT model, as its graph encoder, and uses a dot product decoder. The training task will do **LM-Graph co-training**. The BERT model will compute the text embeddings of ``movie`` nodes and ``user`` nodes on the fly. And the graph structure is used as the suppervision to tune the BERT models. Specifically, during training, GraphStorm will randomly select 10 nodes for each mini-batch to participate the gradient computation to tune the BERT models. For more detials, please refer to https://graphstorm.readthedocs.io/en/latest/advanced/language-models.html#fine-tune-lms-on-graph-data.

  * ``ml_lm_input_lp.yaml``defines a link prediction task on the ``(user, rating, movie)`` edges. It uses a language model, i.e., the BERT model, plus an MLP layer as its graph encoder, and uses a dot product decoder. Similar to ``ml_lm_lp.yaml``, it also does **LM-Graph co-training**.

  * ``mag_lp.yaml`` defines a link prediction task on ``(author, writes, paper)`` edges on a partitioned ogbn-mag dataset. It uses a 2-layer RGCN model as its graph encoder, and uses a dot product decoder.

  * ``arxiv_lp.yaml`` defines a link prediction task on ``(node, interacts, node)`` edges on a partitioned ogbn-arxiv dataset. It uses a single-layer RGCN model as its graph encoder, and uses a dot product decoder.

The example inference configurations are in ``inference_scripts/lp_infer/README``.

The following example script shows how to launch a GraphStorm link prediction training task.
You may need to change the arguments according to your tasks.
For more detials please refer to https://graphstorm.readthedocs.io/en/latest/cli/model-training-inference/index.html.

```
python3 -m graphstorm.run.gs_link_prediction \
    --workspace graphstorm/training_scripts/gsgnn_lp \
    --num-trainers 4 \
    --num-servers 1 \
    --num-samplers 0 \
    --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json \
    --ip-config ip_list.txt \
    --ssh-port 2222 \
    --cf ml_lp.yaml \
    --save-model-path /data/gsgnn_lp_ml_dot/
```
The script loads a paritioned graph from ``/data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json`` and saves the trained model in ``/data/gsgnn_lp_ml_dot/``.

Note: All example movielens graph data are generated by ``tests/end2end-tests/create_data.sh``.
