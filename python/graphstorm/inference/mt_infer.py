"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GNN inferrer for multi-task learning in GraphStorm
"""
import os
import time
import logging
from typing import Any, Dict, Optional

import torch as th

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION)
from ..dataloading import GSgnnMultiTaskDataLoader
from ..eval.evaluator import GSgnnMultiTaskEvaluator
from .graphstorm_infer import GSInferrer
from ..model.utils import save_full_node_embeddings as save_gsgnn_embeddings
from ..model.utils import (save_node_prediction_results,
                           save_edge_prediction_results,
                           save_relation_embeddings)
from ..model.utils import NodeIDShuffler
from ..model import (do_full_graph_inference,
                     do_mini_batch_inference,
                     gen_emb_for_nfeat_reconstruct)
from ..model.multitask_gnn import multi_task_mini_batch_predict
from ..model.lp_gnn import run_lp_mini_batch_predict

from ..model.edge_decoder import LinkPredictDistMultDecoder

from ..utils import sys_tracker, get_rank, barrier

class GSgnnMultiTaskLearningInferrer(GSInferrer):
    """ Multi task inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to do multi task model inference.

    Parameters
    ----------
    model : GSgnnMultiTaskModel
        The GNN model for prediction.
    """

    # pylint: disable=unused-argument
    def infer(self, data,
              predict_test_loader: Optional[GSgnnMultiTaskDataLoader] = None,
              lp_test_loader: Optional[GSgnnMultiTaskDataLoader] = None,
              recon_nfeat_test_loader: Optional[GSgnnMultiTaskDataLoader] = None,
              recon_efeat_test_loader: Optional[GSgnnMultiTaskDataLoader] = None,
              save_embed_path=None,
              save_prediction_path=None,
              use_mini_batch_infer=False,
              node_id_mapping_file=None,
              edge_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch",
              infer_batch_size=1024):
        """ Do inference

        The inference can do three things:

        1. Generate node embeddings.
        2. Comput inference results for each tasks
        3. (Optional) Evaluate the model performance on a test set if given.

        Parameters
        ----------
        data: GSgnnData
            Graph data.
        predict_test_loader: GSgnnMultiTaskDataLoaders
            Test dataloader for prediction tasks.
        lp_test_loader: GSgnnMultiTaskDataLoaders
            Test dataloader for link prediction tasks.
        recon_nfeat_test_loader: GSgnnMultiTaskDataLoaders
            Test dataloader for node feature reconstruction tasks.
        recon_efeat_test_loader: GSgnnMultiTaskDataLoaders
            Test dataloader for edge feature reconstruction tasks.
        save_embed_path: str
            The path to save the node embeddings.
        save_prediction_path: str
            The path to save the prediction resutls.
        use_mini_batch_infer: bool
            Whether or not to use mini-batch inference when computing node embedings.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        edge_id_mapping_file: str
            Path to the file storing edge id mapping generated by the
            graph partition algorithm.
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.
        save_embed_format : str
            Specify the format of saved embeddings.
        infer_batch_size: int
            Specify the inference batch size when computing node embeddings
            with mini batch inference.

        .. versionchanged:: 0.4.0
            Add a new argument "recon_efeat_test_loader" for test
            dataloaders of edge feature reconstruction tasks.
        """
        do_eval = self.evaluator is not None
        sys_tracker.check('start inferencing')
        model = self._model
        model.eval()

        # All the tasks share the same GNN encoder so the fanouts are same
        # for different tasks.
        fanout = None
        if predict_test_loader is not None:
            for task_fanout in predict_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        elif lp_test_loader is not None:
            for task_fanout in lp_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        elif recon_nfeat_test_loader is not None:
            for task_fanout in recon_nfeat_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        elif recon_efeat_test_loader is not None:
            for task_fanout in recon_efeat_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        else:
            raise ValueError("All the test data loaders are None.")

        def gen_embs(edge_mask=None):
            # Generate node embeddings.
            # Note(xiangsx): In DistDGl, as we are using the
            # same DistTensor to save the node embeddings
            # so the node embeddings are updated inplace.
            if use_mini_batch_infer:
                embs = do_mini_batch_inference(
                    model, data, batch_size=infer_batch_size,
                    fanout=fanout,
                    edge_mask=edge_mask,
                    task_tracker=self.task_tracker)
            else:
                embs = do_full_graph_inference(
                    model, data,
                    fanout=fanout,
                    edge_mask=edge_mask,
                    task_tracker=self.task_tracker)
            return embs

        embs = gen_embs()
        sys_tracker.check('compute embeddings')
        device = self.device

        g = data.g
        # Note(xiangsx): Save embeddings should happen
        # before conducting prediction results.
        if save_embed_path is not None:
            logging.info("Saving node embeddings")
            node_norm_methods = model.node_embed_norm_methods
            # Save the original embs first
            save_gsgnn_embeddings(g,
                                  save_embed_path,
                                  embs,
                                  node_id_mapping_file=node_id_mapping_file,
                                  save_embed_format=save_embed_format)
            barrier()
            for task_id, norm_method in node_norm_methods.items():
                if norm_method is None:
                    continue
                normed_embs = model.normalize_task_node_embs(task_id, embs, inplace=False)
                save_embed_path = os.path.join(save_embed_path, task_id)
                save_gsgnn_embeddings(g,
                                      save_embed_path,
                                      normed_embs,
                                      node_id_mapping_file=node_id_mapping_file,
                                      save_embed_format=save_embed_format)
            sys_tracker.check('save embeddings')

            # save relation embedding if any for link prediction tasks
            if get_rank() == 0:
                decoders = model.task_decoders
                for task_id, decoder in decoders.items():
                    if isinstance(decoder, LinkPredictDistMultDecoder):
                        rel_emb_path = os.path.join(save_embed_path, task_id)
                        os.makedirs(rel_emb_path, exist_ok=True)
                        save_relation_embeddings(rel_emb_path, decoder)

        barrier()

        # As re-computing node embeddings, for reconstruct node
        # feature evaluation and link prediction evaluation,
        # will directly update the underlying DistTensors,
        # we have to do the evaluation (prediction) of each
        # task in the following priority:
        # 1. node and edge prediction tasks (classificaiton/regression)
        # 2. node feature reconstruction (as it has the chance
        #    to reuse the node embeddings generated at the beginning)
        # 3. link prediction.
        pre_results: Dict[str, Any] = {}
        test_lengths = None
        if predict_test_loader is not None:
            # compute prediction results for node classification,
            # node regressoin, edge classification
            # and edge regression tasks.
            pre_results = \
                multi_task_mini_batch_predict(
                    model,
                    emb=embs,
                    dataloaders=predict_test_loader.dataloaders,
                    task_infos=predict_test_loader.task_infos,
                    device=device,
                    return_proba=return_proba,
                    return_label=do_eval)

        if recon_efeat_test_loader is not None:
            # We also need to compute test scores for edge feature reconstruction tasks.
            dataloaders = recon_efeat_test_loader.dataloaders
            task_infos = recon_efeat_test_loader.task_infos

            efeat_recon_results = \
                multi_task_mini_batch_predict(
                    model,
                    emb=embs,
                    dataloaders=dataloaders,
                    task_infos=task_infos,
                    device=self.device,
                    return_proba=return_proba,
                    return_label=True)
            pre_results.update(efeat_recon_results)

        if recon_nfeat_test_loader is not None:
            # We also need to compute test scores for node feature reconstruction tasks.
            dataloaders = recon_nfeat_test_loader.dataloaders
            task_infos = recon_nfeat_test_loader.task_infos

            with th.no_grad():
                # Note: We need to be careful here as embs is a mutable
                # object (a dictionary), use a mutable value
                # as default value for an argument is danerous.
                # pylint: disable=dangerous-default-value
                def nfrecon_gen_embs(skip_last_self_loop=False, node_embs=embs):
                    """ Generate node embeddings for node feature reconstruction
                    """
                    if skip_last_self_loop is True:
                        # Turn off the last layer GNN's self-loop
                        # to compute node embeddings.
                        model.gnn_encoder.skip_last_selfloop()
                        new_embs = gen_embs()
                        model.gnn_encoder.reset_last_selfloop()
                        return new_embs
                    else:
                        # If skip_last_self_loop is False
                        # we will not change the way we compute
                        # node embeddings.
                        if node_embs is not None:
                            # The embeddings have been computed
                            # when handling predict_tasks in L608
                            return node_embs
                        else:
                            return gen_embs()

                # Note(xiangsx): In DistDGl, as we are using the
                # same dist tensor, the node embeddings
                # are updated inplace.
                nfeat_embs = gen_emb_for_nfeat_reconstruct(model, nfrecon_gen_embs)

                nfeat_recon_results = \
                    multi_task_mini_batch_predict(
                        model,
                        emb=nfeat_embs,
                        dataloaders=dataloaders,
                        task_infos=task_infos,
                        device=self.device,
                        return_proba=return_proba,
                        return_label=True)
                pre_results.update(nfeat_recon_results)

        if lp_test_loader is not None:
            # We also need to compute test scores for link prediction tasks.
            dataloaders = lp_test_loader.dataloaders
            task_infos = lp_test_loader.task_infos

            with th.no_grad():
                for dataloader, task_info in zip(dataloaders, task_infos):
                    if dataloader is None:
                        # dataloader is None, skip
                        pre_results[task_info.task_id] = None
                        continue

                    # For link prediction, do evaluation task by task.
                    lp_test_embs = gen_embs(edge_mask=task_info.task_config.train_mask)
                    # normalize the node embedding if needed.
                    # we can do inplace normalization as embeddings are generated
                    # per lp task.
                    lp_test_embs = model.normalize_task_node_embs(task_info.task_id,
                                                                  lp_test_embs,
                                                                  inplace=True)

                    decoder = model.task_decoders[task_info.task_id]
                    ranking, test_lengths = run_lp_mini_batch_predict(
                        decoder, lp_test_embs, dataloader, device, return_batch_lengths=True)
                    pre_results[task_info.task_id] = (ranking, test_lengths)

        if do_eval:
            test_start = time.time()
            assert isinstance(self.evaluator, GSgnnMultiTaskEvaluator)

            val_score, test_score = self.evaluator.evaluate(
                pre_results,
                pre_results,
                0,
            )

            sys_tracker.check('run evaluation')
            if get_rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                       test_score=test_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        if save_prediction_path is not None:
            if predict_test_loader is None:
                logging.warning("There is no prediction tasks."
                                "Will skip saving prediction results.")
                return

            logging.info("Saving prediction results")
            target_ntypes = set()
            task_infos = predict_test_loader.task_infos
            dataloaders = predict_test_loader.dataloaders
            for task_info in task_infos:
                if task_info.task_type in \
                    [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                    target_ntypes.add(task_info.task_config.target_ntype)
                elif task_info.task_type in \
                    [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                    target_ntypes.add(task_info.task_config.target_etype[0][0])
                    target_ntypes.add(task_info.task_config.target_etype[0][2])
                else:
                    # task_info.task_type is BUILTIN_TASK_LINK_PREDICTION
                    # or BUILTIN_TASK_RECONSTRUCT_NODE_FEAT
                    # There is no prediction results.
                    continue

            nid_shuffler = NodeIDShuffler(g, node_id_mapping_file, list(target_ntypes)) \
                    if node_id_mapping_file else None

            for task_info, dataloader in zip(task_infos, dataloaders):
                task_id = task_info.task_id
                if task_id not in pre_results:
                    logging.debug("No Prediction results for %s",
                                  task_id)
                    continue

                # Save prediction results
                save_pred_path = os.path.join(save_prediction_path, task_id)
                if task_info.task_type in \
                    [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                    pred, _ = pre_results[task_id]
                    if pred is not None:
                        shuffled_preds = {}

                        target_ntype = task_info.task_config.target_ntype
                        pred_nids = dataloader.target_nidx[target_ntype]
                        if node_id_mapping_file is not None:
                            pred_nids = nid_shuffler.shuffle_nids(
                                target_ntype, pred_nids)

                        shuffled_preds[target_ntype] = (pred, pred_nids)
                        save_node_prediction_results(shuffled_preds, save_pred_path)
                elif task_info.task_type in \
                    [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                    pred, _ = pre_results[task_id]
                    if pred is not None:
                        shuffled_preds = {}
                        target_etype = task_info.task_config.target_etype[0]
                        pred_eids = dataloader.target_eidx[target_etype]

                        pred_src_nids, pred_dst_nids = \
                            g.find_edges(pred_eids, etype=target_etype)

                        if node_id_mapping_file is not None:
                            pred_src_nids = nid_shuffler.shuffle_nids(
                                target_etype[0], pred_src_nids)
                            pred_dst_nids = nid_shuffler.shuffle_nids(
                                target_etype[2], pred_dst_nids)
                        shuffled_preds[target_etype] = \
                            (pred, pred_src_nids, pred_dst_nids)
                        save_edge_prediction_results(shuffled_preds, save_pred_path)
                else:
                    # There is no prediction results for link prediction
                    # and feature reconstruction
                    continue

        return
