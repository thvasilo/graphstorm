"""GNN model for link prediction in GraphStorm"""
import abc
import torch as th

from .gnn import GSgnnModel, GSgnnModelBase

class GSgnnLinkPredictionModelInterface:
    """ The interface for GraphStorm link prediction model.

    This interface defines two main methods for training and inference.
    """
    @abc.abstractmethod
    def forward(self, input_nodes, blocks, pos_graph, neg_graph,
        node_feats, edge_feats, epoch=-1, total_steps=-1):
        """ The forward function for link prediction.

        This method is used for training. It takes a mini-batch, including
        the graph structure, node features and edge features and
        computes the loss of the model in the mini-batch.

        Parameters
        ----------
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        pos_graph : a DGLGraph
            The graph that contains the positive edges.
        neg_graph : a DGLGraph
            The graph that contains the negative edges.
        node_feats : dict of Tensors
            The input node features of the message passing graphs.
        edge_feats : dict of Tensors
            The input edge features of the message passing graphs.
        epoch: int
            Current training epoch
            Default -1 means epoch is not initialized. (e.g., in prediction)
        total_steps: int
            Current training steps (iterations)
            Default -1 means training step is not initialized. (e.g., in prediction)
        Returns
        -------
        The loss of prediction.
        """

class GSgnnLinkPredictionModelBase(GSgnnModelBase,  # pylint: disable=abstract-method
                                   GSgnnLinkPredictionModelInterface):
    """ The base class for link-prediction GNN

    When a user wants to define a link prediction GNN model and train the model
    in GraphStorm, the model class needs to inherit from this base class.
    A user needs to implement some basic methods including `forward`, `predict`,
    `save_model`, `restore_model` and `create_optimizer`.
    """


class GSgnnLinkPredictionModel(GSgnnModel, GSgnnLinkPredictionModelInterface):
    """ GraphStorm GNN model for link prediction

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnLinkPredictionModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    def forward(self, input_nodes, blocks, pos_graph,
        neg_graph, node_feats, _, epoch=-1, total_steps=-1):
        """ The forward function for link prediction.

        This model doesn't support edge features for now.
        """
        alpha_l2norm = self.alpha_l2norm
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats, epoch, total_steps)
        else:
            # GNN message passing
            encode_embs = self.compute_embed_step(blocks, node_feats, epoch, total_steps)

        # TODO add w_relation in calculating the score. The current is only valid for
        # homogenous graph.
        pos_score = self.decoder(pos_graph, encode_embs)
        neg_score = self.decoder(neg_graph, encode_embs)
        pred_loss = self.loss_func(pos_score, neg_score)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

def lp_mini_batch_predict(model, emb, loader, device):
    """ Perform mini-batch prediction.

        This function follows full-grain GNN embedding inference.
        After having the GNN embeddings, we need to perform mini-batch
        computation to make predictions on the GNN embeddings.

        Parameters
        ----------
        model : GSgnnModel
            The GraphStorm GNN model
        emb : dict of Tensor
            The GNN embeddings
        loader : GSgnnEdgeDataLoader
            The GraphStorm dataloader
        device: th.device
            Device used to compute test scores

        Returns
        -------
        dict of (list, list):
            Return a dictionary of edge type to
            (positive scores, negative scores)
    """
    decoder = model.decoder
    with th.no_grad():
        scores = {}
        for pos_neg_tuple, neg_sample_type in loader:
            score = \
                decoder.calc_test_scores(
                    emb, pos_neg_tuple, neg_sample_type, device)
            for canonical_etype, s in score.items():
                # We do not concatenate pos scores/neg scores
                # into a single pos score tensor/neg score tensor
                # to avoid unnecessary data copy.
                if canonical_etype in scores:
                    scores[canonical_etype].append(s)
                else:
                    scores[canonical_etype] = [s]
    return scores
