""" utils
"""
from .hf_bert import wrap_hf_bert
from .hf_bert import load_hfbert_model

BUILTIN_HF_BERT = "bert"
BUILTIN_LM_MODELS = [BUILTIN_HF_BERT]

def init_lm_model(lm_config, num_train=0, lm_infer_batchszie=16, profile=False):
    """ Init language model

    Parameters
    ----------
    lm_config: dict
        Language model config.
    num_train: int
        Number of trainable texts.
    lm_infer_batchszie: int
        Batch size used in lm model inference
    profile: bool
        If True, provide LM forward/backward profiling.

    Return
    ------

    """
    lm_model_type = lm_config["lm_type"]
    if lm_model_type == BUILTIN_HF_BERT:
        bert_model = load_hfbert_model(lm_config)
        lm_model = wrap_hf_bert(bert_model, num_train, lm_infer_batchszie, profile)
    else:
        assert lm_model_type in BUILTIN_LM_MODELS, \
            f"Unsupported builtin language model {lm_model_type}"

    return lm_model

def get_lm_node_feats(g, lm_model, lm_ntypes):
    """ Collect language model related node features

    Parameters
    ----------
    g: graph
        Graph data.
    lm_model: xxx
        GS language model wrapper.
    lm_ntypes: list of str
        A list of node types.

    Return
    ------
    A dict of dict of distributed tensor.
        {Node type: {Feature name: Feature stored as distributed tensor}}
    """
    lm_feat_names = lm_model.lm_fnames
    lm_feats = {ntype: {lm_fname: g.nodes[ntype].data[lm_fname] for lm_fname in lm_feat_names}
        for ntype in lm_ntypes}
    return lm_feats
