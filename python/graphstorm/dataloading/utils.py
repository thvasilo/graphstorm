""" Utils for data loading. """
import torch as th
import torch.distributed as dist

def trim_data(nids, device):
    """ In distributed traning scenario, we need to make sure that
        each worker has same number of batches. Otherwise the synchronization
        (barrier) is called diffirent times, which results in the worker
        with more batches hangs up.

        This function trims the nids to the same size for all workers.

        Parameters
        ----------
        nids: th.Tensor
            node ids
        device: th.device
            Device

        Returns
        -------
        Trimed nids: th.Tensor
    """
    # NCCL backend only supports GPU tensors, thus here we need to allocate it to gpu
    num_nodes = th.tensor(nids.numel()).to(device)
    assert num_nodes.is_cuda, "NCCL does not support CPU all_reduce"
    dist.all_reduce(num_nodes, dist.ReduceOp.MIN)
    min_num_nodes = int(num_nodes)
    nids_length = nids.shape[0]
    if min_num_nodes < nids_length:
        new_nids = nids[:min_num_nodes]
        print(f"Pad nids from {nids_length} to {min_num_nodes}")
    else:
        new_nids = nids
    assert new_nids.shape[0] == min_num_nodes
    return new_nids

def modify_fanout_for_target_etype(g, fanout, target_etypes):
    """ This function specifies a zero fanout for the target etype
        removing this etype from the message passing graph

        Parameters
        ----------
        g:
            The graph
        fanout:
            Sampling fanout
        target_etypes:
            Target etype to change the fanout

        Returns
        -------
        Modified fanout: list
    """

    edge_fanout_lis = []
    # The user can decide to not use the target etype for message passing.
    for fan in fanout:
        edge_fanout_dic = {}
        for etype in g.etypes:
            if g.to_canonical_etype(etype) not in target_etypes:
                edge_fanout_dic[etype] = fan if not isinstance(fan, dict) else fan[etype]
            else:
                print(f"Ignoring edges for {etype} etype")
                edge_fanout_dic[etype] = 0
        edge_fanout_lis.append(edge_fanout_dic)
    return edge_fanout_lis