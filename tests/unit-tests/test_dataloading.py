"""
Test functions and classes in the dataloading.py
"""
import tempfile
import numpy as np

import torch as th
import dgl
import pytest
from data_utils import generate_dummy_dist_graph

from graphstorm.dataloading import GSgnnNodeTrainData, GSgnnNodeInferData
from graphstorm.dataloading import GSgnnEdgeTrainData, GSgnnEdgeInferData
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import GSgnnNodeDataLoader, GSgnnEdgeDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER

from numpy.testing import assert_equal

def get_nonzero(mask):
    mask = mask[0:len(mask)]
    return th.nonzero(mask, as_tuple=True)[0]

def test_GSgnnEdgeData():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    tr_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    tr_single_etype = [("n0", "r1", "n1")]
    va_etypes = [("n0", "r1", "n1")]
    ts_etypes = [("n0", "r1", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=tmpdirname)
        tr_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=tr_etypes, eval_etypes=va_etypes,
                                     label_field='label')
        tr_data1 = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                      train_etypes=tr_etypes)
        # pass train etypes as None
        tr_data2 = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=None,
                                     label_field='label')
        # train etypes does not cover all etypes.
        tr_data3 = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=tr_single_etype,
                                     label_field='label')
        ev_data = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                     eval_etypes=va_etypes)
        # pass eval etypes as None
        ev_data2 = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                      eval_etypes=None)

    # successful initialization with default setting
    assert tr_data.train_etypes == tr_etypes
    assert tr_data.eval_etypes == va_etypes
    assert tr_data1.train_etypes == tr_etypes
    assert tr_data1.eval_etypes == tr_etypes
    assert ev_data.eval_etypes == va_etypes
    assert tr_data2.train_etypes == tr_data2.eval_etypes
    assert tr_data2.train_etypes == dist_graph.canonical_etypes
    assert tr_data3.train_etypes == tr_single_etype
    assert tr_data3.eval_etypes == tr_single_etype
    assert ev_data2.eval_etypes == dist_graph.canonical_etypes

    # sucessfully split train/val/test idxs
    assert len(tr_data.train_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data.train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
    assert len(ev_data.train_idxs) == 0

    assert len(tr_data.val_idxs) == len(va_etypes)
    for etype in va_etypes:
        assert th.all(tr_data.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    assert len(tr_data1.val_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data1.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    assert len(ev_data.val_idxs) == 0

    assert len(tr_data.test_idxs) == len(ts_etypes)
    for etype in ts_etypes:
        assert th.all(tr_data.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
    assert len(tr_data1.test_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data1.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
    assert len(ev_data.test_idxs) == len(va_etypes)
    for etype in va_etypes:
        assert th.all(ev_data.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    # pass train etypes as None
    assert len(tr_data2.train_idxs) == len(dist_graph.canonical_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data2.train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
    for etype in tr_etypes:
        assert th.all(tr_data2.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    for etype in tr_etypes:
        assert th.all(tr_data2.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    # train etypes does not cover all etypes.
    assert len(tr_data3.train_idxs) == len(tr_single_etype)
    for etype in tr_single_etype:
        assert th.all(tr_data3.train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
    for etype in tr_single_etype:
        assert th.all(tr_data3.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    for etype in tr_single_etype:
        assert th.all(tr_data3.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    # pass eval etypes as None
    assert len(ev_data2.test_idxs) == 2
    for etype in dist_graph.canonical_etypes:
        assert th.all(ev_data2.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    labels = tr_data.get_labels({('n0', 'r1', 'n1'): [0, 1]})
    assert len(labels.keys()) == 1
    assert ('n0', 'r1', 'n1') in labels
    try:
        labels = tr_data.get_labels({('n0', 'r0', 'n1'): [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label
    try:
        labels = tr_data1.get_labels({('n0', 'r1', 'n1'): [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_GSgnnNodeData():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    tr_ntypes = ["n1"]
    va_ntypes = ["n1"]
    ts_ntypes = ["n1"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=tmpdirname)
        tr_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=tr_ntypes, eval_ntypes=va_ntypes,
                                     label_field='label')
        tr_data1 = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                      train_ntypes=tr_ntypes)
        ev_data = GSgnnNodeInferData(graph_name='dummy', part_config=part_config,
                                     eval_ntypes=va_ntypes)

    # successful initialization with default setting
    assert tr_data.train_ntypes == tr_ntypes
    assert tr_data.eval_ntypes == va_ntypes
    assert tr_data1.train_ntypes == tr_ntypes
    assert tr_data1.eval_ntypes == tr_ntypes
    assert ev_data.eval_ntypes == va_ntypes

    # sucessfully split train/val/test idxs
    assert len(tr_data.train_idxs) == len(tr_ntypes)
    for ntype in tr_ntypes:
        assert th.all(tr_data.train_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['train_mask']))
    assert len(ev_data.train_idxs) == 0

    assert len(tr_data.val_idxs) == len(va_ntypes)
    for ntype in va_ntypes:
        assert th.all(tr_data.val_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['val_mask']))
    assert len(tr_data1.val_idxs) == len(tr_ntypes)
    for ntype in tr_ntypes:
        assert th.all(tr_data1.val_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['val_mask']))
    assert len(ev_data.val_idxs) == 0

    assert len(tr_data.test_idxs) == len(ts_ntypes)
    for ntype in ts_ntypes:
        assert th.all(tr_data.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
    assert len(tr_data1.test_idxs) == len(tr_ntypes)
    for ntype in tr_ntypes:
        assert th.all(tr_data1.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
    assert len(ev_data.test_idxs) == len(va_ntypes)
    for ntype in va_ntypes:
        assert th.all(ev_data.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))

    labels = tr_data.get_labels({'n1': [0, 1]})
    assert len(labels.keys()) == 1
    assert 'n1' in labels
    try:
        labels = tr_data.get_labels({'n0': [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label
    try:
        labels = tr_data1.get_labels({'n1': [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("batch_size", [1, 10, 128])
def test_GSgnnAllEtypeLinkPredictionDataLoader(batch_size):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    tr_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=tr_etypes, label_field='label')

    # successful initialization with default setting
    assert lp_data.train_etypes == tr_etypes
    dataloader = GSgnnAllEtypeLinkPredictionDataLoader(
        lp_data,
        target_idx=lp_data.train_idxs,
        fanout=[],
        batch_size=batch_size,
        num_negative_edges=4,
        device='cuda:0',
        exclude_training_targets=False)

    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert "n0" in input_nodes
        assert "n1" in input_nodes

        etypes = pos_graph.canonical_etypes
        assert ("n0", "r1", "n1") in etypes
        assert ("n0", "r0", "n1") in etypes

        etypes = neg_graph.canonical_etypes
        assert ("n0", "r1", "n1") in etypes
        assert ("n0", "r0", "n1") in etypes
    th.distributed.destroy_process_group()

def test_node_dataloader():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n1'], label_field='label')

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False)
    all_nodes = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes.append(seeds['n1'])
    all_nodes = th.cat(all_nodes)
    assert_equal(all_nodes.numpy(), target_idx['n1'])

    # With data shuffling, the seed nodes should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=True)
    all_nodes1 = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes1.append(seeds['n1'])
    all_nodes1 = th.cat(all_nodes1)
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=True)
    all_nodes2 = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes2.append(seeds['n1'])
    all_nodes2 = th.cat(all_nodes2)
    assert not np.all(all_nodes1.numpy() == all_nodes2.numpy())

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_edge_dataloader():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')], label_field='label')

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False, remove_target_edge_type=False)
    all_edges = []
    for input_nodes, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges.append(batch_graph.edata[dgl.EID])
    all_edges = th.cat(all_edges)
    assert_equal(all_edges.numpy(), target_idx[('n0', 'r1', 'n1')])

    # With data shuffling, the seed edges should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=True, remove_target_edge_type=False)
    all_edges1 = []
    for input_nodes, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges1.append(batch_graph.edata[dgl.EID])
    all_edges1 = th.cat(all_edges1)
    all_edges2 = []
    for input_nodes, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges2.append(batch_graph.edata[dgl.EID])
    all_edges2 = th.cat(all_edges2)
    assert not np.all(all_edges1.numpy() == all_edges2.numpy())

    ################### Test removing target edges #######################
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False, remove_target_edge_type=True,
                                     reverse_edge_types_map={('n0', 'r1', 'n1'): ('n0', 'r0', 'n1')})
    all_edges = []
    for input_nodes, batch_graph, blocks in dataloader:
        # All edge types are excluded, so the block doesn't have any edges.
        assert blocks[0].number_of_edges() == 0

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_lp_dataloader():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')])

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], 10, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    all_edges = []
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert len(pos_graph.etypes) == 1
        assert 'r1' in pos_graph.etypes
        all_edges.append(pos_graph.edata[dgl.EID])
    all_edges = th.cat(all_edges)
    assert_equal(all_edges.numpy(), target_idx[('n0', 'r1', 'n1')])

    # With data shuffling, the seed edges should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], 10, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    all_edges1 = []
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert len(pos_graph.etypes) == 1
        assert 'r1' in pos_graph.etypes
        all_edges1.append(pos_graph.edata[dgl.EID])
    all_edges1 = th.cat(all_edges1)
    all_edges2 = []
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert len(pos_graph.etypes) == 1
        assert 'r1' in pos_graph.etypes
        all_edges2.append(pos_graph.edata[dgl.EID])
    all_edges2 = th.cat(all_edges2)
    assert not np.all(all_edges1.numpy() == all_edges2.numpy())

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

# initialize the torch distributed environment
@pytest.mark.parametrize("batch_size", [1, 10, 128])
@pytest.mark.parametrize("num_negative_edges", [1, 16, 128])
def test_GSgnnLinkPredictionTestDataLoader(batch_size, num_negative_edges):
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=test_etypes, label_field='label')
        g = lp_data.g

        dataloader = GSgnnLinkPredictionTestDataLoader(
            lp_data,
            target_idx=lp_data.train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges)

        total_edges = {etype: len(lp_data.train_idxs[etype]) for etype in test_etypes}
        num_pos_edges = {etype: 0 for etype in test_etypes}
        for pos_neg_tuple, sample_type in dataloader:
            assert sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER
            assert isinstance(pos_neg_tuple, dict)
            assert len(pos_neg_tuple) == 1
            for canonical_etype, pos_neg in pos_neg_tuple.items():
                assert len(pos_neg) == 4
                pos_src, neg_src, pos_dst, neg_dst = pos_neg
                assert pos_src.shape == pos_dst.shape
                assert pos_src.shape[0] == batch_size \
                    if num_pos_edges[canonical_etype] + batch_size < total_edges[canonical_etype] \
                    else total_edges[canonical_etype] - num_pos_edges[canonical_etype]
                eid = lp_data.train_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else lp_data.train_idxs[canonical_etype] \
                        [num_pos_edges[canonical_etype]:]
                src, dst = g.find_edges(eid, etype=canonical_etype)

                assert_equal(pos_src.numpy(), src.numpy())
                assert_equal(pos_dst.numpy(), dst.numpy())
                num_pos_edges[canonical_etype] += batch_size
                assert neg_dst.shape[0] == pos_src.shape[0]
                assert neg_dst.shape[1] == num_negative_edges
                assert th.all(neg_dst < g.number_of_nodes(canonical_etype[2]))

                assert neg_src.shape[0] == pos_src.shape[0]
                assert neg_src.shape[1] == num_negative_edges
                assert th.all(neg_src < g.number_of_nodes(canonical_etype[0]))

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

# initialize the torch distributed environment
@pytest.mark.parametrize("batch_size", [1, 10, 128])
@pytest.mark.parametrize("num_negative_edges", [1, 16, 128])
def test_GSgnnLinkPredictionJointTestDataLoader(batch_size, num_negative_edges):
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=test_etypes, label_field='label')
        g = lp_data.g

        dataloader = GSgnnLinkPredictionJointTestDataLoader(
            lp_data,
            target_idx=lp_data.train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges)

        total_edges = {etype: len(lp_data.train_idxs[etype]) for etype in test_etypes}
        num_pos_edges = {etype: 0 for etype in test_etypes}
        for pos_neg_tuple, sample_type in dataloader:
            assert sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER
            assert isinstance(pos_neg_tuple, dict)
            assert len(pos_neg_tuple) == 1
            for canonical_etype, pos_neg in pos_neg_tuple.items():
                assert len(pos_neg) == 4
                pos_src, neg_src, pos_dst, neg_dst = pos_neg
                assert pos_src.shape == pos_dst.shape
                assert pos_src.shape[0] == batch_size \
                    if num_pos_edges[canonical_etype] + batch_size < total_edges[canonical_etype] \
                    else total_edges[canonical_etype] - num_pos_edges[canonical_etype]
                eid = lp_data.train_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else lp_data.train_idxs[canonical_etype] \
                        [num_pos_edges[canonical_etype]:]
                src, dst = g.find_edges(eid, etype=canonical_etype)
                assert_equal(pos_src.numpy(), src.numpy())
                assert_equal(pos_dst.numpy(), dst.numpy())
                num_pos_edges[canonical_etype] += batch_size
                assert len(neg_dst.shape) == 1
                assert neg_dst.shape[0] == num_negative_edges
                assert th.all(neg_dst < g.number_of_nodes(canonical_etype[2]))

                assert len(neg_src.shape) == 1
                assert neg_src.shape[0] == num_negative_edges
                assert th.all(neg_src < g.number_of_nodes(canonical_etype[0]))

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

if __name__ == '__main__':
    test_GSgnnNodeData()
    test_GSgnnEdgeData()
    test_lp_dataloader()
    test_edge_dataloader()
    test_node_dataloader()
    test_GSgnnAllEtypeLinkPredictionDataLoader(10)
    test_GSgnnAllEtypeLinkPredictionDataLoader(1)
    test_GSgnnLinkPredictionTestDataLoader(1, 1)
    test_GSgnnLinkPredictionTestDataLoader(10, 20)
    test_GSgnnLinkPredictionJointTestDataLoader(1, 1)
    test_GSgnnLinkPredictionJointTestDataLoader(10, 20)
