import dgl
import numpy as np
import torch as th
import argparse
import time
from graphstorm.data import OGBTextFeatDataset
from graphstorm.data import MovieLens100kNCDataset
from graphstorm.data import ConstructedGraphDataset

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition DGL graphs for node and edge classification "
                                        + "or regression tasks")
    # dataset and file arguments
    argparser.add_argument("-d", "--dataset", type=str, required=True,
                           help="dataset to use")
    argparser.add_argument("--filepath", type=str, default=None)
    # node arguments
    argparser.add_argument('--predict_ntype', type=str, help='The node type for making prediction'
                           + 'Currently only support one node type.')
    argparser.add_argument('--ntask_type', type=str, default='classification', nargs='?',
                           choices=['classification', 'regression'],
                           help='The node prediction type. Only support either \"classsification\" or '
                                '\"regression\". Default is \"classification\"')
    argparser.add_argument('--nlabel_field', type=str, help='The field that stores label on nodes.'
                           + 'The format is \"nodetype:label\", e.g., paper:subject')
    # edge arguments
    argparser.add_argument('--predict_etype', type=str, help='The canonical edge type for making '
                           + 'prediction. The format is \"scr_ntype,etype,dst_ntype\". '
                           + 'Currently only support one edge type.')
    argparser.add_argument('--etask_type', type=str, default='classification',nargs='?',
                           choices=['classification', 'regression'],
                           help='The edge prediction type. Only support either \"classsification\" or '
                                '\"regression\". Default is \"classification\"')
    argparser.add_argument('--elabel_field', type=str, help='The field that stores label on edges.'
                           + 'The format is \"srcnodetype,etype,dstnodetype:label\", e.g., '
                           + '\"user,review,movie:stars\".')
    # label split arguments
    argparser.add_argument("--generate_new_node_split", type=lambda x:(str(x).lower() in ['true', '1']),
                           default=False, help="If we are splitting the data from scatch we should "
                           + "not do it by default.")
    argparser.add_argument("--generate_new_edge_split", type=lambda x:(str(x).lower() in ['true', '1']),
                           default=False, help="If we are splitting the data from scatch we should "
                           + "not do it by default.")
    argparser.add_argument('--train_pct', type=float, default=0.8,
                           help='The pct of train nodes/edges. Should be > 0 and < 1, and only work in '
                                + 'generating new split')
    argparser.add_argument('--val_pct', type=float, default=0.1,
                           help='The pct of validation nodes/edges. Should be > 0 and < 1, and only work in '
                                + 'generating new split')
    # graph modification arguments
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--retain_original_features',  type=lambda x: (str(x).lower() in ['true', '1']),
                           default=True, help= "If true we will use the original features either wise we "
                                               "will use the tokenized title or abstract"
                                                "for the ogbn datasets")
    # partition arguments
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    # output arguments
    argparser.add_argument('--output', type=str, default='data',
                           help='The output directory to store the partitioned results.')

    args = argparser.parse_args()
    print(args)
    start = time.time()

    constructed_graph = False

    # load graph data
    if args.dataset == 'ogbn-arxiv':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset,
                                     retain_original_features=args.retain_original_features)
    elif args.dataset == 'ogbn-products':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset,
                                     retain_original_features=args.retain_original_features)
    elif args.dataset == 'ogbn-papers100m':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset,
                                     retain_original_features=args.retain_original_features)
    elif args.dataset == 'movie-lens-100k':
        dataset = MovieLens100kNCDataset(args.filepath)
    elif args.dataset == 'movie-lens-100k-text':
        dataset = MovieLens100kNCDataset(args.filepath, use_text_feat=True)
    else:
        constructed_graph = True
        print("Loading user defined dataset " + str(args.dataset))
        dataset = ConstructedGraphDataset(args.dataset, args.filepath)

    # ------------------ Arguments sanity check ------------------
    # train and validation percentage check. Their sum should less or equal to 1.
    assert (args.train_pct + args.val_pct) <= 1, \
        "The sum of train and validation percentages should NOT larger than 1."

    # predict node types and edge types check. At least one argument should be given.
    pred_ntypes = args.predict_ntype.split(',') if args.predict_ntype is not None else None
    if pred_ntypes is None:
        try:
            pred_ntypes = [dataset.predict_category]
        except:
            pass
    pred_etypes = [tuple(args.predict_etype.split(','))] if args.predict_etype is not None else None
    if pred_etypes is None:
        try:
            pred_etypes = [dataset.target_etype]
        except:
            pass
    assert pred_ntypes is not None or pred_etypes is not None, \
        "For partition graph datasets, you must provide predict_ntype or predict_etype, or both"
    if pred_ntypes is not None:
        assert len(pred_ntypes) == 1, "We currently only support one node type prediction."
    if pred_etypes is not None:
        assert len(pred_etypes) == 1, "We currently only support one edge type prediction."

    # predict node type and node label field check. The two's node type should be the same
    if args.nlabel_field is not None:
        label_ntype, nlabel_field = args.nlabel_field.split(':')
    else:
        label_ntype = None
        nlabel_field = None
    if pred_ntypes is not None and constructed_graph:
        assert pred_ntypes[0] == label_ntype, 'The predict node type must have label associated.'

    # predict edge type and edge label filed check. The two's edge type should be the same
    if args.elabel_field is not None:
        label_etype, elabel_field = args.elabel_field.split(':')
        label_etype = tuple(label_etype.split(','))
    else:
        label_etype = None
        elabel_field = None
    if pred_etypes is not None and constructed_graph:
        assert pred_etypes[0] == label_etype, 'The predict edge type must have label associated.'

    # extract the DGL graph
    g = dataset[0]

    # set node label tensor type to meet required tasks
    if pred_ntypes is not None and label_ntype is not None:
        nlabel_tensor = g.nodes[label_ntype].data.get(nlabel_field)
        assert nlabel_tensor is not None, f'The given {label_ntype} node type does NOT have data '\
                                         + f'filed {nlabel_field} ...'
        if args.ntask_type == 'classification':
            g.nodes[label_ntype].data[nlabel_field] = nlabel_tensor.to(th.int64)
        elif args.ntask_type == 'regression':
            g.nodes[label_ntype].data[nlabel_field] = nlabel_tensor.to(th.float32)
        else:
            raise NotImplementedError('Currently only support either node classification or '
                                      + 'node regression ...')

    # set edge label tensor type to meet required tasks
    if pred_etypes is not None and label_etype is not None:
        elabel_tensor = g.edges[label_etype].data.get(elabel_field)
        assert elabel_tensor is not None, f'The given {label_etype} edge type does NOT have data '\
                                          + f'field: {elabel_field} ...'
        if args.etask_type == 'classification':
            g.edges[label_etype].data[elabel_field] = elabel_tensor.to(th.int64)
        elif args.etask_type == 'regression':
            g.edges[label_etype].data[elabel_field] = elabel_tensor.to(th.float32)
        else:
            raise NotImplementedError('Currently only support either edge classification or '
                                      + 'edge regression ...')
    # add reverse edges
    if args.undirected:
        print("Creating reverse edges ...")
        edges = {}
        for src_ntype, etype, dst_ntype in g.canonical_etypes:
            src, dst = g.edges(etype=(src_ntype, etype, dst_ntype))
            edges[(src_ntype, etype, dst_ntype)] = (src, dst)
            edges[(dst_ntype, etype + '-rev', src_ntype)] = (dst, src)
        num_nodes_dict = {}
        for ntype in g.ntypes:
            num_nodes_dict[ntype] = g.num_nodes(ntype)
        new_g = dgl.heterograph(edges, num_nodes_dict)
        # Copy the node data and edge data to the new graph. The reverse edges will not have data.
        for ntype in g.ntypes:
            for name in g.nodes[ntype].data:
                new_g.nodes[ntype].data[name] = g.nodes[ntype].data[name]
        for etype in g.canonical_etypes:
            for name in g.edges[etype].data:
                new_g.edges[etype].data[name] = g.edges[etype].data[name]
        g = new_g
        new_g = None

    # Split train/val/test sets for each predicted node type
    if args.generate_new_node_split:
        if pred_ntypes is not None:
            assert len(pred_ntypes) == 1, 'Currently only support one node type for prediction'
            for ntype in pred_ntypes:
                num_nodes = g.number_of_nodes(ntype)
                # shuffled node ids and extract train/val/test indexes
                shuffled_index = np.random.permutation(np.arange(num_nodes))
                train_idx = shuffled_index[: int(num_nodes * args.train_pct)]
                val_idx = shuffled_index[int(num_nodes * args.train_pct): \
                                        int(num_nodes * (args.train_pct + args.val_pct))]
                test_idx = shuffled_index[int(num_nodes * (args.train_pct + args.val_pct)): ]
                # build boolean masks
                train_mask = th.zeros((num_nodes,), dtype=th.bool)
                train_mask[train_idx] = True
                val_mask = th.zeros((num_nodes,), dtype=th.bool)
                val_mask[val_idx] = True
                test_mask = th.zeros((num_nodes,), dtype=th.bool)
                test_mask[test_idx] = True
                # set split masks as predict node's features
                g.nodes[ntype].data['train_mask'] = train_mask
                g.nodes[ntype].data['val_mask'] = val_mask
                g.nodes[ntype].data['test_mask'] = test_mask
        else:
            raise Exception('There is no predicted node type to split. Please set the '
                            +'predict_ntype argument ......')

    # Split train/val/test sets for each predicted node type
    if args.generate_new_edge_split:
        if pred_etypes is not None:
            assert len(pred_etypes) == 1, 'Currently only support one edge type for prediction'
            for etype in pred_etypes:
                num_edges = g.num_edges(etype)
                # shuffled edge ids and extract train/val/test indexes
                shuffled_index = np.random.permutation(np.arange(num_edges))
                train_idx = shuffled_index[: int(num_edges * args.train_pct)]
                val_idx = shuffled_index[int(num_edges * args.train_pct): \
                                        int(num_edges * (args.train_pct + args.val_pct))]
                test_idx = shuffled_index[int(num_edges * (args.train_pct + args.val_pct)): ]
                # build boolean masks
                train_mask = th.zeros((num_edges,), dtype=th.bool)
                train_mask[train_idx] = True
                val_mask = th.zeros((num_edges,), dtype=th.bool)
                val_mask[val_idx] = True
                test_mask = th.zeros((num_edges,), dtype=th.bool)
                test_mask[test_idx] = True
                # set split masks as predict edge's features
                g.edges[etype].data['train_mask'] = train_mask
                g.edges[etype].data['val_mask'] = val_mask
                g.edges[etype].data['test_mask'] = test_mask
        else:
            raise Exception('There is no predicted edge type to split. Please set the '
                            +'predict_etype argument ......')

    # Output general graph information
    print(f'load {args.dataset} takes {time.time() - start:.3f} seconds')
    print(f'\n|V|={g.number_of_nodes()}, |E|={g.number_of_edges()}\n')
    print('node types:', g.ntypes)
    for ntype in g.ntypes:
        for name in g.nodes[ntype].data:
            data = g.nodes[ntype].data[name]
            print(f'node \'{ntype}\' has data \'{name}\' of type {data.dtype}')
    print('edge types:', g.canonical_etypes)
    for etype in g.canonical_etypes:
        for name in g.edges[etype].data:
            data = g.edges[etype].data[name]
            print(f'edge \'{etype}\' has data \'{name}\' of type {data.dtype}')

    # Output split information if predicted node/edge types specified
    if pred_ntypes is not None:
        for ntype in pred_ntypes:
            train_total = th.sum(g.nodes[ntype].data['train_mask'])
            val_total = th.sum(g.nodes[ntype].data['val_mask']) if 'val_mask' in g.nodes[ntype].data else 0
            test_total = th.sum(g.nodes[ntype].data['test_mask']) if 'test_mask' in g.nodes[ntype].data else 0
            print(f'\ntraining target node type: \'{ntype}\', '
                  + f'train: {train_total}, valid: {val_total}, test: {test_total}')
    if pred_etypes is not None:
        for etype in pred_etypes:
            train_total = th.sum(g.edges[etype].data['train_mask'])
            val_total = th.sum(g.edges[etype].data['val_mask']) if 'val_mask' in g.edges[etype].data else 0
            test_total = th.sum(g.edges[etype].data['test_mask']) if 'test_mask' in g.edges[etype].data else 0
            print(f'\ntraining target edge type: \'{etype}\', '
                  + f'train: {train_total}, valid: {val_total}, test: {test_total}')

    if args.balance_train and pred_ntypes is not None:
        balance_ntypes = {category: g.nodes[category].data['train_mask'] for category in pred_ntypes}
    else:
        balance_ntypes = None

    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges,
                                    num_trainers_per_machine=args.num_trainers_per_machine)
