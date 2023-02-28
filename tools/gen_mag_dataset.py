"""tools to test the preprocess of OGBN-MAG data
"""
import argparse
from graphstorm.data.ogbn_mag import OGBMAGTextFeatDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process OGBN-MAG data for GraphStorm')
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--edge_pct", type=float, default=1)
    args = parser.parse_args()
    # only for test
    dataset = OGBMAGTextFeatDataset(args.filepath,
                                    edge_pct=args.edge_pct)
    dataset.save_graphs(args.savepath)