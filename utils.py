import argparse
import warnings
import logging

import dgl, random, math, sys
import numpy as np

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Hypergraph representation learning')

    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index. Necessary for specifying the CUDA device.")
    parser.add_argument("--ns_method", type=str, default='SNS', help="Negative sampling method.")
    parser.add_argument('--dataset', type=str, default='citeseer', help='Name of dataset.')
    parser.add_argument('--data_path', type=str, default='./data', help='Data path.')
    parser.add_argument("--num_split", type=int, default=5, help="Number of split datasets.")

    parser.add_argument("--h_dim", type=int, default=256, help="Hidden Embedding dimensionality.")
    parser.add_argument("--proj_dim", type=int, default=256, help="Projection dimensionality.")
    parser.add_argument("--drop_incidence_rate", type=float, default=0.4, help="Incidence matrix dropping rate for augmentation.")
    parser.add_argument("--drop_feature_rate", type=float, default=0.4, help="Node feauture dropping rate for augmentation.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads of self-attention layer.")
    parser.add_argument("--num_layers", default=1, type=int, help='Number of self-attention layers')
    parser.add_argument('--augment_method', type=str, default='hyperedge', help='Hypergraph augmentation method: graph or hyperedge')
    parser.add_argument('--aggre_method', type=str, default='attention', help='Node aggregation method: attention or maxmin')
    parser.add_argument('--use_contrastive', type=int, default=1, help='Use Contrastive Loss: 1 (use) or 0 (no use)')

    parser.add_argument("--learning_rate", type=float, default=5e-04, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=5e-04, help='Dropout probability.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability.')
    parser.add_argument('--alpha', type=float, default=1, help='Hyperedge Normalization Factor of HNHN.')
    parser.add_argument('--beta', type=float, default=1, help='Node Normalization Factor of HNHN.')
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size for one process.")
    parser.add_argument("--clip", type=float, default=0.01, help="Upper/lower bound of parameters.")

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='Path for saving the trained model')

    args = parser.parse_args()

    return args

def print_summary(args):
    # Summary of training information
    print('========================================== Training Summary ==========================================')
    print ('    - GPU INDEX = %s' % (args.gpu_index))
    print ('    - DATASET = %s' % (args.dataset))
    print ('    - NUM SPLITS = %s' % (args.num_split))
    print ('    - AUGMENT METHOD = %s' % (args.augment_method))
    print ('    - NODE AGGREGATE METHOD = %s' % (args.aggre_method))
    print ('    - USE CONTRASTIVE = %s' % (args.use_contrastive))
    print ('    - NS METHOD = %s' % (args.ns_method))
    print (' ')
    print ('    - HIDDEN DIM = %s' % (args.h_dim))
    print ('    - PROJECTION DIM = %s' % (args.proj_dim))
    print ('    - NODE FEATURE DROP RATE = %s' % (args.drop_feature_rate))
    print ('    - INCIDENCE DROP RATE = %s' % (args.drop_incidence_rate))
    print ('    - NUM LAYERS = ' + str(args.num_layers))
    print ('    - NUM HEADS = ' + str(args.num_heads))
    print (' ')
    print ('    - NUM EPOCHS = ' + str(args.num_epochs))
    print ('    - BATCH SIZE = %s' % str(args.batch_size))
    print ('    - LEARNING RATE = ' + str(args.learning_rate))
    print ('    - WEIGHT DECAY = %s' % (args.weight_decay))
    print ('    - GRADIENT CLIP = ' + str(args.clip))
    print ('    - DROPOUT = %s' % (args.dropout))
    print (' ')


def gen_DGLGraph_with_droprate(ground, drop_rate, method='hyperedge'):
    he, hv = [], []
    total_he, total_hv = [], []
    for i, hedge in enumerate(ground): # hyperedge: a set of nodes

        if method == 'graph':
            for v in hedge:
                if np.random.binomial(1, (1.0 - drop_rate), 1) == 1:
                    he.append(i)
                    hv.append(v)

                total_he.append(i)
                total_hv.append(v)

        elif method == 'hyperedge':
            aug_hedge = edge_augmentation(hedge, drop_rate)

            for v in aug_hedge :
                he.append(i)
                hv.append(v)

            for v in hedge : # for indices of the original hypergraph
                total_he.append(i)
                total_hv.append(v)

        else:
            sys.exit( "Wrong Augmentation Name! 'graph or hyperedge'")

    data_dict = {
        ('node', 'in', 'hedge'): (hv, he),
        ('hedge', 'con', 'node'): (he, hv)
    }
    num_nodes_dict = {'node': len(set(total_hv)), 'hedge': len(set(total_he))} # the number of nodes and hyperedges in the original hypergraph

    return dgl.heterograph(data_dict, num_nodes_dict)



def edge_augmentation(hedge, drop_rate):
    aug_edge_size = max(math.ceil(len(hedge) * (1.0 - drop_rate)), 1)
    aug_edge = random.sample(hedge, aug_edge_size)

    return aug_edge


def gen_feature_mask(drop_rate):
    mask = dgl.FeatMask(p=drop_rate, node_feat_names=['node'], edge_feat_names=None)
    return mask


def get_num_iters(data_dict, batch_size: int, label: str = 'Train'):
    if label == 'Train':
        train_iters = math.ceil(len(data_dict["train_only_pos"] + data_dict["ground_train"])/batch_size)
        val_pos_iters = math.ceil(len(data_dict["valid_only_pos"] + data_dict["ground_valid"])/batch_size)
        val_neg_iters = math.ceil(len(data_dict["valid_sns"])/batch_size)

        return train_iters, val_pos_iters, val_neg_iters

    elif label == 'Test':
        test_pos_iters = math.ceil(len(data_dict["test_pos"])/batch_size)
        test_neg_iters = math.ceil(len(data_dict["test_sns"])/batch_size)

        return test_pos_iters, test_neg_iters

    else:
        sys.exit( "Wrong Label Name! 'Train or Test'")


