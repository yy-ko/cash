import torch, sys
from torchtext.data.functional import to_map_style_dataset

import numpy as np
from collections import defaultdict


def get_datainfo(args, device):
    dataset = args.dataset
    dim_hidden = args.h_dim
    alpha = args.alpha
    beta = args.beta

    data_info = {}
    data_path = './data/' + dataset + '.pt'

    data_dict = torch.load(data_path)
    NodeEdgePair = torch.LongTensor(data_dict['NodeEdgePair'])
    EdgeNodePair = torch.LongTensor(data_dict['EdgeNodePair'])

    data_info['num_nodes'] = data_dict['N_nodes']
    data_info['num_hyperedges'] = data_dict['N_edges']
    data_info['node_degree'] = data_dict['nodewt'] # degree of a node (# of hyperedges)
    data_info['hyperedge_size'] = data_dict['edgewt'] # hyperedge size (# of nodes in a hyperedge)

    node_feat = data_dict['node_feat'] # initial node feature
    data_info['input_dim'] = node_feat.shape[-1] # input embedding dimensionality

    data_info['incidence_matrix'] = torch.zeros(data_info['num_hyperedges'], data_info['num_nodes'])
    for he_idx, n_idx in EdgeNodePair:
        data_info['incidence_matrix'][he_idx, n_idx] = 1

    data_info['node_feat'] = torch.from_numpy(node_feat.astype(np.float32)).to(device) # num_nodes X input_dim
    data_info['hyperedge_feat'] = torch.ones(data_info['num_hyperedges'], dim_hidden).to(device) # num_hyperedge X dim_hidden


    # HNHN terms
    node_norm = torch.Tensor([(1/w if w > 0 else 1) for w in data_info['node_degree']]).unsqueeze(-1).to(device)
    edge_norm = torch.Tensor([(1/w if w > 0 else 1) for w in data_info['hyperedge_size']]).unsqueeze(-1).to(device)

    node2sum = defaultdict(list)
    edge2sum = defaultdict(list)
    
    node_normalized = torch.zeros(data_info['num_nodes']) 
    edge_normalized = torch.zeros(data_info['num_hyperedges']) 

    for i, (node_idx, edge_idx) in enumerate(NodeEdgePair.tolist()):
        e_norm = edge_norm[edge_idx]**alpha
        edge_normalized[edge_idx] = e_norm # D_{E, r, a}
        node2sum[node_idx].append(e_norm)  # D_{V, l, a}
        
        n_norm = node_norm[node_idx]**beta
        node_normalized[node_idx] = n_norm # D_{E, l, b}
        edge2sum[edge_idx].append(n_norm) # D_{V, r, b}

    node_normalized_sum = torch.zeros(data_info['num_nodes']) 
    edge_normalized_sum = torch.zeros(data_info['num_hyperedges']) 
    for node_idx, norm_list in node2sum.items():
        node_normalized_sum[node_idx] = sum(norm_list)
    for edge_idx, norm_list in edge2sum.items():
        edge_normalized_sum[edge_idx] = sum(norm_list)

    node_normalized_sum[node_normalized_sum==0] = 1 # ?
    edge_normalized_sum[edge_normalized_sum==0] = 1

    # for node update
    data_info['edge_normalized'] = torch.Tensor(edge_normalized).unsqueeze(-1).to(device)
    data_info['node_normalized_sum'] = torch.Tensor(node_normalized_sum).unsqueeze(-1).to(device)
    # for edge update
    data_info['node_normalized'] = torch.Tensor(node_normalized).unsqueeze(-1).to(device)
    data_info['edge_normalized_sum'] = torch.Tensor(edge_normalized_sum).unsqueeze(-1).to(device)

    return data_info



def get_dataloaders(data_dict, batch_size, device, ns_method, label='Train'):

    if label == 'Train':
        train_pos_data = data_dict["train_only_pos"] + data_dict["ground_train"]
        train_pos_labels = [1 for i in range(len(train_pos_data))] # training positive hyperedges
        train_pos_dataloader = BatchDataloader(train_pos_data, train_pos_labels, batch_size, device, is_Train=True)

        train_neg_data = None
        if ns_method == 'SNS':
            train_neg_data = data_dict["train_sns"]
        elif ns_method == 'MNS':
            train_neg_data = data_dict["train_mns"]
        elif ns_method == 'CNS':
            train_neg_data = data_dict["train_cns"]
        elif ns_method == 'Mixed':
            d = len(data_dict["train_sns"]) // 3
            train_neg_data = data_dict["train_sns"][0:d] + data_dict["train_mns"][0:d] + data_dict["train_cns"][0:d]
        elif ns_method == 'OURS':
            train_neg_data = []
        else:
            sys.exit('Wrong NS method name')

        train_neg_labels = [0 for i in range(len(train_neg_data))] # training negative hyperedges
        train_neg_dataloader = BatchDataloader(train_neg_data, train_neg_labels, batch_size, device, is_Train=True)

        return train_pos_dataloader, train_neg_dataloader

    elif label == 'Valid':
        val_pos_data = data_dict["valid_only_pos"] + data_dict["ground_valid"]
        val_pos_labels = [1 for i in range(len(val_pos_data))] # validation positive hyperedges
        val_pos_dataloader = BatchDataloader(val_pos_data, val_pos_labels, batch_size, device, is_Train=False)

        val_neg_sns_data = data_dict["valid_sns"]
        val_neg_mns_data = data_dict["valid_mns"]
        val_neg_cns_data = data_dict["valid_cns"]

        val_neg_labels = [0 for i in range(len(val_neg_sns_data))] # validation negative hyperedges
        val_neg_sns_dataloader = BatchDataloader(val_neg_sns_data, val_neg_labels, batch_size, device, is_Train=False)
        val_neg_mns_dataloader = BatchDataloader(val_neg_mns_data, val_neg_labels, batch_size, device, is_Train=False)
        val_neg_cns_dataloader = BatchDataloader(val_neg_cns_data, val_neg_labels, batch_size, device, is_Train=False)

        return val_pos_dataloader, val_neg_sns_dataloader, val_neg_mns_dataloader, val_neg_cns_dataloader

    elif label == 'Test':
        test_pos_data = data_dict["test_pos"]
        test_pos_labels = [1 for i in range(len(test_pos_data))] # validation positive hyperedges
        test_pos_dataloader = BatchDataloader(test_pos_data, test_pos_labels, batch_size, device, is_Train=False)

        test_neg_sns_data = data_dict["test_sns"]
        test_neg_mns_data = data_dict["test_mns"]
        test_neg_cns_data = data_dict["test_cns"]

        test_neg_labels = [0 for i in range(len(test_neg_sns_data))] # validation negative hyperedges
        test_neg_sns_dataloader = BatchDataloader(test_neg_sns_data, test_neg_labels, batch_size, device, is_Train=False)
        test_neg_mns_dataloader = BatchDataloader(test_neg_mns_data, test_neg_labels, batch_size, device, is_Train=False)
        test_neg_cns_dataloader = BatchDataloader(test_neg_cns_data, test_neg_labels, batch_size, device, is_Train=False)

        return test_pos_dataloader, test_neg_sns_dataloader, test_neg_mns_dataloader, test_neg_cns_dataloader
    else:
        sys.exit('Invalid data labels: Train, Eval, Test')




class BatchDataloader(object):
    def __init__(self, hyperedges, labels, batch_size, device, is_Train=False):
        """Creates an instance of Hyperedge Batch Dataloader.
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
        """
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.is_Train = is_Train

        if is_Train:
            self.shuffle()

    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]

    def __iter__(self):
        self._cursor = 0
        return self

    def next(self):
        return self._next_batch()

    def _next_batch(self):
        ncursor = self._cursor+self.batch_size # next cursor position

        next_hyperedges = None
        next_labels = None

        if ncursor >= len(self.hyperedges): # end of each epoch
            next_hyperedges = self.hyperedges[self._cursor:]
            next_labels = self.labels[self._cursor:]
            self._cursor = 0

            if self.is_Train:
                self.shuffle() # data shuffling at every epoch

        else:
            next_hyperedges = self.hyperedges[self._cursor:self._cursor + self.batch_size]
            next_labels = self.labels[self._cursor:self._cursor + self.batch_size]
            self._cursor = ncursor % len(self.hyperedges)

        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in next_hyperedges]
        labels = torch.FloatTensor(next_labels).to(self.device)

        return hyperedges, labels


