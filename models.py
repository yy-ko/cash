import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dgl import DGLGraph
from dgl.transforms import FeatMask


class HypergraphEncoder(nn.Module): # based on HNHN (ICML'2020)
    def __init__(self, h_dim: int, input_dim: int, dropout: float, node_norm: float, node_norm_sum: float, edge_norm: float, edge_norm_sum: float):
        super(HypergraphEncoder, self).__init__()
        self.h_dim = h_dim
        self.input2hidden = nn.Linear(input_dim, h_dim) # input layer: input_dim X hidden_dim
        self.vtx_lin = nn.Linear(h_dim, h_dim) # (TODO): test needed

        self.node2edge = nn.Linear(h_dim, h_dim) # W_E + B_E
        self.edge2node = nn.Linear(h_dim, h_dim) # W_V + B_V

        # HNHN normalization terms
        self.node_norm = node_norm
        self.node_norm_sum = node_norm_sum
        self.edge_norm = edge_norm
        self.edge_norm_sum = edge_norm_sum

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.PReLU()


    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']

        return {'weight': weight}

    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        weight = nodes.mailbox['weight']
        fvalue = nodes.mailbox['Wh']
        aggr = torch.sum(weight * fvalue, dim=1)
        return {'h': aggr}


    def forward(self, g: DGLGraph, mask: FeatMask,  n_feat: Tensor, he_feat: Tensor):
        with g.local_scope():
            g = mask(g)
            nfeat = self.input2hidden(n_feat)
            efeat = he_feat

            g.ndata['reg_weight'] = {'node':self.node_norm, 'hedge':self.edge_norm}
            g.ndata['reg_sum'] = {'node':self.node_norm_sum, 'hedge':self.edge_norm_sum}

            for _ in range(1):
                # node --> hyperedge
                g.ndata['Wh'] = {'node' : self.node2edge(nfeat)}
                g.apply_edges(self.weight_fn, etype='in')
                g.update_all(self.message_func, self.reduce_func, etype='in')
                efeat = self.activation(g.ndata['h']['hedge'])

                # hyperedge --> node
                g.ndata['Wh'] = {'hedge' : self.edge2node(efeat)}
                g.apply_edges(self.weight_fn, etype='con')
                g.update_all(self.message_func, self.reduce_func, etype='con')
                nfeat = self.activation(g.ndata['h']['node'])

            return nfeat, efeat


class OurModel(nn.Module): # self-supervised learning for hyperedge prediction
    def __init__(self, encoder: HypergraphEncoder, proj_dim: int, node_aggr_info):
        super(OurModel, self).__init__()

        # 1. Hypergraph Encoder
        self.encoder = encoder

        # 2. Projection
        self.fc1_node = nn.Linear(self.encoder.h_dim, proj_dim)
        self.fc2_node = nn.Linear(proj_dim, self.encoder.h_dim)
        self.fc1_hedge = nn.Linear(self.encoder.h_dim, proj_dim)
        self.fc2_hedge = nn.Linear(proj_dim, self.encoder.h_dim)

        # 3. Classifier for Hyperedge prediction
        self.classifier = nn.Linear(self.encoder.h_dim, 1)

        # Node Aggregation
        node_aggregate_layer = TransformerEncoderLayer(self.encoder.h_dim, node_aggr_info['nhead'], node_aggr_info['h_dim'], node_aggr_info['dropout'], batch_first=True)
        self.node_aggregation = TransformerEncoder(node_aggregate_layer, node_aggr_info['nlayer'])

        # for Contrastive loss
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, g: DGLGraph, mask: FeatMask,  n_feat: Tensor, he_feat: Tensor):
        # same input/output of hypergraph encoder
        n_feat, he_feat = self.encoder(g, mask, n_feat, he_feat)
        return n_feat, he_feat

    def node_projection(self, n_feat: Tensor):
        return self.fc2_node(F.elu(self.fc1_node(n_feat)))

    def hedge_projection(self, he_feat: Tensor):
        return self.fc2_hedge(F.elu(self.fc1_hedge(he_feat)))

    def cosine_similarity(self, feat1: Tensor, feat2: Tensor):
        cosine_similarity = self.cosine(feat1, feat2)
        return torch.mean(cosine_similarity)

    def __aggregate(self, embeddings: Tensor, method='attention'):
        aggregated_embedding = None

        # multi-head self attention 
        if method == 'attention':
            embeddings = self.node_aggregation(embeddings)
            aggregated_embedding, _ = torch.max(embeddings, dim=0)

        # max-min pooling 
        elif method == 'maxmin':
            max_val, _ = torch.max(embeddings, dim=0)
            min_val, _ = torch.min(embeddings, dim=0)
            aggregated_embedding = max_val - min_val
        else: 
            sys.exit('Wrong Node Aggregation Name')

        pred = F.sigmoid(self.classifier(aggregated_embedding))

        return pred, aggregated_embedding


    def aggregate(self, nfeat: Tensor, hedges: Tensor, mode='Train', method='attention'):
        preds = []
        if mode == 'Train':
            for he in hedges:
                feat = nfeat[he]
                pred, _ = self.__aggregate(feat, method)
                preds.append(pred)
            return torch.stack(preds).squeeze()

        elif mode == 'Eval':
            for he in hedges:
                feat = nfeat[he]
                pred, _ = self.__aggregate(feat, method)
                preds.append(pred.detach())
            return preds
        else:
            sys.exit('Wrong Mode Name')


# used in NHP, AHP (SIGIR'22)
# MaxMin aggregator for node embeddings
class MaxminAggregator(nn.Module):
    def __init__(self, layers):
        super(MaxminAggregator, self).__init__()

        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))

        self.classifier = nn.Sequential(*Layers)

    def forward(self, embeddings):
        max_val, _ = torch.max(embeddings, dim=0)
        min_val, _ = torch.min(embeddings, dim=0)

        aggregated_embedding = max_val - min_val

        pred = F.sigmoid(self.classifier(aggregated_embedding))

        return pred, aggregated_embedding


