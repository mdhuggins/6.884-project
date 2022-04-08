import sys

import numpy as np
import torch
import torch as th
import torch.nn as nn

import dgl
from dgl.nn.pytorch import RelGraphConv
import torch.nn.functional as F



class RGCN(nn.Module):
    def __init__(self, num_nodes, e_dim, h_dim, out_dim, num_rels,
                 node_initialization, max_len,
                 regularizer="basis", num_bases=-1, dropout=0.,
                 self_loop=False, ns_mode=False):
        super(RGCN, self).__init__()

        self.max_len = max_len
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.num_nodes = num_nodes
        self.size_matcher = th.nn.Linear(self.e_dim * max_len, self.h_dim)

        if num_bases == -1:
            num_bases = num_rels
        self.emb = nn.Embedding(num_nodes, e_dim)
        self.emb.weight = nn.Parameter(th.tensor(node_initialization, dtype=th.float32), requires_grad=True)
        self.conv1 = RelGraphConv(h_dim, h_dim, num_rels, regularizer,
                                  num_bases, self_loop=self_loop)
        self.conv2 = RelGraphConv(h_dim, out_dim, num_rels, regularizer, num_bases, self_loop=self_loop)
        self.dropout = nn.Dropout(dropout)
        self.ns_mode = ns_mode

    def forward(self, g, nids=None):
        if self.ns_mode:
            # forward for neighbor sampling
            x = self.emb(g[0].srcdata[dgl.NID])
            h = self.conv1(g[0], x, g[0].edata[dgl.ETYPE], g[0].edata['norm'])
            h = self.dropout(F.relu(h))
            h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], g[1].edata['norm'])
            return h
        else:
            node_feats = g.ndata['feat']
            nf = self.emb(node_feats)
            x = th.reshape(nf , (node_feats.shape[0], -1))
            x = self.size_matcher(x)
            # x = self.emb.weight if nids is None else self.emb(nids)
            h = self.conv1(g, x, g.edata[dgl.ETYPE], g.edata['norm'])
            h = self.dropout(F.relu(h))
            h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata['norm'])
            return h


class LinkPredict(nn.Module):
    def __init__(self, in_dim, e_dim, h_dim, num_rels, num_bases=-1,
                 dropout=0.2, use_cuda=False, reg_param=0.001,
                 node_initialization=None, max_len=20, vocabulary=[]):
        super(LinkPredict, self).__init__()

        self.rgcn = RGCN(in_dim, e_dim, h_dim, h_dim, num_rels * 2, node_initialization,
                         max_len, regularizer="bdd", num_bases=num_bases,
                         dropout=dropout, self_loop=True)
        self.e_dim = e_dim
        self.reg_param = reg_param
        self.dropout = nn.Dropout(dropout)
        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

        self.embedding_matrix = node_initialization
        self.trained_embeddings = None

        self.pred_linear = nn.Linear(h_dim, 1)
        self.activation = nn.Sigmoid()

        self.vocabulary = vocabulary

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = self.pred_linear(s * r * o)
        return self.activation(score)

    def forward(self, g, nids):
        return self.dropout(self.rgcn(g, nids=nids))

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        score = self.calc_score(embed, triplets).reshape(-1)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

    def load_embeddings(self, word, checkpoint_path=None):
        if checkpoint_path:
            self.load_weights(checkpoint_path)
        try:
            i = self.vocabulary.index(word)
            return self.trained_embeddings[i]
        except:
            return th.zeros(1, self.e_dim)

    def set_trained_embeddings(self,path,filename):
        from components.kgg_lit import load_data
        samples = load_data(path, filename, -1)[0:100]
        from components.kgg_lit import create_graph
        G, edge_types, vocabulary, d = create_graph(samples)
        unique_v = np.unique([_[0] for _ in d]+[_[2] for _ in d])
        G = dgl.graph(([_[0] for _ in d], [_[2] for _ in d]), num_nodes=len(unique_v))
        self.eval()
        with torch.no_grad():
            nids = [torch.tensor([self.vocabulary.index(word)]) for word in vocabulary]
            G.ndata["feat"] = torch.stack(nids)
            self.trained_embeddings = self(G,nids)
        print(here)



class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, e_dim, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=3, dropout=0.1, use_self_loop=False, 
                 use_cuda=False, node_initialization=None, max_len=20):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.node_initialization = node_initialization
        self.max_len = max_len
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    storage_dev_id : int
        The device to store the weights of the layer.
    out_dev_id : int
        Device to return the output embeddings on.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 storage_dev_id,
                 out_dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.storage_dev_id = th.device( \
            storage_dev_id if storage_dev_id >= 0 else 'cpu')
        self.out_dev_id = th.device(out_dev_id if out_dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer, device=self.storage_dev_id)
                else:
                    sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    sparse_emb.cuda(self.storage_dev_id)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.empty([input_emb_size, self.embed_size],
                                              device=self.storage_dev_id))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

    @property
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.out_dev_id)

        # transfer input to the correct device
        type_ids = type_ids.to(self.storage_dev_id)
        node_tids = node_tids.to(self.storage_dev_id)

        # build locs first
        locs = [None for i in range(self.num_of_ntype)]
        for ntype in range(self.num_of_ntype):
            locs[ntype] = (node_tids == ntype).nonzero().squeeze(-1)
        for ntype in range(self.num_of_ntype):
            loc = locs[ntype]
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.out_dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.out_dev_id)
            else:
                embeds[loc] = features[ntype][type_ids[loc]].to(self.out_dev_id) @ self.embeds[str(ntype)].to(self.out_dev_id)

        return embeds