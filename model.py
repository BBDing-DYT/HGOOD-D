from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import DenseGraphConv, GINConv, global_add_pool, GCNConv, global_mean_pool, global_max_pool
import torch
import torch.nn.functional as F
import torch.nn as nn
from hyptorch.nn import ToPoincare
from hyptorch.pmath import dist_matrix
from torch_geometric.nn import dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_sparse import coalesce, SparseTensor
import torch_scatter
# from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool
# from torch_geometric.utils import softmax
from torch_scatter import scatter
from wgin_conv import WGINConv
from view_learner import ViewLearner
from utils.utils import get_feat_mask
from graph_learners import *
import copy


class HCL(nn.Module):
    def __init__(self, args, input_dim, str_dim):
        super(HCL, self).__init__()
        self.device = args.device
        self.embedding_dim = args.hidden_dim
        self.maskfeat_rate_learner = args.maskfeat_rate_learner

        if args.readout == 'concat':
            self.embedding_dim = 128
        if args.type_learner == 'mlp':
            self.view_learner = ViewLearner(args.type_learner, MLP_learner(args.encoder_layers, input_dim, args.hidden_dim, args.dropout))
        elif args.type_learner == 'gnn':
            self.view_learner = ViewLearner(args.type_learner, GNN_learner(args.encoder_layers, input_dim+str_dim, args.hidden_dim, args.dropout))
        elif args.type_learner == 'att':
            self.view_learner = ViewLearner(args.type_learner, ATT_learner(args.encoder_layers, input_dim, args.hidden_dim, args.dropout))
        else:
            self.view_learner = None

        self.encoder_implicit = GIN(input_dim+str_dim, args.hidden_dim, args.encoder_layers, args.dropout, args.pooling, args.readout)
        self.encoder_explicit = GIN(input_dim+str_dim, args.hidden_dim, args.encoder_layers, args.dropout, args.pooling, args.readout)
        self.toPoincare = ToPoincare(c=args.hyper_c,
                                     ball_dim=self.embedding_dim,
                                     riemannian=False,
                                     clip_r=args.clip_r)
        self.proj_head_g1 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),nn.Linear(128, 128))
        self.proj_head_g1_sub = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),nn.Linear(128, 128))
        self.proj_head_g2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),nn.Linear(128, 128))
        # self.proj_head_n1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(),nn.Linear(self.embedding_dim, self.embedding_dim))
        # self.proj_head_n1_sub = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim))
        # self.proj_head_n2 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(),nn.Linear(self.embedding_dim, self.embedding_dim))
        self.hyp = nn.Sequential(nn.Linear(128, 128), self.toPoincare)
        # self.hyp = nn.Sequential(self.toPoincare)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x_em = torch.cat([data.x, data.x_s], dim=1)
        # x_em = data.x_s
        if self.view_learner:
            edge_logits = self.view_learner(x_em, data.edge_index)
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
            if self.maskfeat_rate_learner:
                mask, _ = get_feat_mask(x_em, self.maskfeat_rate_learner)
                features_v2 = x_em * (1 - mask)
            else:
                features_v2 = copy.deepcopy(x_em)
        else:
            features_v2 = x_em
            batch_aug_edge_weight = None
        g1, n1 = self.encoder_implicit(x_em, data.edge_index, None, data.batch)
        g2, n2 = self.encoder_explicit(x_em, data.edge_index, None, data.batch)
        g1_sub, n1_sub = self.encoder_implicit(x_em, data.edge_index, batch_aug_edge_weight, data.batch)
        g1_hyper = self.hyp(self.proj_head_g1(g1))
        g2_hyper = self.hyp(self.proj_head_g2(g2))
        g1_sub_hyper = self.hyp(self.proj_head_g1_sub(g1_sub))
        # n1_hyper = self.hyp(self.proj_head_n1(n1))
        # n2_hyper = self.hyp(self.proj_head_n2(n2))
        # n1_sub_hyper = self.hyp(self.proj_head_n1_sub(n1_sub))
        if self.view_learner:
            row, col = data.edge_index
            edge_batch = data.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight
            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")
            reg = []
            for b_id in range(data.num_graphs):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
        else:
            reg = None
        # return g1_hyper, g2_hyper, None, None, None, None, reg, batch_aug_edge_weight
        return g1_hyper, g2_hyper, g1_sub_hyper, None, None, None, reg, batch_aug_edge_weight
        # return g1_hyper, g2_hyper, g1_sub_hyper, n1_hyper, n2_hyper, n1_sub_hyper, reg, batch_aug_edge_weight

    def contrastive_loss(self, x0, x1, tau, hyper_c):
        # x0 and x1 - positive pair
        # tau - temperature
        # hyp_c - hyperbolic curvature, "0" enables sphere mode
        if hyper_c == 0:
            dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
        else:
            dist_f = lambda x, y: -dist_matrix(x, y)
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = dist_f(x0, x0) / tau - eye_mask
        logits00_numpy = logits00.detach().cpu().numpy()
        logits01 = dist_f(x0, x1) / tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target, reduction="none")
        stats = {
            "logits/min": logits01.min().item(),
            "logits/mean": logits01.mean().item(),
            "logits/max": logits01.max().item(),
            "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
            "logits": logits,
        }
        return loss

    @staticmethod
    def calc_loss_info(x0, x1, hyper_c, temperature=0.2):
        batch_size, _ = x0.size()
        if hyper_c == 0:
            dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
        else:
            dist_f = lambda x, y: -dist_matrix(x, y)
        logits01 = torch.exp(dist_f(x0, x1) / temperature)
        pos_sim = logits01[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (logits01.sum(dim=0) - pos_sim + 1e-20)
        loss_1 = pos_sim / (logits01.sum(dim=1) - pos_sim + 1e-20)
        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss
    def scoring_g(self, args, g, cluster_result):
        temperature = args.tau
        hyper_c = args.hyper_c
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']
        if hyper_c == 0:
            dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
        else:
            dist_f = lambda x, y: -dist_matrix(x, y, hyper_c)
        centers_level = torch.tensor(prototypes[0], dtype=torch.float32).to(args.device)
        sim_matrix = dist_f(g, centers_level)
        sim_matrix = torch.exp(sim_matrix / (temperature))
        torch.set_printoptions(edgeitems=torch.inf)
        v, id = torch.min(sim_matrix, 1)
        return v

class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, drop_ratio, pooling, readout):
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout
        self.drop_ratio = drop_ratio

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dim = dim
        self.pool = self.get_pool()

        for i in range(num_gc_layers):
            if i:
                mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                mlp = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            # conv = GINConv(mlp)
            conv = WGINConv(mlp)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, edge_weight, batch):
        xs = []
        for i in range(self.num_gc_layers):
            # x = F.relu(self.convs[i](x, edge_index))
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            # x = self.convs[i](x, edge_index, edge_weight)
            # # x = self.bns[i](x)
            # if i == self.num_gc_layers - 1:
            #     x = F.dropout(x, self.drop_ratio, training=self.training)
            # else:
            #     x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
            node_emb = xs[-1]
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
            node_emb = torch.cat(xs, dim=1)
        elif self.readout == 'add':
            graph_emb = 0
            node_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)
                node_emb += x

        return graph_emb, node_emb

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool

