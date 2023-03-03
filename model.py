import torch.nn as nn
import torch
from gnn import GNN
import dgl
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x


class ConRelEncoder(nn.Module):
    def __init__(self, args):
        super(ConRelEncoder, self).__init__()
        self.args = args

        self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

        self.feat_mlp = MLP(args.ent_dim, args.ent_dim, args.ent_dim)

    def forward(self, g_bidir):
        with g_bidir.local_scope():
            num_edge = g_bidir.num_edges()
            etypes = g_bidir.edata['rel']
            g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.ent_dim).to(self.args.gpu)
            rh_idx = (g_bidir.edata['inv'] == 1)
            rt_idx = (g_bidir.edata['inv'] == 0)
            g_bidir.edata['ent_e'][rh_idx] = self.rel_head_emb[etypes[rh_idx]]
            g_bidir.edata['ent_e'][rt_idx] = self.rel_tail_emb[etypes[rt_idx]]

            message_func = dgl.function.copy_e('ent_e', 'msg')
            reduce_func = dgl.function.mean('msg', 'feat')
            g_bidir.update_all(message_func, reduce_func)
            g_bidir.edata.pop('ent_e')

            zero_idx = ((g_bidir.in_degrees() + g_bidir.out_degrees()) == 0)
            rand_feat = torch.Tensor(torch.sum(zero_idx), self.args.ent_dim).to(self.args.gpu)
            nn.init.xavier_normal_(rand_feat, gain=nn.init.calculate_gain('relu'))
            g_bidir.ndata['feat'][zero_idx] = rand_feat

            return self.feat_mlp(g_bidir.ndata['feat'])


class MulHopEncoder(nn.Module):
    def __init__(self, args):
        super(MulHopEncoder, self).__init__()
        self.args = args

        self.rel_feat = nn.Parameter(torch.Tensor(args.num_rel, args.rel_dim))
        nn.init.xavier_uniform_(self.rel_feat, gain=nn.init.calculate_gain('relu'))

        self.gnn = GNN(args, node_dim=args.ent_dim, edge_dim=args.rel_dim, nlayer=args.num_layers)

    def forward(self, train_g, ent_feat):
        ent_emb, rel_emb = self.gnn(train_g,
                                    rel_emb=self.rel_feat, ent_emb=ent_feat)
        return ent_emb, rel_emb