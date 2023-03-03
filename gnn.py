import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F
import dgl


class GNNLayer(nn.Module):
    def __init__(self, args, node_dim, edge_dim, act=None, feat_drop=0, self_update=True):
        super(GNNLayer, self).__init__()
        self.args = args
        self.act = act

        self.edge_dim = edge_dim
        self.node_dim = node_dim

        # define in/out/loop transform layer
        self.W_O_r = nn.Linear(edge_dim, node_dim)
        self.W_O_e = nn.Linear(node_dim, node_dim)
        self.W_I_r = nn.Linear(edge_dim, node_dim)
        self.W_I_e = nn.Linear(node_dim, node_dim)
        self.W_S = nn.Linear(node_dim, node_dim)

        # define relation transform layer
        self.W_R = nn.Linear(edge_dim, edge_dim)

        self.feat_drop = nn.Dropout(feat_drop)

        self.self_update = self_update

    def msg_func(self, edges):
        comp_h = torch.cat((edges.data['h'], edges.src['h']), dim=-1)

        non_inv_idx = (edges.data['inv'] == 0)
        inv_idx = (edges.data['inv'] == 1)

        msg = torch.zeros_like(edges.src['h'])
        msg[non_inv_idx] = self.W_I(comp_h[non_inv_idx])
        msg[inv_idx] = self.W_O(comp_h[inv_idx])

        return {'msg': msg}

    def apply_node_func(self, nodes):
        comp_h_s = nodes.data['h']

        if self.self_update:
            h_new = self.W_S(comp_h_s) + nodes.data['h_agg']
        else:
            h_new = nodes.data['h_agg']

        h_new = self.feat_drop(h_new)

        if self.act is not None:
            h_new = self.act(h_new)

        return {'h': h_new}

    def edge_update(self, rel_emb):
        h_edge_new = self.W_R(rel_emb)

        if self.act is not None:
            h_edge_new = self.act(h_edge_new)

        # # Compute relation output
        # h_edge_new = self.W_R(rel_emb)

        return h_edge_new

    def forward(self, g, non_inv_g, inv_g, ent_emb, rel_emb):
        with g.local_scope():
            g.ndata['h'] = ent_emb
            g.edata['msg'] = torch.zeros(g.num_edges(), self.args.ent_dim).to(self.args.gpu)

            with non_inv_g.local_scope():
                non_inv_g.edata['h'] = rel_emb[non_inv_g.edata['rel']]
                non_inv_g.ndata['h'] = ent_emb[non_inv_g.ndata[dgl.NID]]

                non_inv_msg_node_h = self.W_I_e(non_inv_g.srcdata['h'])
                non_inv_msg_edge_h = self.W_I_r(non_inv_g.edata['h'])
                non_inv_g.srcdata.update({'msg_node_h': non_inv_msg_node_h})
                non_inv_g.edata.update({'msg_edge_h': non_inv_msg_edge_h})
                non_inv_g.apply_edges(fn.u_add_e('msg_node_h', 'msg_edge_h', 'h_agg'))
                g.edata['msg'][non_inv_g.edata[dgl.EID]] = non_inv_g.edata['h_agg']

            # torch.cuda.empty_cache()

            with inv_g.local_scope():
                inv_g.edata['h'] = rel_emb[inv_g.edata['rel']]
                inv_g.ndata['h'] = ent_emb[inv_g.ndata[dgl.NID]]

                inv_msg_node_h = self.W_O_e(inv_g.srcdata['h'])
                inv_msg_edge_h = self.W_O_r(inv_g.edata['h'])
                inv_g.srcdata.update({'msg_node_h': inv_msg_node_h})
                inv_g.edata.update({'msg_edge_h': inv_msg_edge_h})
                inv_g.apply_edges(fn.u_add_e('msg_node_h', 'msg_edge_h', 'h_agg'))
                g.edata['msg'][inv_g.edata[dgl.EID]] = inv_g.edata['h_agg']

            # torch.cuda.empty_cache()

            g.update_all(fn.copy_e('msg', 'msg'), fn.mean('msg', 'h_agg'), self.apply_node_func)

            rel_emb = self.edge_update(rel_emb)
            ent_emb = g.ndata['h']

        return ent_emb, rel_emb


class GNN(nn.Module):
    def __init__(self, args, node_dim, edge_dim, nlayer=2, self_update=True):
        super(GNN, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        for idx in range(nlayer):
            if idx == nlayer - 1:
                self.layers.append(GNNLayer(args, node_dim, edge_dim, act=None, feat_drop=0, self_update=True))
            else:
                self.layers.append(GNNLayer(args, node_dim, edge_dim, act=F.relu, feat_drop=0, self_update=True))

    def forward(self, g, **param):
        rel_emb = param['rel_emb']
        ent_emb = param['ent_emb']
        with g.local_scope():
            g_cpu = g.cpu()
            non_inv_idx = torch.nonzero(g.edata['inv'] == 0).flatten().cpu()
            non_inv_g = dgl.edge_subgraph(g_cpu, non_inv_idx).to(self.args.gpu)
            inv_idx = torch.nonzero(g.edata['inv'] == 1).flatten().cpu()
            inv_g = dgl.edge_subgraph(g_cpu, inv_idx).to(self.args.gpu)

            for layer in self.layers:
                ent_emb, rel_emb = layer(g, non_inv_g, inv_g, ent_emb, rel_emb)
                # torch.cuda.empty_cache()

        return ent_emb, rel_emb
