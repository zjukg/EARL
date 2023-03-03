import os
import torch
import pickle as pkl
from dataset import OneShotIterator, KGETrainDataset
from torch.utils.data import DataLoader
from torch import optim
from trainer import Trainer
import torch.nn as nn
from model import MLP, ConRelEncoder, MulHopEncoder


class EARLTrainer(Trainer):
    def __init__(self, args):
        super(EARLTrainer, self).__init__(args)

        self.num_step = args.num_step
        self.train_bs = args.train_bs
        self.lr = args.lr
        self.log_per_step = args.log_per_step
        self.check_per_step = args.check_per_step
        self.early_stop_patience = args.early_stop_patience

        self.train_iter = OneShotIterator(DataLoader(self.train_dataset,
                                                      batch_size=self.train_bs,
                                                      shuffle=True,
                                                      num_workers=max(1, args.cpu_num // 2),
                                                      collate_fn=KGETrainDataset.collate_fn))

        res_ent = pkl.load(open(os.path.join(args.data_path, f'res_ent_{self.args.res_ent_ratio}.pkl'), 'rb'))

        self.res_ent_map = res_ent['res_ent_map'].to(self.args.gpu)
        num_res_ent = self.res_ent_map.shape[0]
        self.res_ent_emb = nn.Parameter(torch.Tensor(num_res_ent, args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.res_ent_emb, gain=nn.init.calculate_gain('relu'))

        self.topk_idx = res_ent['topk_idx'].to(self.args.gpu)
        self.topk_idx = self.topk_idx[:, :args.top_k]

        self.ent_sim = res_ent['topk_sim'].to(self.args.gpu)
        self.ent_sim = self.ent_sim[:, :args.top_k]
        self.ent_sim = torch.softmax(self.ent_sim/0.2, dim=-1)

        self.con_rel_encoder = ConRelEncoder(args).to(args.gpu)
        self.mul_hop_encoder = MulHopEncoder(args).to(args.gpu)

        self.proj = MLP(args.ent_dim*2, args.ent_dim, args.ent_dim).to(args.gpu)


        # optimizer
        self.optimizer = optim.Adam(
                                    list(self.mul_hop_encoder.parameters()) +
                                    list(self.con_rel_encoder.parameters()) +
                                    list(self.proj.parameters()) +
                                    [self.res_ent_emb],
                                    lr=self.lr)

        self.cal_num_param()

    def cal_num_param(self):
        num_param = 0
        print('parameters:')
        for name, param in self.mul_hop_encoder.named_parameters():
            self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
            num_param += param.numel()

        for name, param in self.con_rel_encoder.named_parameters():
            self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
            num_param += param.numel()

        for name, param in self.proj.named_parameters():
            self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
            num_param += param.numel()

        name = 'res_ent_emb'
        param = self.res_ent_emb
        self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
        num_param += param.numel()

        name = 'res_ent_map'
        param = self.res_ent_map
        self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
        num_param += param.numel()

        self.logger.info(f'\ttotal: {num_param / 1e6} M')

        return num_param

    def kn_res_ent_encode(self):
        topk_res_ent = torch.index_select(self.res_ent_emb, 0, self.topk_idx.reshape(-1)).reshape(
            self.topk_idx.shape[0], self.topk_idx.shape[1], self.args.ent_dim)
        topk_res_ent = self.ent_sim.unsqueeze(2) * topk_res_ent
        kn_res_ent_emb = torch.sum(topk_res_ent, dim=1)

        return kn_res_ent_emb

    def con_rel_encode(self):
        con_rel_info = self.con_rel_encoder(self.train_g_bidir)
        return con_rel_info

    def get_emb(self):
        con_rel_info = self.con_rel_encode()
        kn_res_ent_info = self.kn_res_ent_encode()
        cat_ent_emb = self.proj(torch.cat([kn_res_ent_info, con_rel_info], dim=-1))

        cat_ent_emb[self.res_ent_map] = self.res_ent_emb

        ent_emb, rel_emb = self.mul_hop_encoder(self.train_g_bidir, cat_ent_emb)

        return ent_emb, rel_emb

    def train_one_step(self):
        # batch data
        batch = next(self.train_iter)
        pos_triple, neg_tail_ent, neg_head_ent = [b.to(self.args.gpu) for b in batch]

        # get ent and rel emb
        ent_emb, rel_emb = self.get_emb()

        # cal loss
        kge_loss = self.get_loss(pos_triple, neg_tail_ent, neg_head_ent, ent_emb, rel_emb)
        loss = kge_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, istest=False, num_cand='all'):
        if istest:
            dataloader = self.test_dataloader
        else:
            dataloader = self.valid_dataloader

        with torch.no_grad():
            # get ent and rel emb
            ent_emb, rel_emb = self.get_emb()

        results, count = self.get_rank(dataloader, ent_emb, rel_emb, num_cand)

        for k, v in results.items():
            results[k] = v / count

        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results









































