from torch.utils.tensorboard import SummaryWriter
from utils import Log
import json
import os
import csv
import torch
from dataset import KGEEvalDataset, get_dataset_and_g
from torch.utils.data import DataLoader
from kge_model import KGEModel
import torch.nn.functional as F
from collections import defaultdict as ddict


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # writer and logger
        self.name = args.task_name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps({k: v for k, v in vars(args).items() if k not in ['sub_ent_map']}))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # dataloader and g
        train_dataset, valid_dataset, test_dataset, train_g_sidir, train_g_bidir = get_dataset_and_g(args)

        self.train_g_sidir = train_g_sidir.to(args.gpu)
        self.train_g_bidir = train_g_bidir.to(args.gpu)

        self.train_dataset = train_dataset

        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.eval_bs,
                                           shuffle=False,
                                           num_workers=max(1, args.cpu_num // 2),
                                           collate_fn=KGEEvalDataset.collate_fn)

        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.eval_bs,
                                          shuffle=False,
                                          num_workers=max(1, args.cpu_num // 2),
                                          collate_fn=KGEEvalDataset.collate_fn)

        # model
        self.kge_model = KGEModel(args).to(args.gpu)

        # parameters
        self.num_step = None
        self.train_bs = None
        self.lr = None
        self.log_per_step = None
        self.check_per_step = None
        self.early_stop_patience = None

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_named_loss(self, loss, step, name):
        self.writer.add_scalar(f"training/{name}", loss, step)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def log_evaluation_result(self, results, text):
        self.logger.info('{} | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            text,
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

    def write_rst_csv(self, suffix_dict, query_part):
        for suf, rst in suffix_dict.items():
            with open(os.path.join(self.args.log_dir, f"{self.args.task_name}_{suf}_{query_part}.csv"), "a") as rstfile:
                rst_writer = csv.writer(rstfile)
                rst_writer.writerow([self.name, round(rst["mrr"], 4), round(rst["hits@1"], 4),
                                     round(rst["hits@5"], 4), round(rst["hits@10"], 4)])

    def save_checkpoint(self, e, state):
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def get_curr_state(self):
        state = {
                 'con_rel_encoder': self.con_rel_encoder.state_dict(),
                 'mul_hop_encoder': self.mul_hop_encoder.state_dict(),
                 'proj': self.proj.state_dict(),
                 'res_ent_map': self.res_ent_map,
                 'res_ent_emb': self.res_ent_emb
        }
        return state

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.mul_hop_encoder.load_state_dict(state['mul_hop_encoder'])
        self.con_rel_encoder.load_state_dict(state['con_rel_encoder'])
        self.proj.load_state_dict(state['proj'])
        self.res_ent_map = state['res_ent_map']
        self.res_ent_emb = state['res_ent_emb']

    def get_loss(self, tri, neg_tail_ent, neg_head_ent, ent_emb, rel_emb):
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, rel_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, rel_emb, mode='head-batch')
        neg_score = torch.cat([neg_tail_score, neg_head_score])
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_score = self.kge_model(tri, ent_emb, rel_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss

    def train_one_step(self):
        raise NotImplementedError

    def train(self):
        best_step = 0
        best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@3': 0, 'hits@10': 0}
        bad_count = 0
        self.logger.info('start training')

        for i in range(1, self.num_step + 1):
            loss = self.train_one_step()

            self.write_training_loss(loss.item(), i)
            if i % self.log_per_step == 0:
                self.logger.info('step: {} | loss: {:.4f}'.format(i, loss.item()))

            if i % self.check_per_step == 0 or i == 1:
                eval_rst = self.evaluate()
                self.write_evaluation_result(eval_rst, i)

                if eval_rst['mrr'] > best_eval_rst['mrr']:
                    best_eval_rst = eval_rst
                    best_step = i
                    self.logger.info('best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                    self.save_checkpoint(i, self.get_curr_state())
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info('best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_step, best_eval_rst['mrr'], bad_count))

            if bad_count >= self.early_stop_patience:
                self.logger.info('early stop at step {}'.format(i))
                break

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            best_eval_rst['mrr'], best_eval_rst['hits@1'],
            best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        self.before_test_load()

        self.evaluate(istest=True)

    def get_emb(self):
        raise NotImplementedError

    def get_rank(self, eval_dataloader, ent_emb, rel_emb, num_cand='all'):
        results = ddict(float)
        count = 0

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, mode='tail-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, mode='head-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
        else:
            for i in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]

                    b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64) + num_cand
                    # tail prediction
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, rel_emb, mode='tail-batch')
                    tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    # head prediction
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, rel_emb, mode='head-batch')
                    head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]

                    ranks = torch.cat([tail_ranks, head_ranks])
                    ranks = ranks.float()
                    count += torch.numel(ranks)
                    results['mr'] += torch.sum(ranks).item()
                    results['mrr'] += torch.sum(1.0 / ranks).item()

                    for k in [1, 5, 10]:
                        results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        return results, count








































