import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import dgl


class Data(object):
    def __init__(self, args):
        self.data_path = args.data_path

        self.entity2id, self.relation2id = self.get_ent_rel_map()

        args.num_rel = len(self.relation2id)
        args.num_ent = len(self.entity2id)

        # sub_ent_map_np = args.sub_ent_map.numpy()
        # ent2submap = torch.zeros(args.num_ent, dtype=torch.int64) - 1
        # for i in range(args.num_ent):
        #     if i in args.sub_ent_map:
        #         ent2submap[i] = np.where(sub_ent_map_np == i)[0].item()
        # args.ent2submap = ent2submap

        self.train_triples, self.valid_triples, self.test_triples = self.read_triple(self.entity2id, self.relation2id)
        self.hr2t_train, self.rt2h_train, self.hr2t_all, self.rt2h_all = self.get_hr2t_rt2h(self.train_triples, self.valid_triples, self.test_triples)

    def get_ent_rel_map(self):
        with open(os.path.join(self.data_path, 'entities.dict')) as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(self.data_path, 'relations.dict')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        return entity2id, relation2id

    def read_triple(self, entity2id, relation2id):
        train_triples = []
        with open(os.path.join(self.data_path, 'train.txt')) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                train_triples.append((entity2id[h], relation2id[r], entity2id[t]))

        valid_triples = []
        with open(os.path.join(self.data_path, 'valid.txt')) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                valid_triples.append((entity2id[h], relation2id[r], entity2id[t]))

        test_triples = []
        with open(os.path.join(self.data_path, 'test.txt')) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                test_triples.append((entity2id[h], relation2id[r], entity2id[t]))

        return train_triples, valid_triples, test_triples

    def get_hr2t_rt2h(self, train_triples, valid_triples, test_triples):
        hr2t_train = ddict(list)
        rt2h_train = ddict(list)

        hr2t_all = ddict(list)
        rt2h_all = ddict(list)

        for tri in train_triples:
            h, r, t = tri
            hr2t_train[(h, r)].append(t)
            rt2h_train[(r, t)].append(h)
            hr2t_all[(h, r)].append(t)
            rt2h_all[(r, t)].append(h)

        for tri in valid_triples:
            h, r, t = tri
            hr2t_all[(h, r)].append(t)
            rt2h_all[(r, t)].append(h)

        for tri in test_triples:
            h, r, t = tri
            hr2t_all[(h, r)].append(t)
            rt2h_all[(r, t)].append(h)

        return hr2t_train, rt2h_train, hr2t_all, rt2h_all


def get_dataset_and_g(args):
    data = Data(args)
    train_dataset = KGETrainDataset(args, data.train_triples, args.num_ent, args.num_neg, data.hr2t_train, data.rt2h_train)
    valid_dataset = KGEEvalDataset(args, data.valid_triples, args.num_ent, data.hr2t_all, data.rt2h_all)
    test_dataset = KGEEvalDataset(args, data.test_triples, args.num_ent, data.hr2t_all, data.rt2h_all)

    train_g_bidir = get_train_g_bidir(data.train_triples, args.num_ent)
    train_g_sidir = get_train_g_sidir(data.train_triples, args.num_ent)

    return train_dataset, valid_dataset, test_dataset, train_g_sidir, train_g_bidir


def get_train_g_bidir(train_triples, num_ent):
    triples = torch.LongTensor(train_triples)
    num_tri = triples.shape[0]
    g = dgl.graph((torch.cat([triples[:, 0].T, triples[:, 2].T]),
                   torch.cat([triples[:, 2].T, triples[:, 0].T])), num_nodes=num_ent)
    g.edata['rel'] = torch.cat([triples[:, 1].T, triples[:, 1].T])
    g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])

    return g


def get_train_g_sidir(train_triples, num_ent):
    triples = torch.LongTensor(train_triples)
    num_tri = triples.shape[0]
    g = dgl.graph((triples[:, 0].T, triples[:, 2].T), num_nodes=num_ent)
    g.edata['rel'] = triples[:, 1].T
    g.edata['inv'] = torch.zeros(num_tri)

    return g


class KGETrainDataset(Dataset):
    def __init__(self, args, train_triples, num_ent, num_neg, hr2t, rt2h):
        self.args = args
        self.triples = train_triples
        self.num_ent = num_ent
        self.num_neg = num_neg
        self.hr2t = hr2t
        self.rt2h = rt2h

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t = pos_triple

        neg_tail_ent = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r)]),
                                        self.num_neg)

        neg_head_ent = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t)]),
                                        self.num_neg)

        pos_triple = torch.LongTensor(pos_triple)
        neg_tail_ent = torch.from_numpy(neg_tail_ent)
        neg_head_ent = torch.from_numpy(neg_head_ent)

        return pos_triple, neg_tail_ent, neg_head_ent

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        neg_tail_ent = torch.stack([_[1] for _ in data], dim=0)
        neg_head_ent = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, neg_tail_ent, neg_head_ent


class KGEEvalDataset(Dataset):
    def __init__(self, args, eval_triples, num_ent, hr2t, rt2h):
        self.args = args
        self.triples = eval_triples
        self.num_ent = num_ent
        self.hr2t = hr2t
        self.rt2h = rt2h
        self.num_cand = 'all'

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t = pos_triple
        if self.num_cand == 'all':
            tail_label, head_label = self.get_label(self.hr2t[(h, r)], self.rt2h[(r, t)])
            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_label, head_label
        else:
            neg_tail_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r)]),
                                             self.num_cand)

            try:
                neg_head_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t)]),
                                                 self.num_cand)
            except:
                print(pos_triple)
            tail_cand = torch.from_numpy(np.concatenate(([t], neg_tail_cand)))
            head_cand = torch.from_numpy(np.concatenate(([h], neg_head_cand)))

            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_cand, head_cand

    def get_label(self, true_tail, true_head):
        y_tail = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_tail:
            y_tail[e] = 1.0
        y_head = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_head:
            y_head[e] = 1.0

        return torch.FloatTensor(y_tail), torch.FloatTensor(y_head)

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_label_or_cand = torch.stack([_[1] for _ in data], dim=0)
        head_label_or_cand = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, tail_label_or_cand, head_label_or_cand


class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data