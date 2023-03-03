import dgl
import os
import logging
import torch
import numpy as np
import random
import pickle


# def get_g_dir(triples):
#     triples = torch.LongTensor(triples)
#     num_tri = triples.shape[0]
#     g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
#     g.edata['rel'] = triples[:, 1].T
#     g.edata['inv'] = torch.zeros(num_tri)
#
#     return g

def get_g(triples):
    triples = np.array(triples)
    g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
    g.edata['rel'] = torch.tensor(triples[:, 1].T)
    return g


def get_g_bidir(triples):
    triples = np.array(triples)
    g = dgl.graph((np.concatenate([triples[:, 0].T, triples[:, 2].T]),
                   np.concatenate([triples[:, 2].T, triples[:, 0].T])))
    g.edata['rel'] = torch.tensor(np.concatenate([triples[:, 1].T, triples[:, 1].T]))
    g.edata['inv'] = torch.cat([torch.zeros(triples.shape[0]), torch.ones(triples.shape[0])])

    return g


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    data_tuple = pickle.loads(data)
    return data_tuple


def occupy_mem(args):
    def check_mem(args):
        devices_info = os.popen(
            '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
            "\n")
        total, used = devices_info[int(args.gpu.split(':')[1])].split(',')
        return total, used

    total, used = check_mem(args)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.8)
    block_mem = max_mem - used
    x = torch.FloatTensor(256, 1024, block_mem).to(args.gpu)
    del x


def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    # dgl.seed(seed)
    # dgl.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


class Log(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        # file handler
        log_file = os.path.join(log_dir, name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # console handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        fh.close()
        sh.close()

    def get_logger(self):
        return self.logger


class FileLog(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        # file handler
        log_file = os.path.join(log_dir, name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

        fh.close()

    def get_logger(self):
        return self.logger
