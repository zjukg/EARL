import argparse
from utils import init_dir
from earl_trainer import EARLTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # task level setting
    parser.add_argument('--data_path', default='./data/FB15k-237')

    parser.add_argument('--task_name', default='rotate_fb15k237_dim150_finalopentest')

    # file setting
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', default='./tb_log', type=str)

    # training setting
    parser.add_argument('--num_step', default=100000, type=int)
    parser.add_argument('--train_bs', default=1024, type=int)
    parser.add_argument('--eval_bs', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--log_per_step', default=10, type=int)
    parser.add_argument('--check_per_step', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--res_ent_ratio', default='0p1', type=str)

    # model setting
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--dim', default=150, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rel', default=None)
    parser.add_argument('--num_ent', default=None)

    # device
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--cpu_num', default=10, type=int)

    args = parser.parse_args()
    init_dir(args)

    # dim for RotatE
    args.ent_dim = args.dim * 2
    args.rel_dim = args.dim

    trainer = EARLTrainer(args)

    trainer.train()

