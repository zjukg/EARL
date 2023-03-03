#!/bin/sh

dataset='./data/FB15k-237' # 'wn18rr', 'YAGO3-10', or 'codex-l'
task_name='earl_rotate_fb15k237'
dim=150
gpu=0

python main.py --data_path ${dataset} --task_name ${task_name} --dim ${dim} --gpu cuda:${gpu}