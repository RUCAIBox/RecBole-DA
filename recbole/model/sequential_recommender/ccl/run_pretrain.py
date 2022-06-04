# -*- coding: utf-8 -*-
# @Time    : 2020/11/18 11:09
# @Author  : Shuqing Bian

import numpy as np
import random
import torch
import os
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import CPDataset
from trainers import CPTrainer
from models import CPModel
from utils import get_user_seqs, check_path, set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../SIGIR2021/data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)

    # model args
    parser.add_argument("--model_name", default='CP4Rec', type=str)
    parser.add_argument('--aug_type_1', default='mask', type=str)
    parser.add_argument('--aug_type_2', default='reorder', type=str)

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=200, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=256)

    parser.add_argument("--mask_ratio", type=float, default=0.5, help="mask ratio")
    parser.add_argument("--crop_ratio", type=float, default=0.5, help="crop ratio")
    parser.add_argument("--reorder_ratio", type=float, default=0.5, help="reorder ratio")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'

    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item = get_user_seqs(args.data_file, pretrain=True)

    args.item_size = max_item + 2
    args.mask_item_token = max_item + 1
    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.aug_type_1}-{args.aug_type_2}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    model = CPModel(args=args)
    trainer = CPTrainer(model, None, None, None, args)

    for epoch in range(args.pre_epochs):
        pretrain_dataset = CPDataset(args, user_seq)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)
        trainer.pretrain(epoch, pretrain_dataloader)
        if (epoch+1) % 10 == 0:
            ckp = f'CP4Rec-{args.data_name}-{args.aug_type_1}-{args.aug_type_2}-epochs-{epoch+1}.pt'
            checkpoint_path = os.path.join(args.output_dir, ckp)
            trainer.save(checkpoint_path)
main()