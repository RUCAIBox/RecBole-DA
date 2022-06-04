# -*- coding: utf-8 -*-
# @Time    : 2021/8
# @Author  : Shuqing Bian
import os
import numpy as np
import random
import torch
import argparse
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataset import MMInfoRecDataset, collate_fn
from trainer import MMInfoRecTrainer
from model import MMInfoRecModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', default='Toys_and_Games', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_full', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--num_hidden_layers_gru', default=1, type=int)
    parser.add_argument('--enc', default='attcc', type=str, help='att, meancc, attcc')
    parser.add_argument("--loss_fuse_dropout_prob", type=float, default=0.5, help="dropout for fusing")
    parser.add_argument("--mil", type=int, default=4, help="number of milnce samples")
    parser.add_argument("--mb_dp", type=float, default=0.2, help="dropout for memory bank")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=21, type=int)

    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--pred_step", type=int, default=1, help="prediction step")
    parser.add_argument("--tau", type=float, default=0.05, help="prediction step")
    parser.add_argument("--dc_s", type=int, default=1, help="number of steps to decrease of lr")
    parser.add_argument("--dc", type=float, default=1, help="decrease rate of lr")

    parser.add_argument("--mem", type=int, default=64, help="number of memory units")

    args = parser.parse_args()

    # set_seed(args.seed)
    # check_path(args.output_dir)
    
    cur_dir = os.getcwd()
    
    args.data_dir = cur_dir + '/data/'
    args.output_dir = cur_dir + '/output/'

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'mem{args.mem}-{args.enc}-lr{args.lr}-b_sz{args.batch_size}-l2{args.weight_decay}-dpA{args.attention_probs_dropout_prob}' \
               f'-dpH{args.hidden_dropout_prob}-dpF{args.loss_fuse_dropout_prob}-layer{args.num_hidden_layers}-head{args.num_attention_heads}' \
               f'-pred{args.pred_step}-tau{args.tau}-dc{args.dc_s}-{args.dc}-mil{args.mil}-'
    args.log_dir = os.path.join(args.output_dir, args.data_name + '/', args_str + time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    print('logging to ' + args.log_dir)
    with open(args.log_dir + '/log.txt', 'a') as f:
        f.write(str(args) + '\n')

    print(str(args))

    args.item2attribute = item2attribute
    args.max_att_num = max([len(args.item2attribute[att]) for att in args.item2attribute])
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    args.checkpoint_path = args.log_dir + '/' + args_str + '.pt'

    train_dataset = MMInfoRecDataset(args, user_seq, args.max_att_num, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    eval_dataset = MMInfoRecDataset(args, user_seq, args.max_att_num, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    test_dataset = MMInfoRecDataset(args, user_seq, args.max_att_num, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = MMInfoRecModel(args=args)

    trainer = MMInfoRecTrainer(model, train_dataloader, eval_dataloader,
                               test_dataloader, args)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optim, step_size=args.dc_s, gamma=args.dc)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        pretrained_path = args.checkpoint_path
        try:
            trainer.load(pretrained_path)
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found! MMInfoRec is trained from scratch')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            # early_stopping(np.array(scores[-1:]), trainer.model)
            early_stopping(np.array(scores[0:3]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.step()

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_dir + '/log.txt', 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()
