# -*- coding: utf-8 -*-
# @Time    : 2021/8
# @Author  : Shuqing Bian

import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        
        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        
        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()
    
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)
    
    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)
    
    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)
    
    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError
    
    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        # with open(self.args.log_file, 'a') as f:
        with open(self.args.log_dir + '/log.txt', 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)
    
    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
    
    def bpr(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        
        return loss
    
    def cross_entropy(self, seq_out, pos_ids, seq_len):
        indi_out = torch.cat([indi_seq[:indi_len] for indi_seq, indi_len in zip(seq_out, seq_len)])
        target = torch.cat([indi_tar[:indi_len] for indi_tar, indi_len in zip(pos_ids, seq_len)])
        pred = torch.mm(indi_out, self.model.item_embeddings.weight[1:].t())
        return self.model.ce(pred, target - 1)
    
    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits
    
    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def predict_full_att(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.all_fused_embedding
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class MMInfoRecTrainer(Trainer):
    
    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(MMInfoRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
    
    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        
        str_code = "train" if train else "test"
        
        # Setting the tqdm progress bar
        
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                
                # milnce
                loss = self.model.finetune(batch, 'train')
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
            
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }
            
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            
            # with open(self.args.log_file, 'a') as f:
            with open(self.args.log_dir + '/log.txt', 'a') as f:
                f.write(str(post_fix) + '\n')
        
        else:
            self.model.eval()
            
            pred_list = None
            
            answer_list = None
            with torch.no_grad():
                self.model.cal_test_emb()
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, answer, _, seq_len, _, _ = batch
                    recommend_output = self.model.finetune(batch, flag='test')
                    
                    rating_pred = self.predict_full_att(recommend_output)
                    
                    answers = answer.unsqueeze(dim=1)
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[torch.tensor(self.args.train_matrix[batch_user_index].toarray() > 0)] = 0
                    
                    _, batch_pred_list = rating_pred.topk(20)
                    
                    if i == 0:
                        pred_list = batch_pred_list.cpu().numpy()
                        answer_list = answers.cpu().numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list.cpu().numpy(), axis=0)
                        answer_list = np.append(answer_list, answers.cpu().numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)
            