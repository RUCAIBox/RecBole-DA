# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Shuqing Bian

import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_at_k, mrr_at_k, get_metric


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

    def valid(self, epoch, full_sort=True):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=True):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
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

    def get_full_sort_scores(self, epoch, answers, pred_list):
        recall, ndcg, mrr = [], [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_at_k(answers, pred_list, k))
            mrr.append(mrr_at_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@10": '{:.4f}'.format(recall[0]), "NDCG@10": '{:.4f}'.format(ndcg[0]), "MRR@10": '{:.4f}'.format(mrr[0]),
            "HIT@20": '{:.4f}'.format(recall[1]), "NDCG@20": '{:.4f}'.format(ndcg[1]), "MRR@20": '{:.4f}'.format(mrr[1]),
            "HIT@50": '{:.4f}'.format(recall[2]), "NDCG@50": '{:.4f}'.format(ndcg[2]), "MRR@50": '{:.4f}'.format(mrr[2])
        }

        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return recall + ndcg + mrr, str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        print(f'load ckp from {file_name}')

    def load_part(self, file_name):
        state_dicts = torch.load(file_name)
        self.model.item_embeddings.load_state_dict(state_dicts['item_embedding'])
        self.model.position_embeddings.load_state_dict(state_dicts['position_embedding'])
        self.model.item_encoder.load_state_dict(state_dicts['item_encoder'])
        print(f'load ckp from {file_name}')

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

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


class CPTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        super(CPTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight[:-1]
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_scores(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_at_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2])
        }

        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return recall + ndcg, str(post_fix)

    def pretrain(self, epoch, pretrain_dataloader):

        desc = 'CP4Rec'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")
        self.model.train()
        cp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            item_seq = batch.to(self.device)
            cp_loss = self.model.pretrain(item_seq)
            self.optim.zero_grad()
            cp_loss.backward()
            self.optim.step()
            cp_loss_avg += cp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "cp_loss_avg": '{:.4f}'.format(cp_loss_avg /num),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

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
                _, input_ids, target_pos, target_neg, _ = batch
                # 采用 负采样 cross_entropy
                sequence_output = self.model(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()  # 修改loss
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

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()


            if full_sort:
                # 推荐任务的测试
                pred_list = None
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_scores(epoch, answer_list, pred_list)
            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)