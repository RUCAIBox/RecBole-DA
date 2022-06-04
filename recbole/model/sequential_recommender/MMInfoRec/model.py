# -*- coding: utf-8 -*-
# @Time    : 2021/8
# @Author  : Shuqing Bian

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from modules import LayerNorm


class MMInfoRecModel(nn.Module):
    def __init__(self, args):
        super(MMInfoRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length,
                                                2 * args.hidden_size)  # times 2 for possible concat feature
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        
        self.item_dropout = nn.Dropout(args.loss_fuse_dropout_prob)
        self.att_dropout = nn.Dropout(0.25)
        
        self.res_dropout = nn.Dropout(args.hidden_dropout_prob)
        
        # attention for fusing item embedding and attribute embedding
        att_encoder = nn.TransformerEncoderLayer(args.hidden_size, args.num_attention_heads, args.hidden_size,
                                                 args.attention_probs_dropout_prob)
        self.g_enc = nn.TransformerEncoder(att_encoder, args.num_hidden_layers)
        
        # attention for sequence encoder
        if args.enc == 'att':
            self.ar_att_hidden_sz = args.hidden_size
        else:
            self.ar_att_hidden_sz = 2 * args.hidden_size
        item_encoder = nn.TransformerEncoderLayer(self.ar_att_hidden_sz, args.num_attention_heads,
                                                  self.ar_att_hidden_sz,
                                                  args.attention_probs_dropout_prob)
        self.ar_att = nn.TransformerEncoder(item_encoder, args.num_hidden_layers)
        
        # autoregressive module for multi-step prediction
        self.ar = nn.GRU(self.ar_att_hidden_sz, self.ar_att_hidden_sz, args.num_hidden_layers_gru, batch_first=False)
        
        # all attributes for every item in the itemset
        # for evaluation
        self.all_att = [[0] * args.max_att_num]  # for the padding 0 in item id
        self.all_att_mask = [[0] * (args.max_att_num + 1)]
        for item in range(1, args.item_size):
            att = args.item2attribute[str(item)]
            self.all_att.append(att + [0] * (args.max_att_num - len(att)))
            
            # [0] + 1 for the id_embedding appended to the start of the attribute embeddings
            self.all_att_mask.append([0.] * (len(att) + 1) + [1.] * (args.max_att_num - len(att)))
        
        self.all_att = torch.tensor(self.all_att)
        self.all_att_mask = torch.tensor(self.all_att_mask)
        
        self.all_fused_embedding = None
        
        # memory
        self.mb = nn.Embedding(args.mem, self.ar_att_hidden_sz)
        self.mb_fc = nn.Linear(self.ar_att_hidden_sz, args.mem)
        self.softmax = nn.Softmax(dim=-1)
        self.mb_dp = nn.Dropout(args.mb_dp)
        
        self.ce = nn.CrossEntropyLoss()
        self.milnce = MILNCE(t=args.tau)
        # self.apply(self.init_weights)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
        # AAP
    
    def add_pos_inforec(self, sequence):
        seq_length = sequence.shape[0]
        hidden_sz = sequence.shape[-1]
        return self.dropout(
            sequence.transpose(0, 1) + self.position_embeddings.weight[:seq_length, :hidden_sz]).transpose(0, 1)
    
    def mem_read(self, dense):
        dense_out_logit = self.mb_fc(dense)
        dense_out_dist = self.softmax(dense_out_logit)
        return self.mb_dp(dense_out_dist.mm(self.mb.weight) + dense)
    
    def finetune(self, batch, flag):
        user_id, answer, input_ids, seq_len, attributes, att_att_mask = batch
        device = user_id.device
        
        item_embeddings = self.item_embeddings(input_ids)  # N_(all items in the batch) * hidden_sz
        att_embeddings = self.attribute_embeddings(
            attributes)  # N_(all items in the batch) * N_(max attribute number) * hidden_sz
        
        fused_embedding = self.fusing_embedding(item_embeddings, att_embeddings, att_att_mask)
        
        # split the non-sequentialized embedding and ids into individual sequence
        indi_seq = fused_embedding.split(seq_len.tolist())
        indi_ids = input_ids.split(seq_len.tolist())
        
        if flag == 'train':
            # for mask in attention
            batch_ids = [ids[:-1] for ids in indi_ids]
            
            seq_emb = [seq[:-1] for seq in indi_seq]
            
            target = torch.cat([indi[1:] for indi in indi_ids])
            # target = pad_sequence([indi[1:] for indi in indi_ids]).t()
        elif flag == 'test':
            # for mask in attention
            batch_ids = indi_ids
            
            seq_emb = indi_seq
            
            target = None
        else:
            raise Exception('Model phase needs to be specified.')
        
        padded_seq_emb = pad_sequence(seq_emb)  # max_seq_len(in batch) * batch_sz * hidden_sz
        
        sequence_emb = self.add_pos_inforec(padded_seq_emb)  # max_seq_len(in batch) * batch_sz * hidden_sz
        
        padded_batch_ids = pad_sequence(batch_ids)  # max_seq_len(in batch) * batch_sz
        
        # sequence_emb = self.add_pos_emb_new(padded_batch_ids.t())
        src_key_mask = padded_batch_ids == 0  # padding mask, same size as padded_batch_ids
        src_mask = self.generate_square_subsequent_mask(padded_batch_ids.shape[0]).to(
            device)  # single direction mask, max_seq_len(in batch) * max_seq_len(in batch)
        
        # batch_sz * max_seq_len(in batch) * hidden_sz
        sequence_output = self.ar_att(sequence_emb, mask=src_mask,
                                      src_key_padding_mask=src_key_mask.t()).transpose(0, 1).contiguous()
        if flag == 'train':
            # unpadded_out is aligned with input_ids {N_(all items in the batch) - batch_sz} * hidden_sz
            unpadded_out = [seq[:l - 1] for seq, l in zip(sequence_output, seq_len)]
            
            # no memory
            if self.args.mem == 0:
                pred = [torch.cat(unpadded_out).unsqueeze(
                    dim=0)]  # match the size of target, 1 * (N_(all items in the batch) - batch_sz) * hidden_sz
            
            # with memory
            else:
                cat_out = torch.cat(unpadded_out)
                
                mem_read = self.mem_read(cat_out)
                pred = [mem_read.unsqueeze(
                    dim=0)]  # match the size of target, 1 * (N_(all items in the batch) - batch_sz) * hidden_sz
            
            # multi step prediction, the first step is already calculated as 'pred'
            hidden = self.init_hidden(pred[-1].shape[1], device)
            for n in range(self.args.pred_step - 1):
                out, hidden = self.ar(pred[-1], hidden)
                # with memory
                if self.args.mem > 0:
                    out = self.mem_read(out.squeeze())
                    pred.append(out.unsqueeze(dim=0))
                else:
                    pred.append(out)
            
            # mask for calculating nce loss
            # -1 for negative, 1 for positive and 0 for out of range prediction
            target_mask = [-1 * torch.ones(input_ids.shape[0] - len(user_id), input_ids.shape[0]).to(device)
                           for _ in range(self.args.pred_step)]
            
            for s in range(self.args.pred_step):
                cur = 0  # current pointer
                for i in range(len(seq_len)):
                    s_l = seq_len[i] - 1
                    if s_l > s:
                        for j in range(s_l - s):
                            # single positive
                            # target_mask[s][cur + j, cur + j + i + s + 1] = 1
                            
                            #  multiple positive
                            pos_id = torch.nonzero(input_ids == input_ids[cur + j + i + s + 1])
                            target_mask[s][cur + j][pos_id] = 1
                        if s > 0:
                            target_mask[s][cur + s_l - s:cur + s_l] = 0
                    else:
                        target_mask[s][cur:cur + s_l] = 0
                    cur += s_l
            
            # multiple samples
            for i in range(self.args.mil - 1):
                fused_embedding = torch.cat((fused_embedding,
                                             self.fusing_embedding(item_embeddings, att_embeddings, att_att_mask)))
            repeated_mask = []
            for mask in target_mask:
                repeated_mask.append(mask.repeat(1, self.args.mil))
            
            target_mask = repeated_mask
            
            sim_score = torch.mm(torch.cat(pred, dim=1).squeeze(dim=0), self.item_dropout(fused_embedding).t())
            
            return self.milnce(sim_score, torch.cat(target_mask))
            
        elif flag == 'test':
            
            rec = []
            for j in range(len(seq_len)):
                rec.append(sequence_output[j, seq_len[j] - 1, :])
            
            recommend_output = torch.stack(rec)
            
            # with memory
            if self.args.mem > 0:
                recommend_output = self.mem_read(recommend_output)
            
            return recommend_output
    
    def fusing_embedding(self, item_embeddings, att_embeddings, att_att_mask):
        if self.args.enc == 'att':
            # ID self-attention
            # append the item embedding to the start of the attribute embeddings
            fused_embedding = self.g_enc(
                torch.cat([torch.unsqueeze(item_embeddings, dim=1), att_embeddings], dim=1).transpose(0, 1),
                src_key_padding_mask=att_att_mask.bool())[0]  # N_(all items in the batch) * hidden_sz
        elif self.args.enc == 'meancc':
            # ID concat mean
            # N_(all items in the batch) * (2 * hidden_sz)
            fused_embedding = torch.cat([item_embeddings,
                                         (att_embeddings * att_att_mask[:, 1:].unsqueeze(dim=-1)).sum(dim=1)
                                         / (att_att_mask[:, 1:].sum(dim=1) + 1e-24).unsqueeze(dim=-1)], dim=1)
        
        elif self.args.enc == 'attcc':
            # ID concat self-attention
            # N_(all items in the batch) * (2 * hidden_sz)
            fused_embedding = torch.cat([item_embeddings, self.g_enc(
                torch.cat([torch.unsqueeze(item_embeddings, dim=1), att_embeddings], dim=1).transpose(0, 1),
                src_key_padding_mask=att_att_mask.bool())[0]], dim=1)
        
        return fused_embedding
    
    def cal_test_emb(self):
        all_item_embeddings = self.item_embeddings.weight
        all_att_embeddings = self.attribute_embeddings(self.all_att.to(self.item_embeddings.weight.device))
        
        if self.args.enc == 'att':
            # ID self-attention
            # append the item embedding to the start of the attribute embeddings
            self.all_fused_embedding = self.g_enc(
                torch.cat([torch.unsqueeze(all_item_embeddings, dim=1), all_att_embeddings], dim=1).transpose(0, 1),
                src_key_padding_mask=self.all_att_mask.to(self.item_embeddings.weight.device).bool())[0]
        
        elif self.args.enc == 'meancc':
            # ID concat mean
            self.all_fused_embedding = torch.cat([all_item_embeddings, (
                    all_att_embeddings * self.all_att_mask.to(self.item_embeddings.weight.device)[:, 1:].unsqueeze(
                dim=-1)).sum(dim=1) / (self.all_att_mask.to(self.item_embeddings.weight.device)[:, 1:].sum(
                dim=1) + 1e-24).unsqueeze(dim=-1)], dim=1)
        
        elif self.args.enc == 'attcc':
            # ID concat self-attention
            # N_(all items in the batch) * (2 * hidden_sz)
            self.all_fused_embedding = torch.cat([all_item_embeddings, self.g_enc(
                torch.cat([torch.unsqueeze(all_item_embeddings, dim=1), all_att_embeddings], dim=1).transpose(0, 1),
                src_key_padding_mask=self.all_att_mask.to(self.item_embeddings.weight.device).bool())[0]], dim=1)
    
    def init_hidden(self, batch_size, device):
        return torch.zeros((self.args.num_hidden_layers_gru, batch_size, self.ar_att_hidden_sz), requires_grad=True).to(
            device)
    
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MILNCE(nn.Module):
    """
    multiple positive instance NCE loss
    """
    
    def __init__(self, t=1.0):
        super(MILNCE, self).__init__()
        self.t = t
    
    def forward(self, sim_score, target_mask):
        """
        refer to the MIL-NCE loss function of Eq. (1) in https://arxiv.org/pdf/1912.06430.pdf
        :param sim_score: (N_(all items in the batch) - batch_sz) * N_(all items in the batch)
        :param target_mask: (N_(all items in the batch) - batch_sz) * N_(all items in the batch)
        :param t: temperature
        :return:
        """
        nominator = torch.log((torch.exp(self.t * sim_score) * ((target_mask == 1) + 1e-24)).sum(dim=1))
        denominator = torch.logsumexp(self.t * sim_score, dim=1)
        loss = -(nominator - denominator) * (target_mask[:, 0] != 0)
        return loss.mean()
