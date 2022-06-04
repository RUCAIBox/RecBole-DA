# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 10:57
# @Author  : Shuqing Bian

import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, VanillaAttention, Attention


class CPModel(nn.Module):
    def __init__(self, args):
        super(CPModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.cp_norm = nn.Linear(args.hidden_size, args.hidden_size)

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def pretrain_bpr(self, item_seq):
        pass
        # 0  1  0  0
        # 0  0  1  0
        # 0  0  0  1
        # 0  0  0  0
        # diag = torch.diag(torch.ones(sim.size(0)-1), 1)

        # 0   0   -1  -1
        # -1  0   0   -1
        # -1  -1  0   0
        # -1  -1  -1  0
        # neg = torch.tensor(-1).expand_as(diag)
        #
        # ones_1 = torch.diag(torch.ones(sim.size(0)))
        # ones_2 = torch.diag(torch.ones(sim.size(0)-1), 1)

        # 0   1  -1 -1
        # -1  0  1  -1
        # -1 -1  0  1
        # -1 -1 -1  0
        # sign = diag + (neg + ones_1 + ones_2)

        # 0   1  -1 -1
        # -1 -1  0  1

    def pretrain(self, item_seqs):
        # [B 2 Len]
        item_seq = item_seqs.view(-1, self.args.max_seq_length)  # [2B Len]
        item_output = self.forward(item_seq)

        user_emb = self.cp_norm(item_output[:, -1, :])  # [2B H]

        scores = torch.matmul(user_emb, user_emb.t())  # [2B 2B]
        user_emb_l2 = torch.norm(user_emb, dim=1).unsqueeze(-1)  # [2B]
        user_emb_l2 = torch.matmul(user_emb_l2, user_emb_l2.t())  # [2B 2B]
        sim = torch.exp(scores / user_emb_l2)  # [2B 2B]

        # 0  1  0  0
        # 0  0  1  0
        # 0  0  0  1
        # 0  0  0  0
        pos_index = torch.diag(torch.ones(sim.size(0) - 1, device=sim.device), 1)

        # 0  1  1  1
        # 1  0  1  1
        # 1  1  0  1
        # 1  1  1  0
        # 为了快速求每一行的和 对角元素是 '自身与自身' 忽略
        ones = torch.tensor(1.0, device=sim.device).expand_as(sim)
        ones_diag = torch.diag(torch.ones(sim.size(0), device=sim.device))
        sign = ones - ones_diag
        # 一个用户序列只需计算一次loss 所以固定间隔取
        sum_sim = torch.sum(sim[::2] * sign[::2], dim=1)  # [B]
        pos_sim = torch.sum(sim[::2] * pos_index[::2], dim=1)  # [B]

        cp_loss = torch.sum(-torch.log(pos_sim/sum_sim))

        return cp_loss

    # fine-tune same as SASRec
    def forward(self, item_seq):
        # position_embedding
        seq_length = item_seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embeddings(position_ids)  # [B L H]

        # item sequence attention
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64 # [B 1 1 Len]
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)  # [1 1 len len]
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # attention_mask
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb = item_seq_emb + position_emb
        item_seq_emb = self.LayerNorm(item_seq_emb)
        item_seq_emb = self.dropout(item_seq_emb)

        sequence_output = self.item_encoder(item_seq_emb,
                                            extended_attention_mask,
                                            output_all_encoded_layers=True)[-1] # [B L H]
        return sequence_output

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