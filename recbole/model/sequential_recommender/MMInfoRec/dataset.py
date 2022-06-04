# -*- coding: utf-8 -*-
# @Time    : 2021/8
# @Author  : Shuqing Bian

import torch
from torch.utils.data import Dataset


class MMInfoRecDataset(Dataset):
    """
    Same dataset for training and testing
    """
    
    def __init__(self, args, user_seq, max_att_num, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.max_att_num = max_att_num
        self.data_type = data_type
        self.max_len = args.max_seq_length
    
    def __getitem__(self, index):
        
        user_id = index
        items = self.user_seq[index]
        
        assert self.data_type in {"train", "valid", "test"}
        
        if self.data_type == "train":
            input_ids = items[:-2]
            answer = 0  # no use
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            answer = items[-2]
        else:
            input_ids = items[:-1]
            answer = items[-1]
        
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
        
        attributes = []
        att_att_mask = []
        for item in input_ids:
            att = self.args.item2attribute[str(item)]
            attributes.append(att + [0] * (self.max_att_num - len(att)))
            
            # [0] + 1 for the id_embedding appended to the start of the attribute embeddings
            att_att_mask.append([0.] * (len(att) + 1) + [1.] * (self.max_att_num - len(att)))
        
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(len(input_ids), dtype=torch.long),
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(att_att_mask, dtype=torch.float)
        )
        
        return cur_tensors
    
    def __len__(self):
        return len(self.user_seq)


def collate_fn(batch):
    """
    collate sequences of different lengths for single item attention
    """
    
    user_id = torch.tensor([sample[0] for sample in batch])
    answer = torch.tensor([sample[1] for sample in batch])
    input_ids = torch.cat([sample[2] for sample in batch])
    seq_len = torch.tensor([sample[3] for sample in batch])
    attributes = torch.cat([sample[4] for sample in batch])
    att_att_mask = torch.cat([sample[5] for sample in batch])
    
    return user_id, answer, input_ids, seq_len, attributes, att_att_mask
