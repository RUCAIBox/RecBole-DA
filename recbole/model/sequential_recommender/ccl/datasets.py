import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample

class CPDataset(Dataset):

    def __init__(self, args, user_seq):
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.item_size = args.item_size
        self.mask_item_token = args.mask_item_token
        self.aug_type_1 = args.aug_type_1
        self.aug_type_2 = args.aug_type_2

        self.crop_ratio = args.crop_ratio
        self.mask_ratio = args.mask_ratio
        self.reorder_ratio = args.reorder_ratio

    def item_crop(self, item_seq):
        len_crop = int(len(item_seq)*self.crop_ratio)
        begin = random.randint(0, len(item_seq)-len_crop)  # 确定能满足crop的长度
        aug_seq = item_seq[begin:begin+len_crop]
        return aug_seq

    def item_mask(self, item_seq):
        aug_seq = []
        for item in item_seq:
            prob = random.random()
            if prob < self.mask_ratio:
                aug_seq.append(self.mask_item_token)
            else:
                aug_seq.append(item)
        return aug_seq

    def item_reorder(self, item_seq):
        len_reorder = int(len(item_seq) * self.reorder_ratio)
        begin = random.randint(0, len(item_seq) - len_reorder)  # 确定能满足reorder的长度
        reorder_part = item_seq[begin:begin+len_reorder]
        random.shuffle(reorder_part)
        aug_seq = item_seq[:begin]+reorder_part+item_seq[begin+len_reorder:]
        return aug_seq

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):

        item_seq = self.user_seq[index]  # pos_items

        if self.aug_type_1 == 'crop':
            aug_seq_1 = self.item_crop(item_seq)
        elif self.aug_type_1 == 'mask':
            aug_seq_1 = self.item_mask(item_seq)
            assert len(aug_seq_1) == len(item_seq)
        elif self.aug_type_1 == 'reorder':
            aug_seq_1 = self.item_reorder(item_seq)
            assert len(aug_seq_1) == len(item_seq)
        else:
            raise NotImplementedError

        pad_len = self.max_len - len(aug_seq_1)
        aug_seq_1 = [0] * pad_len + aug_seq_1
        aug_seq_1 = aug_seq_1[-self.max_len:]

        if self.aug_type_2 == 'crop':
            aug_seq_2 = self.item_crop(item_seq)
        elif self.aug_type_2 == 'mask':
            aug_seq_2 = self.item_mask(item_seq)
            assert len(aug_seq_2) == len(item_seq)
        elif self.aug_type_2 == 'reorder':
            aug_seq_2 = self.item_reorder(item_seq)
            assert len(aug_seq_2) == len(item_seq)
        else:
            raise NotImplementedError

        pad_len = self.max_len - len(aug_seq_2)
        aug_seq_2 = [0] * pad_len + aug_seq_2
        aug_seq_2 = aug_seq_2[-self.max_len:]


        # padding sequence
        # pad_len = self.max_len - len(item_seq)
        # item_seq = [0] * pad_len + item_seq
        # item_seq = item_seq[-self.max_len:]
        # assert len(item_seq) == self.max_len

        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        seqs = [aug_seq_1, aug_seq_2]
        cur_tensors = (torch.tensor(seqs, dtype=torch.long)
                       )
        return cur_tensors

class FDSADataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):

        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.max_attr_len = args.max_attr_length

        self.item2attribute = args.item2attribute
        self.item_size = args.item_size

        self.data_type = data_type
        self.test_neg_items = test_neg_items


    def neg_sample(self, sequence):  # 前闭后闭
        item = random.randint(1, self.item_size)
        while item in sequence:
            item = random.randint(1, self.item_size)
        return item

    def pad_attributes(self, attributes):
        pad_len = self.max_attr_len - len(attributes)
        attributes = attributes + [0]*pad_len
        attributes = attributes[-self.max_attr_len:]
        return attributes

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test"}
        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))


        attrs = []
        for item in input_ids:
            attrs.append(self.pad_attributes(self.item2attribute.get(str(item), [])))

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        assert len(attrs) == self.max_len


        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(index, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(attrs, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(attrs, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(index, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)