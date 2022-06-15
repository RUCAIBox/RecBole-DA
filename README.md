# RecBole-DA

**RecBole-DA** is a library built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing data augmentation for sequential recommendation. 

## 1）Highlights

* **Easy-to-use API**:
    Our library provides extensive API based on common data augmentation strategies, users can further develop own new models based on our library.
* **Full Coverage of Classic Methods**:
    We provide seven data augmentation methods based on recommender systems in three major categories.

## 2）Implemented Models

Our library includes algorithms covering three major categories:

* **Heuristic-based Methods**: CL4SRec, DuoRec
* **Model-based Methods**: MMInfoRec, CauseRec
* **Hybird Methods**: CASR, CCL, CoSeRec

## 3）Requirements

```
recbole>=1.0.0
pytorch>=1.7.0
python>=3.7.0
```

## 4）Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_seq.py --dataset='ml-1m' --train_batch_size=256 lmd=0.1 --lmd_sem=0.1 --model='CL4SRec' --contrast='us_x' --sim='dot' --tau=1
```

If you want to change the models or datasets, just run the script by setting additional command parameters:

```bash
python run_seq.py -m [model] -d [dataset]
```

## 5）The Team

RecBole-DA is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the developer is Shuqing Bian ([@fancybian](https://github.com/fancybian)).
