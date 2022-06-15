
# Experimental setting

**Dataset**: [Amazon-Beauty](http://jmcauley.ucsd.edu/data/amazon)

**Evaluation**: all users in target dataset, ratio-based 8:1:1, full sort

**Metrics**: Recall, Precision, NDCG, MRR, Hit

**Topk**: 10, 20, 50

**Properties**:
```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 20
POSITION_FIELD: position_id
load_col:
  inter: [session_id, item_id, timestamp]
user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 2000
valid_metric: MRR@10
eval_args:
  split: {'LS':"valid_and_test"}
  mode: full
  order: TO
neg_sampling: ~
```

For fairness, we restrict users' and items' embedding dimension as following. Please adjust the name of the corresponding args of different models.
```
embedding_size: 64
```

# Dataset Statistics
| Dataset      | #Users | #items | #Interactions | Sparsity |
|--------------|--------|--------|---------------|----------|
| Amazon-Beauty| 22363 | 12101 | 747827      | 99.93%   |


# Hyper-parameters

| Method      | Best hyper-parameters                                                                                                                                                      |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CCL**   | learning_rate=0.005<br/>mlp_hidden_size=[32,32,16,8]<br/>reg_weight=0.001                                                                                                  |
| **CL4SRec**    | learning_rate=0.0005<br/>share_embedding_size=32<br/>alpha=0.1<br/>reg_weight=0.0001                                                                                       |
| **DuoRec**   | learning_rate=0.0005<br/>mlp_hidden_size=[64,64]<br/>dropout_prob=0.3<br/>alpha=0.3<br/>base_model=Transformer                                                                   |
| **MMInfoRec** | learning_rate=0.00001                                                                                                                                                      |
| **CauseRec**  | learning_rate=0.0001<br/>n_layers=3<br/>drop_rate=0.1<br/>reg_weight=0.01                                  |
| **CASR**     | learning_rate=0.0005<br/>lambda=0.2<br/>gamma=0.1<br/>alpha=0.2                                                                                                            |
| **CoSeRec**   | learning_rate=0.001<br/>mapping_function=non_linear<br/>mlp_hidden_size=[128]<br/>reg_weight=0.01<br/> |
