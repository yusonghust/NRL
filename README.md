# NRL
Network Representation Learning   
Tensorflow implemention of deepwalk and non-negative matrix factorization   

**Deepwalk and NMF**      
Args:   
```
--edgelist: edgelist file, looks like node1 node2 <weight_float, optional>;   
--weighted: treat the graph as weighted; this is an action;   
--directed: treat the graph as directed; this is an action;   
--labelfile: node labels, looks like node_id label_id;   
--lr: learning rate, default is 0.01;   
--epochs: training epochs, default is 200;   
--dims: representation dimension;   
--neg_size: neagtive sampling number(only for deepwalk);   
--walk_length: random walk length(only for deepwalk);   
--num_walks: number of walks foe each node(only for deepwalk);   
--clf-retio: the ratio of training data for node classification; the default is 0.5;
```   
if you want to test deepwalk in cora dataset:   
```cd deepwalk```   
```python main.py --edgelist ./data/cora/cora.edgelist --labelfile ./data/cora/cora.labels --embfile ./data/cora/cora.emb```   

if you want to test nmf in cora dataset:   
```cd NMF```   
```python main.py --edgelist ./data/cora/cora.edgelist --labelfile ./data/cora/cora.labels --embfile ./data/cora/cora.emb```   

**node classification result of deepwalk**   
|dataset|cora|wiki|
|:---|:---|:---|
|micro-f1|0.774|0.569|
|macro-f1|0.759|0.418|
   
**node classification result of NMF**   
|dataset|cora|wiki|
|:---|:---|:---|
|micro-f1|0.710|0.618|
|macro-f1|0.694|0.479|
