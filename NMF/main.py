from nmf import NMF
import networkx as nx
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from collections import Counter
from classify import *
import os
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--edgelist',required=True,help='input edgelist file')
    parser.add_argument('--weighted',action='store_true',help='treat graph as weighted')
    parser.add_argument('--directed',action='store_true',help='treat graph as directed')
    parser.add_argument('--labelfile',required=True,help='input labels file')
    parser.add_argument('--embfile',required=True,help='save embeddings')
    parser.add_argument('--dims',default=128,type=int,help='representation dimensions')
    parser.add_argument('--epochs',default=200,type=int,help='number of training epochs')
    parser.add_argument('--clf_ratio',default=0.5,type=float,help='training data ratio')
    args = parser.parse_args()
    return args

def save_embeddings(embeddings,node_size,nodes,embfile):
    N = node_size
    assert np.shape(embeddings)[0]==N
    dims = np.shape(embeddings)[1]
    if os.path.exists(embfile):
        os.remove(embfile)
    with open(embfile,'w') as f:
        ls = str(N) + ' ' + str(dims) + '\n'
        f.write(ls)
        for k in range(N):
            vec = embeddings[k]
            ls = str(nodes[k])
            for v in vec:
                ls = ls + ' ' + str(v)
            ls = ls + '\n'
            f.write(ls)
    f.close()
    print('save embeddings done!')

def main(args):
    if args.weighted:
        G = nx.read_edgelist(args.edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.edgelist, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()

    nodes = list(G.nodes())
    adj = nx.to_numpy_matrix(G,nodes)
    node_size = 0
    look_up = {}
    for n in nodes:
        look_up[n] = node_size
        node_size += 1

    model = NMF(max_iter=args.epochs,display_step=10)
    W,H = model.fit_transform(adj,r_components=args.dims, initW=False, givenW=0)
    save_embeddings(W,node_size,nodes,args.embfile)
    X, Y = read_node_label(args.labelfile)
    vectors = load_embeddings(args.embfile)
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, args.clf_ratio)


if __name__ == '__main__':
    main(parse_args())