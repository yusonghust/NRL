# -*- coding: utf-8 -*-
###simple link prediction task###
#Reference from https://github.com/lucashu1/link-prediction/blob/master/node2vec.ipynb#
import numpy as np
import random
import networkx as nx
import itertools as it
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

class predictor():
    def __init__(self,emb_mat,edge_list,look_up,clf_ratio):
        self.emb_mat = emb_mat #embedding matrix
        self.look_up = look_up #node to id
        self.edgelist = edge_list
        self.clf_ratio = clf_ratio


    def preprocess(self):
        #preform train_test split#
        G = nx.read_edgelist(self.edgelist,nodetype=int,data=(('weight',float),), create_using=nx.Graph())
        pos_edges = list(G.edges())
        nodes = list(set(G.nodes()))
        neg_edges = random.sample(list(set(it.combinations(nodes,2)) - set(pos_edges)),len(pos_edges))

        random.shuffle(pos_edges)
        random.shuffle(neg_edges)

        ###split data###
        train_edge_pos = pos_edges[0:int(len(pos_edges)*self.clf_ratio)]
        train_edge_neg = neg_edges[0:int(len(pos_edges)*self.clf_ratio)]
        test_edge_pos = pos_edges[int(len(pos_edges)*self.clf_ratio):]
        test_edge_neg = neg_edges[int(len(pos_edges)*self.clf_ratio):]
        return train_edge_pos,train_edge_neg,test_edge_pos,test_edge_neg

    def get_edge_embeddings(self,edges):
        edge_embs = []
        for edge in edges:
            node_0 = edge[0]
            node_1 = edge[1]
            emb_0 = self.emb_mat[self.look_up[node_0]]
            emb_1 = self.emb_mat[self.look_up[node_1]]
            edge_emb = np.multiply(emb_0,emb_1)
            edge_embs.append(edge_emb)
        edge_embs = np.array(edge_embs)
        return edge_embs

    def evaluate(self):
        train_edge_pos,train_edge_neg,test_edge_pos,test_edge_neg = self.preprocess()
        pos_train_edge_embs = self.get_edge_embeddings(train_edge_pos)
        neg_train_edge_embs = self.get_edge_embeddings(train_edge_neg)
        train_edge_embs = np.concatenate([pos_train_edge_embs,neg_train_edge_embs])
        train_edge_labels = np.concatenate([np.ones(len(train_edge_pos)),np.zeros(len(train_edge_neg))])

        pos_test_edge_embs = self.get_edge_embeddings(test_edge_pos)
        neg_test_edge_embs = self.get_edge_embeddings(test_edge_neg)
        test_edge_embs = np.concatenate([pos_test_edge_embs,neg_test_edge_embs])
        test_edge_labels = np.concatenate([np.ones(len(test_edge_pos)),np.zeros(len(test_edge_neg))])

        edge_classifier = LogisticRegression(random_state = 0)
        edge_classifier.fit(train_edge_embs,train_edge_labels)
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        test_roc = roc_auc_score(test_edge_labels, test_preds)
        test_ap = average_precision_score(test_edge_labels, test_preds)
        print('link-prediction Test ROC score: ', str(test_roc))
        print('link-prediction Test AP score: ', str(test_ap))




