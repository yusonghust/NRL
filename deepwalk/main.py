#-*-coding:utf-8-*-
#An simple implementation of deepwalk
#After get network embeddings, evaluate it using node classification task
import tensorflow as tf
import numpy as np
import networkx as nx
from utils import graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
import os
from classify import *
from linkpred import predictor
from gensim.models import Word2Vec
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--edgelist',required=True,help='input edgelist file')
    parser.add_argument('--labelfile',help='input labels file')
    parser.add_argument('--embfile',required=True,help='save embeddings in this file')
    parser.add_argument('--weighted',action='store_true',help='treat graph as weighted')
    parser.add_argument('--directed',action='store_true',help='treat graph as directed')
    parser.add_argument('--neg_size',default=50,type=int,help='negative sampling numbers')
    parser.add_argument('--dims',default=128,type=int,help='embedding dimensions')
    parser.add_argument('--batch_size',default=1024,type=int,help='batch size')
    parser.add_argument('--num_walks',default=100,type=int,help='number of random walks per node')
    parser.add_argument('--walk_length',default=6,type=int,help='walk length in random walk')
    parser.add_argument('--epochs',default=30,type=int,help='number of training epochs')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--clf_ratio',default=0.5,type=float,help='training data ratio')
    args = parser.parse_args()
    return args

class deepwalk():
    def __init__(self,G,dims,neg_sample_size,batch_size,epochs,clf_ratio,lr,embfile):
        '''
        Args:
        node_size: total node numbers;
        dims: embedding dimensions;
        neg_sample_size： negative sampling parameter;
        batch_size: training batch size;
        epochs: training epochs;
        clf_ratio: the ratio of labels data for training;
        lr: learning rate;
        embfile: save embeddings in this file
        '''
        self.G = G
        self.node_size = G.node_size
        self.dims = dims
        self.neg_sample_size = neg_sample_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.clf_ratio = clf_ratio
        self.lr = lr
        self.embfile = embfile

        self.label_mat,self.label_dict = self.build_label()
        self.train_mask,self.val_mask,self.test_mask = self.build_mask()
        self.build_placeholders()
        self.embs = tf.Variable(tf.random_uniform([self.node_size, dims], -1, 1),dtype=tf.float32,name="embeddings")
        self.nce_weights = tf.Variable(tf.random_uniform([self.node_size,dims],-1,1),dtype=tf.float32,name='weights')
        self.nce_biases = tf.Variable(tf.zeros(self.node_size),dtype=tf.float32,name='biases')

    def build_placeholders(self):
        self.input_1 = tf.placeholder(tf.int32,shape=(None))
        self.input_2 = tf.placeholder(tf.int32,shape=(None))

        self.input_3 = tf.placeholder(tf.float32,shape=(None,self.dims))
        self.labels = tf.placeholder(tf.float32, shape=(None, self.label_mat.shape[1]))
        self.label_mask = tf.placeholder(tf.int32)

    def construct_feed_dict_dw(self,src_idx,tar_idx):
        feed_dict = dict()
        feed_dict.update({self.input_1:src_idx})
        feed_dict.update({self.input_2:tar_idx})
        return feed_dict

    def construct_feed_dict_nc(self,node_embs,node_label,label_mask):
        feed_dict = dict()
        feed_dict.update({self.input_3:node_embs})
        feed_dict.update({self.labels:node_label})
        feed_dict.update({self.label_mask:label_mask})
        return feed_dict

    def skip_gram(self):
        emb_inputs = tf.nn.embedding_lookup(self.embs,self.input_1)
        nce_labels = tf.reshape(self.input_2,(-1,1))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_weights,
                biases = self.nce_biases,
                labels=nce_labels,
                inputs=emb_inputs,
                num_sampled=self.neg_sample_size,
                num_classes=self.node_size
                ))
        return loss

    def Skip_Gram(self):
        g = self.G
        degrees = dict(g.G.degree(self.G.node_list))
        degrees = list(degrees.values())
        labels = tf.reshape(tf.cast(self.input_2,tf.int64),[-1,1])
        neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels,
                num_true=1,
                num_sampled=self.neg_sample_size,
                unique=True,
                range_max=len(degrees),
                distortion=0.75,
                unigrams=degrees))

        sou_embs = tf.nn.embedding_lookup(self.embs,self.input_1)
        tar_embs = tf.nn.embedding_lookup(self.nce_weights,self.input_2)
        tar_embs_bias = tf.nn.embedding_lookup(self.nce_biases,self.input_2)

        neg_embs = tf.nn.embedding_lookup(self.nce_weights,neg_samples)
        neg_embs_bias = tf.nn.embedding_lookup(self.nce_biases,neg_samples)

        ###loss
        aff = tf.reduce_sum(tf.multiply(sou_embs,tar_embs),1) + tar_embs_bias
        neg_aff = tf.matmul(sou_embs,tf.transpose(neg_embs)) + neg_embs_bias

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff
        )

        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff
        )
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        loss = loss/tf.cast(self.batch_size,tf.float32)
        return loss

    def node_classfication(self):
        w = tf.Variable(tf.random_uniform([self.dims,self.label_mat.shape[1]],-1,1),dtype=tf.float32,name='w')
        b = tf.Variable(tf.zeros(self.label_mat.shape[1]),dtype=tf.float32,name='b')
        preds = tf.matmul(self.input_3,w) + b
        loss = self.masked_softmax_cross_entropy(preds,self.labels,self.label_mask)
        accuracy = self.masked_accuracy(preds,self.labels,self.label_mask)
        return loss,accuracy

    def build_label(self):
        '''
        Graph--Graph class defined in graph.py
        '''
        g = self.G
        G = g.G
        nodes = g.node_list
        look_up = g.look_up
        labels = []
        label_dict = {}
        label_id = 0
        for node in nodes:
            labels.append((node,G.nodes[node]['label']))
            for l in G.nodes[node]['label']:
                if l not in label_dict:
                    label_dict[l] = label_id
                    label_id += 1
        label_mat = np.zeros((len(labels),label_id))
        for node,l in labels:
            node_id = look_up[node]
            for ll in l:
                l_id = label_dict[ll]
                label_mat[node_id][l_id] = 1
        return label_mat,label_dict

    def build_mask(self):
        '''preprocess label mask'''
        train_percent = self.clf_ratio
        g = self.G
        node_size = g.node_size
        look_up = g.look_up
        training_size = int(train_percent * node_size)

        state = np.random.get_state()
        np.random.seed(0)
        shuffle_indices = np.random.permutation(np.arange(node_size))
        np.random.set_state(state)

        def sample_mask(begin,end):
            mask = np.zeros(node_size)
            for i in range(begin, end):
                mask[shuffle_indices[i]] = 1
            return mask

        train_mask = sample_mask(0,training_size - 100)
        val_mask = sample_mask(training_size - 100, training_size)
        test_mask = sample_mask(training_size, node_size)
        return train_mask,val_mask,test_mask

    def masked_softmax_cross_entropy(self, preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


    def masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def save_embeddings(self,embeddings):
        N = self.node_size
        nodes = self.G.node_list
        assert np.shape(embeddings)[0]==N
        if os.path.exists(self.embfile):
            os.remove(self.embfile)
        with open(self.embfile,'w') as f:
            ls = str(N) + ' ' + str(self.dims) + '\n'
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

    def train_and_evaluate(self):
        ###step 1: training skip_gram model to obtain network embeddings
        pairs = self.G.pairs
        N = len(pairs)
        dw_loss = self.skip_gram()
        # dw_loss = self.Skip_Gram()
        nc_loss,nc_accuracy = self.node_classfication()

        opt_dw = tf.train.AdamOptimizer(self.lr).minimize(dw_loss)
        opt_nc = tf.train.AdamOptimizer(self.lr).minimize(nc_loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config = config)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(self.epochs):
            training_pairs = np.random.permutation(pairs)
            head = 0
            loss_values = []
            while head<N:
                tail = head + self.batch_size
                mini_batch_pairs = training_pairs[head:min(tail,N)]
                head = tail
                src_idx = mini_batch_pairs[:,0]
                tar_idx = mini_batch_pairs[:,1]
                feed_dict = self.construct_feed_dict_dw(src_idx,tar_idx)
                _,loss_dw = sess.run([opt_dw,dw_loss],feed_dict = feed_dict)
                loss_values.append(loss_dw)
            print('-'*120)
            print('epochs is ',i+1,'\t loss is ',sum(loss_values)/len(loss_values))

        embeddings = self.embs.eval()
        self.save_embeddings(embeddings)
        return embeddings

def main(args):
    G = graph(args.edgelist,args.labelfile,args.num_walks,args.walk_length,args.weighted,args.directed)
    dw = deepwalk(G,args.dims,args.neg_size,args.batch_size,args.epochs,args.clf_ratio,args.lr,args.embfile)
    embeddings = dw.train_and_evaluate()
    walks = G.simulate_walks(10,100)
    word2vec = Word2Vec(walks, size=args.dims, window=10, min_count=0, sg=1, workers=8, iter=10)
    vectors = {}
    for word in G.node_list:
        vectors[str(word)] = word2vec.wv[str(word)]
    X, Y = read_node_label(args.labelfile)
    print("Training classifier using {:.2f}% nodes...".format(
        args.clf_ratio*100))
    print('gensim implementation deepwalk')
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, args.clf_ratio)
    print('tensorflow implementation deepwalk')
    vectors = load_embeddings(args.embfile)
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, args.clf_ratio)


if __name__ == '__main__':
    main(parse_args())







