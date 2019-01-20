#-*-coding:utf-8-*-
#read graph and random walk
import networkx as nx
import numpy as np
import random
import collections

class graph():
    def __init__(self,edgelist,labelfile,num_walks,walk_length,weighted=False,directed=False):
        self.edgelist = edgelist
        self.labelfile = labelfile
        self.weighted = weighted
        self.directed = directed
        self.G = self.build_graph()
        self.degrees = dict(self.G.degree(list(self.G.nodes())))
        self.node_list = [item[0] for item in sorted(self.degrees.items(),key = lambda x:x[1])]
        self.look_up = {}
        self.node_size = 0
        for node in self.node_list:
            self.look_up[node] = self.node_size
            self.node_size += 1
        self.read_node_labels()
        self.pairs = self.rw(num_walks,walk_length)

    def build_graph(self):
        '''
        Reads the input network using networkx.
        '''
        if self.weighted:
            G = nx.read_edgelist(self.edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(self.edgelist, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
        if not self.directed:
            G = G.to_undirected()
        return G

    def read_node_labels(self):
        '''
        read node labels
        '''
        fin = open(self.labelfile, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[int(vec[0])]['label'] = vec[1:]
        fin.close()

    def rw(self,num_walks,walk_length):
        '''
        random walk
        '''
        G = self.G
        nodes = self.node_list
        walk_pairs = []
        # print('random walk start!')
        for n in nodes:
            if G.degree(n) == 0:
                continue
            for j in range(num_walks):
                current_n = n
                for k in range(walk_length+1):
                    # print(list(G.neighbors(current_n)))
                    neigs = list(G.neighbors(current_n))
                    if len(neigs)>0:
                        next_n = random.choice(neigs)
                    else:
                        break
                    if current_n != n:
                        walk_pairs.append((self.look_up[n],self.look_up[current_n]))
                    current_n = next_n
        return walk_pairs

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        look_up_dict = self.look_up
        node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        walk = [str(i) for i in walk]
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = self.node_list
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))
        return walks