# -*- coding: utf-8 -*-
"""
Definition of node, network, and system as well as the subsampling protocol for communication
"""

import numpy as np
import scipy as sc

from .neural_network import neural_network

class Node():

    """
        Implements behavior of a node 
        Stores local dataset
        Maintains local model
        Maintains an independent clock synchronized with other nodes
    """ 

    def __init__(self, local_data, NN:neural_network, **kwargs):

        """
            Constructor.
            local_data: Stores node's local dataset
            NN: (neural_network object) Local model
        """

        self.local_data = local_data
        self.NN = NN
        self.t = 0                      #Local clock
        self.local_loss = []

    def local_update(self, consensus_weights, learning_rate):
        """
            Function updates the local model.
            consensus_weights: Model weights passed by the system object after taking consensus from its neighbors
            learning_rate: Step size for stochastic gradient descent decided by the system object 
        """

        #Update local clock
        self.t += 1

        #Get gradients
        test_point = (self.NN.nn_arch, self.NN.loss_op, self.NN.params_values)
        loss, memory = neural_network.apply(self.local_data, test_point)
        self.local_loss.append(loss)
        test_point = (self.NN.nn_arch, self.NN.loss_grad, self.NN.params_values, memory)
        grads_values = neural_network.grad(self.local_data, test_point)

        # iteration over network layers
        for layer_idx, layer in enumerate(self.NN.nn_arch, 1):
            self.NN.params_values["W" + str(layer_idx)] = consensus_weights["W" + str(layer_idx)] - learning_rate * grads_values["dW" + str(layer_idx)]        
            self.NN.params_values["b" + str(layer_idx)] = consensus_weights["b" + str(layer_idx)] - learning_rate * grads_values["db" + str(layer_idx)]


    def current_param_val(self):

        """
            Function returns weights of the local model of the neural network.
            Required when system object computes consensus between nodes
        """

        return self.NN.params_values

    def local_accuracy(self, test_data):
        """
            Function calculates local classification accuracy
            test_data: tuple of (_inputs, _labels)
        """
        test_point = (self.NN.nn_arch, self.NN.acc_op, self.NN.params_values)
        return self.NN.accuracy(test_data, test_point)
    

class Network:

    """
        Maintains the connectivity matrix of the network of nodes
    """

    def __init__(self, num_nodes, gtype: str='full', mode: str='static', ring_nbrs=1, **kwargs):

        """
            mode: Allows for the network to be static/dynamic
            num_nodes: Number of worker nodes in the Network
            gtype: Type of graph
            ring_nbrs: #No. of neighbors on each side if it is a ring network
        """
        self.mode = mode
        self.num_nodes = num_nodes
        self.gtype = gtype
        self.ring_nbrs = ring_nbrs
        self.graph_init = self.create_graph()
        
    def doubly_stochastic_from_adjacency(self, adjacency):
        """ Function returns double stochastic P from adjacency matrix"""
        diag = np.diag(np.dot(adjacency, np.ones([self.num_nodes])))
        dmax = np.max(diag)
        doubly_stochastic = np.eye(self.num_nodes) + 1 / (dmax+1) * (adjacency - diag)
        return doubly_stochastic

    def create_graph(self):

        """
            Generates the returns the doubly stochastic connectivity matrix of the network
        """
        if self.gtype == 'full':
            doubly_stochastic = np.ones([self.num_nodes, self.num_nodes])/self.num_nodes
            
        elif self.gtype == 'ring':
            c_ = [0]+[1]*self.ring_nbrs+[0]*(self.num_nodes - 2*self.ring_nbrs - 1) + [1]*self.ring_nbrs
            adjacency = sc.linalg.toeplitz(c_)
            doubly_stochastic = self.doubly_stochastic_from_adjacency(adjacency)

        return doubly_stochastic


    def __call__(self):

        """
            Allows for the network to be time-varying.
            If time-varying, returns a newly generated connectivity matrix every time.
        """

        if self.mode == 'static':
            return self.graph_init

        else:
            return self.create_graph()

    @property 
    def doubly_stochastic(self):
        return self.__call__()


class System:
    """
    Class at the highest level which contains implementation of the distributed learning algorithm.
    It is a collection of nodes and the network that collects them.
    Manages local consensus of models for all the nodes
    """

    def __init__(self, loss_op, acc_op, all_data, network: Network, nn_architecture, learning_rate, subsampling_ratio=1., seed=99):
        """
        local_op: loss function employed for evaluating neural network performance at each node
        all_data: Global datset. To be split among different nodes
        network: Object of network class which maintains the connections between nodes
        """

        self.loss_op = loss_op              #Entropy loss for binary classification
        self.acc_op = acc_op
        self.all_data = all_data            #Global dataset
        self.network = network
        self.num_nodes = network.num_nodes
        self.nn_architecture = nn_architecture
        self.learning_rate = learning_rate
        self.subsampling_ratio = subsampling_ratio
        self.t = 0                                  #(Global) System clock

        batched_data = self.batch_data(self.all_data)
        self.nodes = self.initialize_nodes(batched_data, seed)


    #Call batch_data() function to split global dataset into batches
    def batch_data(self, all_data):
        # split the data among all the nodes
        # if all_data is stored in an array, split directly
        if isinstance(all_data, np.ndarray):
            num_data = int(all_data.shape[0] / self.num_nodes)
            indices_split = range(num_data, all_data.shape[0] - num_data + 1, num_data)
            yield from np.split(all_data, indices_split)
        # if all_data is stored in tuple(X, label), split separately
        if isinstance(all_data, tuple):
            num_data = int(all_data[0].shape[0] / self.num_nodes)
            indices_split = range(num_data, all_data[0].shape[0] - num_data + 1, num_data)
            batch_X = np.split(all_data[0], indices_split)
            batch_Y = np.split(all_data[1], indices_split)
            yield from tuple(zip(batch_X, batch_Y))

        

    #Call initialize_nodes() function to assign local datasets to each node
    def initialize_nodes(self, batched_data, seed=99):
        """ Creates nodes with specified NN architecture
            Assigns local datasets to each node.
        """

        nodes = []
        for data in batched_data:
            local_NN = neural_network(self.nn_architecture, self.loss_op, self.acc_op)
            local_NN.init_layers(seed)
            nodes.append(Node(data, local_NN))
        return nodes

    def update(self):
        """Does one update step for each node"""
        
        #Update global clock
        self.t += 1
        
        #Store the current weights and use them for consensus so that system is synchronous
        weights_current = {}
        for i in range(self.num_nodes):
            for layer_idx, layer in enumerate(self.nn_architecture, 1):     
                weights_current["W" + str(layer_idx) + str(i)] = np.copy(self.nodes[i].NN.params_values["W" + str(layer_idx)])
                weights_current["b" + str(layer_idx) + str(i)] = np.copy(self.nodes[i].NN.params_values["b" + str(layer_idx)])

        #Get the consensus matrix
        P = self.network.doubly_stochastic

        for i in range(self.num_nodes):

            #Compute consensus of neighborhood models
            consensus_weights = {}
            for layer_idx, layer in enumerate(self.nn_architecture, 1):
                consensus_weights["W" + str(layer_idx)] = 0
                consensus_weights["b" + str(layer_idx)] = 0

            for j in range(self.num_nodes):
                for layer_idx, layer in enumerate(self.nn_architecture, 1):
                    
                    #Define the message compressor
                    layer_input_size = layer["input_dim"]
                    
                    #tx_weights_col_dim is the compressed dimensionality of the rows of the weight matrix 
                    #Each row of the weight matrix is compressed. So the transmitted weight matrix 
                    #has the same number of rows but reduced number of columns
                    
                    tx_weights_col_dim = int(np.ceil(self.subsampling_ratio*layer_input_size))
                    
                    #Orthonormal matrix used for subsampling (identity matrix for now)
                    orthonormal = np.eye(layer_input_size)
                    
                    shuffled_indices = np.arange(0,layer_input_size)
                    np.random.shuffle(shuffled_indices)
                    indices = shuffled_indices[np.arange(self.t * tx_weights_col_dim, (self.t + 1) * tx_weights_col_dim) % layer_input_size]
                    
                    #Encoded and subsequently decoding the messages
                    compressed_weight_matrix = np.dot(weights_current["W" + str(layer_idx) + str(j)].T, orthonormal[:,indices])
                    decompressed_weight_matrix = np.dot(compressed_weight_matrix, np.conj(orthonormal[:,indices].T)).T/tx_weights_col_dim*layer_input_size
                    
                    #Need to ensure that a particular node does not compress/decompress its own message
                    if j == i:
                        consensus_weights["W" + str(layer_idx)] += P[i,j]*weights_current["W" + str(layer_idx) + str(j)]    
                    else:   
                        consensus_weights["W" + str(layer_idx)] += P[i,j]*decompressed_weight_matrix
                    
                    #Keep the biases uncompressed
                    consensus_weights["b" + str(layer_idx)] += P[i,j]*weights_current["b" + str(layer_idx) + str(j)]

            #Take a descent step
            self.nodes[i].local_update(consensus_weights, self.learning_rate)



if __name__ == '__main__':
    pass
