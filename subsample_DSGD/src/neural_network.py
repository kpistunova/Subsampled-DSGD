import numpy as np

def sigmoid(Z):
    """Computes and returns sigmoid activation function evaluated element-wise for the vector Z"""
    return 1/(1+np.exp(-Z))

def relu(Z):
    """Computes and returns relu activation function evaluated element-wise for the vector Z"""
    return np.maximum(0,Z)

def identity(Z):
    return np.copy(Z)

def sigmoid_backward(dA, Z):
    """Computes and returns gradient for sigmoid activation"""
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    """Computes and returns gradient for relu activation"""
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def identity_backward(dA, Z):
    return np.copy(dA)

def entropy_loss(Z, Z_label):
    """Computes and returns entropy loss between NN output Z, and dataset Z_label"""
    cost = -1 / Z.shape[0] * (np.matmul(Z_label.T, np.log(Z)) + np.matmul(1 - Z_label.T, np.log(1 - Z)))
    #Z_tmp = np.minimum(np.maximum(Z, -709), 709)
    #cost = 1 / Z.shape[0] * (np.matmul(Z_label.T, np.log(1 + np.exp(-Z_tmp))) + np.matmul(1 - Z_label.T, np.log(1 + np.exp(Z_tmp))))
    return np.squeeze(cost)

def entropy_loss_grad(Z, Z_label):
    """Computes and returns gradient for entropy loss between NN output Z, and dataset Z_label"""
    return -(np.divide(Z_label, Z) - np.divide(1 - Z_label, 1 - Z))
    #Z_tmp = np.minimum(np.maximum(Z, -709), 709)
    #return (1 - Z_label)/(1 + np.exp(-Z_tmp)) - Z_label/(1 + np.exp(Z_tmp))

def mse_loss(Z, Z_label):
    """Computes and returns mse loss between """
    return np.mean((Z-Z_label)**2)/2

def mse_loss_grad(Z, Z_label):
    return (Z - Z_label)/Z.shape[0]


class neural_network():

    """
        Class describing all the aspects of the neural network model
        Each node stores an object of this class
        nn_arch: neural network architecture, list of dictionaries; each dictionary indicates one layer, and contains "input_dim", "output_dim", "activation"
        loss_op: Type of loss function used for the problem
    """

    def __init__(self, nn_arch, loss_op='entropy_loss', acc_op='binary_class'):
        
        #Defines the architecture of the neural network
        self.nn_arch = nn_arch

        #Number of layers 
        self.number_of_layers = len(self.nn_arch)

        #Neural Network weights
        self.params_values = {}

        #Loss function used
        if loss_op == 'entropy_loss':
            self.loss_op = entropy_loss
            self.loss_grad = entropy_loss_grad
        elif loss_op == 'mse_loss':
            self.loss_op = mse_loss
            self.loss_grad = mse_loss_grad
        else:
            raise NotImplementedError

        if acc_op == 'binary_class':
            self.acc_op = neural_network.binary_accuracy
        elif acc_op == 'regression':
            self.acc_op = mse_loss
        else:
            raise NotImplementedError


    def init_layers(self, seed=99):
        """Initializes weights of the different layers of the NN"""

        #Random seed initialization
        np.random.seed(seed)

        #Iterations over network layers
        for idx, layer in enumerate(self.nn_arch):

            #Layers are numbered from 1
            layer_idx = idx + 1

            #Extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            #Initializing the values of the W matrix and vector b for subsequent layers
            self.params_values['W' + str(layer_idx)] = np.random.randn(layer_input_size, layer_output_size)*0.1
            self.params_values['b' + str(layer_idx)] = np.random.randn(1, layer_output_size)*0.1


    @staticmethod
    def binary_accuracy(probs, labels):
        """Takes the vector of probability values and does binary classification element-wise"""
        #probs_ = np.copy(probs)
        #probs_[probs_ > 0.5] = 1
        #probs_[probs_ <= 0.5] = 0
        #probs_[probs_ > 0.] = 1
        #probs_[probs_ <= 0.] = 0
        return np.mean((probs>0.5)==labels)

    @staticmethod
    def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation = "relu"):

        """
            Function evaluates the output of a single layer
            A_prev = Output of the previous layer
        """

        #Calculation of the input value for the activation function
        Z_curr = np.dot(A_prev, W_curr) + b_curr

        #Selection of activation function
        if activation is "relu":
            activation_func = relu
        elif activation is "sigmoid":
            activation_func = sigmoid
        elif activation is "linear":
            activation_func = identity
        else:
            raise Exception('Non-supported activation function')

        #Returning calculated activated values A and the intermediate values Z
        return activation_func(Z_curr), Z_curr

    @staticmethod 
    def full_forward_propagation(X, params_values, nn_architecture):
        """
            Performs a full forward propagation on input vector X to do a binary classification
            X: input vector
            params_values: neural network weights
            nn_architecture: Neural network architecture
        """

        #Creating a temporary memory to store the information needed for a backward step
        memory = {}

        #Vector X is the activation for layer 0
        A_curr = X

        #Iteration over network layers
        for idx, layer in enumerate(nn_architecture):

            #Number network layers from 1
            layer_idx = idx + 1

            #Transfer the activation from the previous iteration
            A_prev = A_curr

            #Extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            #Extraction of W for the current layer
            W_curr = params_values["W" + str(layer_idx)]

            #Extraction of b for the current layer
            b_curr = params_values["b" + str(layer_idx)]

            #Calculation of activation for the current layer
            A_curr, Z_curr = neural_network.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            #Saving calculated values in the memory
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        
        memory["Y"] = A_curr
        #Returning prediction vector (final neural network output) and a dictionary containing intermediate values
        return A_curr, memory


    @staticmethod
    def apply(data, test_point):
        '''
        to achieve a uniform interface with other ops, 
            the data is a tuple of (_inputs, _labels)
            the test_point is a tuple of (nn_architecture, loss_op, params_values)
        '''
        
        # unfold test_point to architecture + params_values
        nn_architecture = test_point[0]
        loss_op = test_point[1]
        params_values = test_point[2]
        labels = data[1]
        
        As, memory = neural_network.full_forward_propagation(data[0], params_values, nn_architecture)
        loss = loss_op(As, labels)
        return loss, memory

    @staticmethod
    def accuracy(test_data, test_point):
        '''
        to achieve a uniform interface with other ops, 
            the test_data is a tuple of (_inputs, _labels)
            the test_point is a tuple of (nn_architecture, acc_op, params_values)
        '''
        
        # unfold test_point to architecture + params_values
        nn_architecture = test_point[0]
        acc_op = test_point[1]
        params_values = test_point[2]
        labels = test_data[1]
        
        As, memory = neural_network.full_forward_propagation(test_data[0], params_values, nn_architecture)
        #Y_hat = acc_op(As)

        #accuracy = np.mean(Y_hat == labels)
        accuracy = acc_op(As, labels)
        return accuracy


    @staticmethod
    def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation = "relu"):

        #Number of examples
        m = A_prev.shape[0]

        #Selection of activation function
        if activation is "relu":
            backward_activation_func = relu_backward
        elif activation is "sigmoid":
            backward_activation_func = sigmoid_backward
        elif activation is "linear":
            backward_activation_func = identity_backward
        else:
            raise Exception('Non-supported activation function')

        #Calculation of activation function derivative
        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        #Derivative with respect to matrix W
        dW_curr = np.matmul(A_prev.T, dZ_curr)/m

        #Derivative of the vector b
        db_curr = np.mean(dZ_curr, axis=0, keepdims=True)

        #Derivative of the matrix A_prev
        dA_prev = np.matmul(dZ_curr, W_curr.T)

        return dA_prev, dW_curr, db_curr

    @staticmethod
    def full_backward_propagation(Y, memory, params_values, nn_architecture, loss_grad):
        """
            Function evaluates and returns gradients w.r.t. all model parameters
            Y_hat: Label returned by the current model
            Y: Label present in the dataset
        """ 

        grads_values = {}
        Y_hat = memory["Y"]

        #A hack ensuring the same shape of the prediction vector and labels vector
        Y = Y.reshape(Y_hat.shape)

        #Initiation of backpropagation algorithm to compute gradients
        dA_prev = loss_grad(Y_hat, Y)

        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):

            #Number network layers from 1
            layer_idx_curr = layer_idx_prev + 1

            #Extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = neural_network.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
            
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values


    @staticmethod
    def grad(data, test_point):
        '''
        to achieve a uniform interface with other ops, 
            the data is a tuple of (_inputs, _labels)
            the test_point is a tuple of (nn_architecture, loss_grad, params_values, memory)
        '''
        return neural_network.full_backward_propagation(data[1], test_point[3], test_point[2], test_point[0], test_point[1])


if __name__ == '__main__':
    pass

