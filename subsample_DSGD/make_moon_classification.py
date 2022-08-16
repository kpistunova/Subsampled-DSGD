from __future__ import absolute_import

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.algorithms import *
from src import neural_network
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def get_args():
    """
    Read input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_nodes', type=int, default=10)
    parser.add_argument('-samples_per_node', type=int, default=1000)
    parser.add_argument('-num_test_samples', type=int, default=2000)
    parser.add_argument('-num_realizations', type=int, default=1)
    parser.add_argument('-max_iters', type=int, default=1000)
    parser.add_argument('-subsample_ratio', type=float, default=1.)
    parser.add_argument('-gtype', choices = ['full', 'ring'], default='full')
    parser.add_argument('-ring_nbrs', type=int, default=1)
    parser.add_argument('-save_data', type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    num_nodes = 10
    num_realizations = 5
    max_iters = 3000
    num_sample_per_node = 1000
    num_test_samples = 2000

    # Define NN architecture
    NN_ARCHITECTURE = [
        {"input_dim": 2, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
    ]

    loss = [[] for _ in range(args.num_realizations)]
    accuracy = []
    for realiz_ind in tqdm(range(args.num_realizations)):
        #Define the network
        network = Network(args.num_nodes, gtype=args.gtype, ring_nbrs=args.ring_nbrs)

        # generate training and test data
        all_X, all_Y = make_moons(n_samples=args.samples_per_node*args.num_nodes, noise=0.2)
        all_data = (all_X, all_Y.reshape((-1,1)))
        test_X, test_Y = make_moons(n_samples=args.num_test_samples, noise=0.2)

        learning_rate = 0.1
        seed=np.random.randint(100)
        BinaryMoonsClassifier = System('entropy_loss', 'binary_class', all_data, network, NN_ARCHITECTURE, learning_rate, subsampling_ratio=args.subsample_ratio, seed=seed)

        #Run subsampling-DSGD algorithm for required number of iterations
        for tind in tqdm(range(args.max_iters)):
            BinaryMoonsClassifier.update()

        loss[realiz_ind] = [np.asarray(_node.local_loss) for _node in BinaryMoonsClassifier.nodes]
        accuracy.append(BinaryMoonsClassifier.nodes[0].local_accuracy((test_X, test_Y.reshape((-1,1)))))

    #Take ensemble average
    #Plot the training loss
    print(accuracy)
    ave_loss = np.array(loss).mean(0)
    if args.save_data:
        np.save('simulation_data/make_moon_loss_'+args.gtype+'_'+str(args.subsample_ratio)+'.npy', ave_loss)
        np.save('simulation_data/make_moon_acc_'+args.gtype+'_'+str(args.subsample_ratio)+'.npy', accuracy)
    plt.plot(ave_loss[0,:])
    plt.show()  