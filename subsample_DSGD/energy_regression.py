#This code trains an NN using a training data stored over a decentralized network of nodes
from __future__ import absolute_import

import numpy as np
import pandas as pd
from src.algorithms import *
from src import neural_network
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def get_args():
    """
    Read input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_nodes', type=int, default=10)
    parser.add_argument('-num_realizations', type=int, default=1)
    parser.add_argument('-max_iters', type=int, default=1000)
    parser.add_argument('-subsample_ratio', type=float, default=1.)
    parser.add_argument('-gtype', choices = ['full', 'ring'], default='full')
    parser.add_argument('-ring_nbrs', type=int, default=1)
    parser.add_argument('-save_data', type=bool, default=False)
    return parser.parse_args()

def energydata_generator():
    data_pd = pd.read_csv('energydata_complete.csv', index_col=0)
    data = data_pd.to_numpy(dtype='float', copy=True)
    date_time_pd = pd.read_csv('energydata_complete.csv', usecols=[0])
    date_time_pd = pd.to_datetime(date_time_pd['date'])
    weekdays = date_time_pd.dt.dayofweek.to_numpy(dtype='float').reshape((-1,1))
    IsWeekday = (weekdays >= 5).astype(float).reshape((-1,1))
    time_of_day = (date_time_pd.dt.hour*6 + date_time_pd.dt.minute/10).to_numpy(dtype='float').reshape((-1,1))

    data = np.hstack((data, weekdays, IsWeekday, time_of_day))
    target_offset = data[:,0].min()
    target_scale = data[:,0].ptp()
    data = (data - data.min(0)) / data.ptp(0)

    np.random.shuffle(data)

    train_set = data[:17760, :]
    test_set = data[17760:, :]
    np.save('train_set.npy', train_set)
    np.save('test_set.npy', test_set)
    np.save('offsets.npy', np.array([target_offset, target_scale]))
    #plt.plot(data[:,0])
    #plt.show()
    return train_set, test_set, np.array([target_offset, target_scale])

if __name__ == "__main__":
    
    # read inputs
    args = get_args()

    # load data
    try:
        data_train = np.load('train_set.npy')
        data_test = np.load('test_set.npy')
        target_offsets = np.load('offsets.npy')
    except:
        data_train, data_test, target_offsets = energydata_generator()
    
    num_sample_per_node = int(data_train.shape[0]/args.num_nodes)

    #Common architecture for all models across the nodes
    NN_ARCHITECTURE = [
        {"input_dim": data_train.shape[1]-1, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 1, "activation": "linear"},
    ]

    loss = [[] for _ in range(args.num_realizations)]
    mse = []
    for realiz_ind in tqdm(range(args.num_realizations)):
        #Define the network
        network = Network(args.num_nodes, gtype=args.gtype, ring_nbrs=args.ring_nbrs)

        # shuffle training data for each realization
        np.random.shuffle(data_train)

        learning_rate = 2
        seed=np.random.randint(100)
        EnergyPredictor = System('mse_loss', 'regression', (data_train[:, 1:], data_train[:,0:1]), network, NN_ARCHITECTURE, learning_rate, subsampling_ratio=args.subsample_ratio, seed=seed)

        #Run the algorithm for required number of iterations
        for tind in tqdm(range(args.max_iters)):
            EnergyPredictor.update()

        loss[realiz_ind] = [np.asarray(_node.local_loss) for _node in EnergyPredictor.nodes]
        mse.append(EnergyPredictor.nodes[0].local_accuracy((data_test[:, 1:], data_test[:, 0:1])))

    ave_loss = np.array(loss).mean(0)
    if args.save_data:
        np.save('simulation_data/energy_loss_'+args.gtype+'_'+str(args.subsample_ratio)+'.npy', ave_loss)
        np.save('simulation_data/energy_acc_'+args.gtype+'_'+str(args.subsample_ratio)+'.npy', mse)
    #Plot the training loss
    print(mse)
    plt.plot(loss[0][0])
    plt.show()

    