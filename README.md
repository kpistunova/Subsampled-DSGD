# Subsampled-DSGD
The repository of the code base of "Challenges in Scalable Distributed Training of Deep Neural Networks under communication constraints". Full report can be found here https://cs230.stanford.edu/projects_winter_2021/reports/70762032.pdf 

In the present era of ever-growing datasets and model sizes, distributed training of neural networks has become a crucial
paradigm. In this project, we will explore the challenges that come up when multiple machines train a model together in
a decentralized fashion. When multiple nodes collaborate to train a model together over a network, practical constraints such as
network bandwidth and latency overwhelm the expected benefits of the distributed framework.

We consider the problem of decentralized non-convex optimization under communication constraints, and propose an algorithm:
Subsampled Decentralized Stochastic Gradient Descent (Subsampled - DSGD), an algorithm in which a node randomly
subsamples a vector and projects it onto a lower dimensional subspace before communicating with other nodes, thereby reducing
communication demands.

We test the algorithm on two tasks: binary classification for the sklearn.make_moon dataset and regression for a residential energy consumption dataset.
