# IS_HW3

Todo:

1.) Add one-hot array as labels. (done)
2.) Validate that forward and back prop are working on a small toy example. (done)
2.01) Confirmed that forward prop works as expected. (done)
2.02) Backprop through 1st layer is good (done)

2.1) Apply weight updates for each layer either per epoch or per sample (done)

2.5) Rework network for MNIST data. (done)
3.) Benchmark timing for one hidden layer for MNIST sized problem. (done). 5 epoch/second on my tower with 10% of the training data each epoch.

5.) Store hit rate every 10th (done)

7.) Add momentum (done)
8.) cache final information trained network (done)
9.) plot error rate per epoch (done)

4.) Add a threshold to prevent learning if "good enough" i.e. above a target value.
6.) Maybe validate on the other portion of the training data that was not used in that batch.
10.) Do a grid-search to find the best test-accuracy.


# IS_HW4

Building off of what I had previously to implement the modifications for HW4

1.) Conditionally implement sparseness penalty (done)
2.) Conditionally implement weight decay (done)
3.) Apply different loss functions for different layers. (done)
4.) Selectively freeze different layers.  (done)


best network so far. Take HW4_1's weights, pop off the layer, put on a randomized
classification layer. Then, set the hidden layer LR to 0.001 and the output LR to
0.05 and train it for about 500 epochs. Got 96.3% Accuracy!

FYI -- Accuracy got w/ first classification network on the data set used in HW 4 is 95.8%
