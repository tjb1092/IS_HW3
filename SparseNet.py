import numpy as np
import math
import random
import time
import pickle
import matplotlib.pyplot as plt
from FeedForwardNN import NN, save_data
from Autoencoder import eval_net, plot_errorbar, plot_errortime, plot_features


def main():
	# Load previous data
	saved_data = pickle.load(open("cached_run/saved_data_3_2.p", "rb"))
	data = saved_data["Data"]
	savedata = False
	# Initialize the network
	epochs = 3000
	H = [100]
	f = 784
	c = 784
	LR = 0.0005
	a = 0.7
	#p=0.1, B = 1, LR = 0.0005, a=0.7, l=0.0001 bout tied w/ 1st
	#p=0.1, B = 1, LR = 0.0005, a=0.7, l=0.00005 best so far
	#p=0.05, B = 1, LR = 0.001, a=0.7, l=0.00005 2nd best

	p = 0.1
	B = 1
	l = 0.00005
	nn = NN(H, [LR, LR], f, c, a, B, p, l)

	mini_batch_per = 0.2
	x_shape = data["x_train"].shape
	last_time = time.time()
	error_train = []
	error_valid = []
	epoch_arr = []
	best_validation = 1e6  # Initialize "best error" extremely high so it will be overridden.
	best_epoch = 0
	for epoch in range(epochs):

		# Pick mini-back of training data
		# Sample 20% of the indices
		batch_index = random.sample(range(x_shape[0] - 1), math.ceil(mini_batch_per*x_shape[0]))
		x_train = data["x_train"][batch_index]
		acc_error = 0
		h_xq_acc = [[] for k in range(nn.n)]
		#First, accumulate the p_hatJ for this epoch/minibatch.
		for i, X in enumerate(x_train):
			X = X.reshape(len(X),1)
			X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term

			delW, y_hatq, h_j_xq = nn.fit(X, X_label, 1)

			for j in range(len(h_xq_acc)):
				# Accumulate the activation for each hidden neuron over the minibatch.
				if i == 0:
					h_xq_acc[j] = h_j_xq[j][1:]
				else:
					h_xq_acc[j] = h_xq_acc[j] + h_j_xq[j][1:]

		p_hat = [np.divide(x,len(x_train)) for x in h_xq_acc]  # Compute p_hat_j for all neurons.

		# Now accumulate the weight changes
		for i, X in enumerate(x_train):
			X = X.reshape(len(X),1)
			X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term

			delW, y_hatq, h_j_xq = nn.fit(X, X_label, 0, p_hat)
			acc_error += (1./2.)*np.sum(np.square(np.subtract(X_label, y_hatq)))  # Error formula
			if i == 0:
				acc_delW = delW
			else:
				# Accumulate weight change for all training points per epoch
				for j, dW in enumerate(delW):
					acc_delW[j] += dW

		if epoch == 0:
			past_delW = acc_delW
			nn.update_weights(acc_delW, past_delW, 1)  # Apply accumulated weights at end of epoch.
		else:
			nn.update_weights(acc_delW, past_delW, 0)  # Apply accumulated weights at end of epoch.
			past_delW = acc_delW

		if (epoch % 10) == 0:
			# measure error on the validation set
			valid_errors = eval_net(nn, data, "validate")
			error_valid.append(valid_errors[0])
			print("Epoch: {}".format(epoch))
			error_train.append(acc_error)
			epoch_arr.append(epoch)

			print("Training Error: {}".format(acc_error))
			print("Validation Error: {}".format(valid_errors[0]))
			if valid_errors[0] < best_validation:
				# If total validation error improves,
				best_validation = valid_errors[0]
				print("Validation Improved!")
				nn.best_layers = nn.layers  # Store best weights
				best_epoch = epoch
			print("Best Epoch: {}".format(best_epoch))
			print('Epoch took {:0.3f} seconds'.format(time.time()-last_time))
			last_time = time.time()

	# Evaulate Test Accuracy
	nn.layers = nn.best_layers  # Roll-back to best weights
	train_errors = eval_net(nn, data, "train")
	test_errors = eval_net(nn, data, "test")

	plot_errorbar(train_errors, test_errors)
	plot_errortime(epoch_arr, error_train, error_valid)
	plot_features(nn.layers[0])
	plt.show()

	# Store the network, data, and parameters for future use.
	if savedata:
		save_data("3_3", nn.best_layers, data, nn.LR, nn.a)

if __name__ == "__main__":
    main()
