"""
Autoencoder.py
using Python 3.5/6
Used for problem 2 in HW 3. It can be ran through a terminal using the command
"python3 Autoencoder.py" or "python Autoencoder.py" depending on how python is setup.
It will run as long as a folder called "cached_run" is created and the
scripts "data_load.py" and "FeedForwardNN" are in the same directory.
"""
import numpy as np
from data_load import preprocessData, create_train_test_split
import math
import random
import time
import pickle
import matplotlib.pyplot as plt
from FeedForwardNN import NN, save_data

def eval_net(nn, data, mode):
	# Iterates over each digit and accumulates the error for each digit
	# as well as the total overall error.
	total_error = 0
	errors = []
	for c in range(10):
		plt_reconstruct = True
		digit_error = 0
		# For each digit, filter out X data points
		# Iterate through the labels and pull out indicies for each class
		i_list = []
		for index, label in enumerate(data["y_{}".format(mode)]):
			if label[c] == 1:
				# Store index if the label matches the current class.
				i_list.append(index)
		# Now enumerate over the matching image data for the class
		for i, X in enumerate(data["x_{}".format(mode)][i_list]):
			X = X.reshape(len(X),1)
			X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term
			delW, y_hatq, f_s_aq = nn.fit(X,X_label, 1)
			digit_error += (1./2.)*np.sum(np.square(np.subtract(X_label, y_hatq)))

			if plt_reconstruct and mode == "test":
				# Debug plotting. Proving that network can actually reconstruct images
				fig, ax = plt.subplots(1,2 )
				ax[0].imshow(X_label.reshape(28,28).T, cmap='gray')
				ax[0].set_title("Input Digit")
				ax[0].axis('off')
				ax[1].imshow(y_hatq.reshape(28,28).T, cmap='gray')
				ax[1].set_title("Reconstructed Digit")
				ax[1].axis('off')
				fig.show()
				plt_reconstruct = False  # Only show one image per class

		errors.append(digit_error)
		total_error += digit_error
	errors.insert(0, total_error)  # Prepend total_error for plotting
	return errors

def plot_errorbar(train_errors, test_errors):
	# Plot error bar graph.
	index = np.arange(11)
	bar_width = 0.35
	fig, ax = plt.subplots()
	rects1 = ax.bar(index, train_errors, bar_width,
					alpha=0.6, color='b', label='Train')
	rects2 = ax.bar(index+bar_width, test_errors, bar_width,
					alpha=0.6, color='r', label='Test')
	ax.set_xlabel("Digits")
	ax.set_ylabel("Accumulated Error")
	ax.set_title("Accumulated Error per Digit")
	ax.set_xticks(index+ bar_width / 2)
	ax.set_xticklabels(('Total', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
	ax.legend()
	fig.show()
	return fig, ax

def plot_errortime(epoch_arr, error_train, error_valid):
	# Plot error rate per epoch
	fig, ax = plt.subplots(figsize=(6,6))
	ax.plot(epoch_arr, error_train, label="Training Error")
	ax.plot(epoch_arr, error_valid, label="Validation Error")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Error (J)")
	ax.set_title("Error per Epoch")
	ax.legend()
	fig.show()
	return fig, ax

def plot_features(W):
	# Plot all of the features
	fig, axarr = plt.subplots(10, 10, figsize=(10, 10))
	for i in range(10):
		for j in range(10):
			im = W[i*10+j,1:].reshape(28,28)
			axarr[i, j].imshow(im.T,cmap='gray')
			axarr[i,j].axis('off')
	fig.show()
	return fig, axarr

def main():
	savedata = False
	# Initialize the network
	epochs = 400
	nn = NN([100], [0.01, 0.01], 784, 784, 0.7)

	X, y = preprocessData()
	data = create_train_test_split(X, y, 0.8)
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
		for i, X in enumerate(x_train):
			X = X.reshape(len(X),1)
			X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term

			delW, y_hatq, f_s_aq = nn.fit(X, X_label, 0)
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
		save_data("3_2", nn.best_layers, data, nn.LR, nn.a)

if __name__ == "__main__":
    main()
