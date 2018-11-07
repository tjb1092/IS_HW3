import numpy as np
from data_load import preprocessData, create_train_test_split
import math
import random
import time
import pickle
import matplotlib.pyplot as plt
from FeedForwardNN import NN, save_data

def main():
	epochs = 1000
	nn = NN(1, [100], [0.02, 0.02], 784, 784, 0.75, 0.25, 0.8)

	X, y = preprocessData()
	data = create_train_test_split(X,y, 0.8)
	mini_batch_per = 0.1
	x_shape = data["x_train"].shape
	last_time = time.time()
	error_train = []
	error_valid = []
	epoch_arr = []
	best_validation = 1e6
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

			delW, y_hatq = nn.fit(X, X_label, 0)
			acc_error += (1./2.)*np.sum(np.square(np.subtract(y_hatq,X_label)))  # Error formula
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
			# measure hit rate on the validation set
			valid_error = 0
			for i, X in enumerate(data["x_validate"]):
				X = X.reshape(len(X),1)
				X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term
				y = data["y_validate"][i].reshape(10,1)
				delW, y_hatq = nn.fit(X, X_label, 1)
				valid_error += (1./2.)*np.sum(np.square(np.subtract(y_hatq,X_label)))

			error_valid.append(valid_error)
			print("Epoch: {}".format(epoch))
			error_train.append(acc_error)
			epoch_arr.append(epoch)

			print("Training Error: {}".format(acc_error))
			print("Validation Error: {}".format(valid_error))
			if valid_error < best_validation:
				# If validation error improves,
				best_validation = valid_error
				print("Validation Improved!")
				nn.best_layers = nn.layers  # Store best weights
				best_epoch = epoch
			print("Best Epoch: {}".format(best_epoch))
			print('Epoch took {:0.3f} seconds'.format(time.time()-last_time))
			last_time = time.time()

	# Evaulate Test Accuracy
	nn.layers = nn.best_layers  # Roll-back to best weights
	train_error=0
	for i, X in enumerate(data["x_train"]):
		X = X.reshape(len(X),1)
		X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term
		delW, y_hatq = nn.fit(X,X_label, 1)
		train_error += (1./2.)*np.sum(np.square(np.subtract(y_hatq,X_label)))

	test_error = 0
	for i, X in enumerate(data["x_test"]):
		X = X.reshape(len(X),1)
		X_label = X[1:,0].reshape(len(X)-1,1)  # Remove bias term
		delW, y_hatq = nn.fit(X,X_label, False)
		test_error += (1./2.)*np.sum(np.square(np.subtract(y_hatq,X_label)))
		#plt.imshow(y_hatq.reshape(28,28).T)
		#plt.show()

	print("Training Error (J): {}".format(train_error))
	print("Test Error (J): {}".format(test_error))

	# Plot error rate per epoch
	fig, ax = plt.subplots(figsize=(6,6))
	ax.plot(epoch_arr, error_train, label="Training Error")
	ax.plot(epoch_arr, error_valid, label="Validation Error")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Error (J)")
	ax.set_title("Error per Epoch")
	ax.legend()
	fig.show()


	# Plot all of the images
	fig2, axarr = plt.subplots(10, 10, figsize=(10, 10))
	W = nn.best_layers[0]
	for i in range(10):
		for j in range(10):
			im = W[i*10+j,1:].reshape(28,28)
			axarr[i, j].imshow(im.T,cmap='gray')

	plt.show()
	save_data("3_2", nn.best_layers, data, nn.LR, nn.a)

if __name__ == "__main__":
    main()
