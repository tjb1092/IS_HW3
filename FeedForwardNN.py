"""
FeedForwardNN.py
using Python 3.5/6
Used for problem 1 in HW 3. It can be ran through a terminal using the command
"python3 Autoencoder.py" or "python Autoencoder.py" depending on how python is setup.
It will run as long as a folder "cached_run" is created and the
script "data_load.py" is in the same directory.
"""
import numpy as np
from data_load import preprocessData, create_train_test_split
import math
import random
import time
import pickle
import matplotlib.pyplot as plt

class NN:
	def __init__(self, H, LR, f, c, a=0.7, B=1, p=0.05, l=0.0001):
		# Store parameters into class.
		self.H = H
		self.n = len(self.H)
		self.LR = LR
		self.f = f
		self.c = c
		self.a = a
		self.B = B
		self.p = p
		self.l = l
		# Initialize weights
		layers = []
		masks = []
		regularize = []  # Keep track of what to implement regularizations on
		for L in range(self.n+1):
			# Get dims and sigma for weight matrix w/ dims i,j
			if L == 0:
				i = self.H[L]
				j = self.f + 1
				sigma = math.sqrt(2/(f+1))
			elif L == self.n:
				i = c
				j = self.H[L-1]+1
				sigma = math.sqrt(2/(H[L-1]+1))
			else:
				i = self.H[L]
				j = self.H[L-1]+1
				sigma = math.sqrt(2/(H[L-1]+1))
			# Initialize the weight matrix as a Gaussian distribution
			# proportional to the size of the hidden layers
			w_ij = np.random.normal(0,sigma,(i,j))
			# Append into layers strucuture that holds all weight matricies.
			layers.append(w_ij)
			masks.append(True)  # Make layer trainable
			regularize.append(True)  # Apply regularization to layertrainable
		# Define class params for the weight matricies and a "best" weight matrix.
		self.layers = layers
		self.best_layers = layers
		self.masks = masks
		self.regularize = regularize

	def fit(self, X_q, y_q, tr_te, p_hat=[]):
		# Forward and backward prop on one data point. Return error matricies
		# and weight change matricies.
		# Assume sigmoid activation functions.

		#Forward Pass
		f_s_aq=[]
		for L in range(self.n):
			w_bar = self.layers[L]
			# Set the bias to 1, will override rest of values.
			h_q = np.ones((w_bar.shape[0]+1,1))
			# Compute the summation via dot product
			if L==0:
				dot_p = np.dot(w_bar,X_q)
			else:
				dot_p = np.dot(w_bar,f_s_aq[L-1])

			h_q[1:] = 1./(1.+np.exp(-dot_p))  # sigmoid activation
			f_s_aq.append(h_q)  # Store hidden layer results for later.

		# Compute the last layer's summation
		dot_p = np.dot(self.layers[self.n],f_s_aq[self.n-1])
		# Compute last layer's output
		y_hatq = 1./(1+np.exp(-dot_p))
		f_s_aq.append(y_hatq)

		# Initialize using list comprehension
		delq = [[] for k in range(self.n+1)]
		delW = [[] for k in range(self.n+1)]
		# Skip this if just doing testing.
		if tr_te == 0:
			# Backward pass: Compute output layer's errors
			# Compute last layer's error using sigmoid function
			if self.masks[-1]:
				train = 1
			else:
				train = 0  # If layer is frozen, don't modify the weights.
			# Work assuming that layer is frozen before training starts.
			# Else, momentum will still change weights a bit.

			delq[-1] = (1-y_hatq)*y_hatq*(y_q - y_hatq) * train
			# Compute last layer's weight changes
			if  self.regularize[-1]:
				weight_decay = self.l * self.layers[-1]
				delW[-1] = self.LR[self.n] * (np.dot(delq[-1], f_s_aq[self.n-1].T)+weight_decay)
			else:
				delW[-1] = self.LR[self.n] * np.dot(delq[-1], f_s_aq[self.n-1].T)

			for L in range(self.n-1, -1, -1):
				# Next layer:
				# Error: h[L]*W*delq[L+1]
				if self.regularize[L]:
					sparse_pen = self.B*((1-self.p)/(1-p_hat[L]) - (self.p/p_hat[L]))

					delq[L] = ((1-f_s_aq[L])*f_s_aq[L])[1:]*(np.dot(self.layers[L+1][:,1:].T,delq[L+1])-sparse_pen)
				else:
					delq[L] = ((1-f_s_aq[L])*f_s_aq[L])[1:]*np.dot(self.layers[L+1][:,1:].T,delq[L+1])

				# pick either the input or one of the saved layer's outputs
				if L==0:
					inputQ = X_q.T
				else:
					inputQ = f_s_aq[L-1].T

				# Compute the layer's weight change matrix.
				if self.regularize[L]:
					weight_decay = self.l * self.layers[L]
					delW[L] = self.LR[L]*(np.dot(delq[L],inputQ) + weight_decay)
				else:
					delW[L] = self.LR[L]*np.dot(delq[L],inputQ)
				#print(delW)


			# Apply weight update after data point has been applied
			for i, W in enumerate(self.layers):
				self.layers[i] = W + delW[i]
		return delW, y_hatq, f_s_aq

	def update_weights(self, delW, past_delW, mode):
		# Apply the delW matrix to the layer weights
		for i, W in enumerate(self.layers):
			# Loop through the layer struct. For each layer, apply the weight update.
			if mode == 1:
				self.layers[i] = W + delW[i]  # First epoch doens't have prev. values
			else:
				self.layers[i] = W + delW[i] + self.a * past_delW[i]

	def load_weights(self, layers):
		# Override weight matricies with loaded weights.
		self.layers = layers
		self.best_layers = layers

	def append_layer(self, layer_len, index):
		i = self.c
		self.n += 1
		L = self.n
		j = self.H[L-1]+1
		sigma = math.sqrt(2/(H[L-1]+1))
		# Initialize the weight matrix as a Gaussian distribution
		# proportional to the size of the hidden layers
		w_ij = np.random.normal(0,sigma,(i,j))
		# Append into layers strucuture that holds all weight matricies.
		self.layers.append(w_ij)

	def pop_layer(self):
		self.layers = self.layers[:-1]
		self.best_layers = self.best_layers[:-1]
		self.n -= 1

	def freeze_layers(self, lst):
		for i in lst:
			self.mask[i] = False


def save_data(problem, weights,data,LR,alpha):
	# Pickle the data to save for later for HW 4.
	save_info = {"Weights": weights, "Data": data, "LR": LR, "alpha":alpha}
	pickle.dump(save_info, open("cached_run/saved_data_{}.p".format(problem),"wb"))
	print("Data Saved!")

def eval_network(nn, data, mode):
	correct=0
	confusion_Matrix = np.zeros((10,10))  # Initialize counts
	for i, X in enumerate(data["x_{}".format(mode)]):
		X = X.reshape(len(X),1)
		y = data["y_{}".format(mode)][i].reshape(10,1)
		delW, y_hatq, f_s_aq = nn.fit(X,y, 1)
		actual_class = np.argmax(y)
		assigned_class = np.argmax(y_hatq)
		confusion_Matrix[assigned_class, actual_class] += 1
		if assigned_class == actual_class:
			correct += 1

	return confusion_Matrix, correct

def main():
	# Parameter definitions and intializations.
	h_l = [100]
	LRs = [0.05]
	alphas = [0.7]
	epoch_lst = [500]
	mini_batch_per = 0.1

	best_validation, best_epoch = 0, 0
	savedata = False
	# Perform gridsearch over potential hyperparameters
	for h in h_l:
		for LR in LRs:
			for a in alphas:
				for epochs in epoch_lst:
					nn = NN([h], [LR, LR], 784, 10, a)

					hit_rates_train = []
					hit_rates_valid = []
					epoch_arr = []

					# Read in data.
					X, y = preprocessData()
					# Create the train-test split structure.
					data = create_train_test_split(X,y, 0.8)
					x_shape = data["x_train"].shape

					last_time = time.time()  # Keep track of epoch duration.
					for epoch in range(epochs):

						# Pick mini-batch of training data
						# Sample 20% of the indices
						batch_index = random.sample(range(x_shape[0] - 1), math.ceil(mini_batch_per*x_shape[0]))
						x_train = data["x_train"][batch_index]
						y_train = data["y_train"][batch_index]

						correct = 0
						for i, X in enumerate(x_train):
							# Setup data
							X = X.reshape(len(X),1)
							y = y_train[i].reshape(10,1)
							# Perform forward and backward passes.
							delW, y_hatq, f_s_aq = nn.fit(X,y, 0)
							# Determine if the correct choice was made.
							if np.argmax(y_hatq) == np.argmax(y):
								correct += 1
							if i == 0:
								acc_delW = delW
							else:
								# Accumulate weight change for all training points per epoch
								for j, dW in enumerate(delW):
									acc_delW[j] += dW

						if epoch == 0:
							past_delW = acc_delW
							nn.update_weights(acc_delW,past_delW, 1)  # Apply accumulated weights at end of epoch.
						else:
							nn.update_weights(acc_delW,past_delW, 0)  # Apply accumulated weights at end of epoch.
							past_delW = acc_delW

						if (epoch % 10) == 0:
							# measure hit rate on the validation set
							valid_correct = 0
							for i, X in enumerate(data["x_validate"]):
								X = X.reshape(len(X),1)
								y = data["y_validate"][i].reshape(10,1)
								delW, y_hatq, f_s_aq = nn.fit(X, y, 1)
								if np.argmax(y_hatq) == np.argmax(y):
									valid_correct += 1
							# Compute validation hit rate.
							valid_hit_rate = valid_correct/len(data["x_validate"])
							hit_rates_valid.append(valid_hit_rate)

							print("Epoch: {}".format(epoch))
							hit_rate = correct / (mini_batch_per*x_shape[0])
							hit_rates_train.append(hit_rate)
							epoch_arr.append(epoch)

							print("Training Hit Rate: {}".format(hit_rate))
							print("Validation Hit Rate: {}".format(valid_hit_rate))
							if valid_hit_rate > best_validation:
								# If validation hit rate improves,
								best_validation = valid_hit_rate
								print("Validation Improved!")
								nn.best_layers = nn.layers  # Store best weights
								best_epoch = epoch
							print("Best Epoch: {}".format(best_epoch))
							print('Epoch took {:0.3f} seconds'.format(time.time()-last_time))
							last_time = time.time()

					# Evaulate Test Accuracy
					nn.layers = nn.best_layers  # Roll-back to best weights

					# Compute the confusion matricies for train and test sets.
					confusion_Matrix_training, train_correct = eval_network(nn, data, "train")
					confusion_Matrix_testing, test_correct = eval_network(nn, data, "test")
					print("params")
					print(h, LR, a)
					print("Training Accuracy: {}".format(train_correct/len(data["x_train"])))
					print(confusion_Matrix_training)
					print("Test Accuracy: {}".format(test_correct/len(data["x_test"])))
					print(confusion_Matrix_testing)

					# Plot error rate per epoch
					fig, ax = plt.subplots(figsize=(6,6))
					error_rates_train = [1-x for x in hit_rates_train]
					error_rates_valid = [1-x for x in hit_rates_valid]
					ax.plot(epoch_arr, error_rates_train, label="Training Error Rate")
					ax.plot(epoch_arr, error_rates_valid, label="Validation Error Rate")
					ax.set_xlabel("Epoch")
					ax.set_ylabel("Error Rate")
					ax.set_title("Error Rates per Epoch")
					ax.legend()
					plt.show()
	# Save the data for future use in HW4.
	if savedata:
		save_data("3_1", nn.best_layers, data, nn.LR, nn.a)

if __name__ == "__main__":
    main()
