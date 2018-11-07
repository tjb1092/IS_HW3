import numpy as np
from data_load import preprocessData, create_train_test_split
import math
import random
import time
import pickle
import matplotlib.pyplot as plt
class NN:
	def __init__(self, n, H, LR, f, c, UpperTarget, LowerTarget, a):
		self.n = n
		self.f = f
		self.c = c
		self.H = H
		self.LR = LR
		self.UT = UpperTarget
		self.LT = LowerTarget
		self.a = a
		# Initialize weights
		layers = []
		for L in range(self.n+1):
			# Get dims for weight matrix w/ dims i,j
			if L == 0:
				i = self.H[L]
				j = self.f + 1
				sigma = math.sqrt(2/(f+1))
			elif L == n:
				i = c
				j = self.H[L-1]+1
				sigma = math.sqrt(2/(H[L-1]+1))
			else:
				i = self.H[L]
				j = self.H[L-1]+1
				sigma = math.sqrt(2/(H[L-1]+1))

			w_ij = np.random.normal(0,sigma,(i,j))
			print("Layer {}".format(L))
			print(w_ij)
			print(w_ij.shape)
			layers.append(w_ij)

		self.layers = layers
		self.best_layers = layers

	def fit(self, X_q, y_q, tr_te):
		#Forward and backward prop on one data point. Return error matricies
		# Assume sigmoid activation functions for now.

		#forward pass
		f_s_aq=[]
		for L in range(self.n):
			w_bar = self.layers[L]
			# Set the bias to 1, will override rest of values.
			h_q = np.ones((w_bar.shape[0]+1,1))
			if L==0:
				dot_p = np.dot(w_bar,X_q)
			else:
				dot_p = np.dot(w_bar,f_s_aq[L-1])

			h_q[1:] = 1./(1.+np.exp(-dot_p))  # sigmoid activation
			f_s_aq.append(h_q)

		dot_p = np.dot(self.layers[self.n],f_s_aq[self.n-1])
		y_hatq = 1./(1+np.exp(-dot_p))
		f_s_aq.append(y_hatq)

		# Initialize using list comprehension
		delq = [[] for k in range(self.n+1)]
		delW = [[] for k in range(self.n+1)]
		if tr_te == 0:
			# Backward pass: Compute output layer's errors
			delq[-1] = (1-y_hatq)*y_hatq*(y_q - y_hatq)
			delW[-1] = self.LR[self.n] * np.dot(delq[-1], f_s_aq[self.n-1].T)
			for L in range(self.n-1, -1, -1):
			    # Next layer:
			    # h[L] ^* W*delq[L+1]
			    delq[L] = ((1-f_s_aq[L])*f_s_aq[L])[1:]*np.dot(self.layers[L+1][:,1:].T,delq[L+1])
			    if L==0:
			        inputQ = X_q.T
			    else:
			        inputQ = f_s_aq[L-1].T

			    delW[L] = self.LR[L]*np.dot(delq[L],inputQ)

			# Apply weight update after data point has been applied
			for i, W in enumerate(self.layers):
			    self.layers[i] = W + delW[i]

		return delW, y_hatq

	def update_weights(self, delW, past_delW, mode):

		for i, W in enumerate(self.layers):
			if mode == 1:
				self.layers[i] = W + delW[i]  # First epoch doens't have prev. values
			else:
				self.layers[i] = W + delW[i] + self.a * past_delW[i]

def save_data(problem, weights,data,LR,alpha):
	save_info = {"Weights": weights, "Data": data, "LR": LR, "alpha":alpha}
	pickle.dump(save_info, open("cached_run/saved_data_{}.p".format(problem),"wb"))

def main():
	epochs = 700
	nn = NN(1, [200], [0.02, 0.02], 784, 10, 0.75, 0.25, 0.6)

	X, y = preprocessData()
	data = create_train_test_split(X,y, 0.8)
	mini_batch_per = 0.1
	x_shape = data["x_train"].shape
	last_time = time.time()
	hit_rates_train = []
	hit_rates_valid = []
	epoch_arr = []
	best_validation = 0
	best_epoch = 0
	for epoch in range(epochs):

		# Pick mini-back of training data
		# Sample 20% of the indices
		batch_index = random.sample(range(x_shape[0] - 1), math.ceil(mini_batch_per*x_shape[0]))
		x_train = data["x_train"][batch_index]
		y_train = data["y_train"][batch_index]
		correct = 0
		for i, X in enumerate(x_train):
			X = X.reshape(len(X),1)
			y = y_train[i].reshape(10,1)
			delW, y_hatq = nn.fit(X,y, 0)
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
				delW, y_hatq = nn.fit(X, y, 1)
				if np.argmax(y_hatq) == np.argmax(y):
					valid_correct += 1

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
	train_correct=0
	confusion_Matrix_training = np.zeros((10,10))  # Initialize counts
	for i, X in enumerate(data["x_train"]):
		X = X.reshape(len(X),1)
		y = data["y_train"][i].reshape(10,1)
		delW, y_hatq = nn.fit(X,y, 1)
		actual_class = np.argmax(y)
		assigned_class = np.argmax(y_hatq)
		confusion_Matrix_training[assigned_class, actual_class] += 1
		if assigned_class == actual_class:
			train_correct += 1

	# This hurts me as a programmer
	test_correct = 0
	confusion_Matrix_testing = np.zeros((10,10))  # Initialize counts
	for i, X in enumerate(data["x_test"]):
		X = X.reshape(len(X),1)
		y = data["y_test"][i].reshape(10,1)
		delW, y_hatq = nn.fit(X,y, False)
		actual_class = np.argmax(y)
		assigned_class = np.argmax(y_hatq)
		confusion_Matrix_testing[assigned_class, actual_class] += 1
		if assigned_class == actual_class:
			test_correct += 1

	print("Training Accuracy: {}".format(train_correct/4000.))
	print(confusion_Matrix_training)
	print("Test Accuracy: {}".format(test_correct/1000.))
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

	save_data("3_1", nn.best_layers, data, nn.LR, nn.a)

if __name__ == "__main__":
    main()
