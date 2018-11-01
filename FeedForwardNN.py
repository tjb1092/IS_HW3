import numpy as np
from data_load import preprocessData, create_train_test_split
import math
import random
import time
class NN:
	def __init__(self, n, H, LR, f, c, UpperTarget, LowerTarget):
		self.n = n
		self.f = f
		self.c = c
		self.H = H
		self.LR = LR
		self.UT = UpperTarget
		self.LT = LowerTarget
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

	def fit(self, X_q,y_q, tr_te):
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
		#print("predictions")
		#print(y_hatq)
		#print("Forward Pass Complete")
		if tr_te == 0:
			# Backward pass: Compute output layer's errors
			# Initialize using list comprehension
			delq = [[] for k in range(self.n+1)]
			delW = [[] for k in range(self.n+1)]
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

			return delW

	def update_weights(self, delW):
		for i, W in enumerate(self.layers):
			self.layers[i] = W + delW[i]


def main():
	epochs = 1000
	nn = NN(1, [200], [0.01, 0.01], 784, 10)

	X, y = preprocessData()
	data = create_train_test_split(X,y, 0.8)
	mini_batch_per = 0.1
	x_shape = data["x_train"].shape
	last_time = time.time()
	for epoch in range(epochs):
		if (epoch % 10) == 0:
			print(epoch)
			print('loop took {:0.3f} seconds'.format(time.time()-last_time))
			last_time = time.time()
		# Pick mini-back of training data
		# Sample 20% of the indices
		batch_index = random.sample(range(x_shape[0] - 1), math.ceil(mini_batch_per*x_shape[0]))
		x_train = data["x_train"][batch_index]
		y_train = data["y_train"][batch_index]

		for i, X in enumerate(x_train):
			X = X.reshape(len(X),1)
			y = y_train[i].reshape(10,1)
			delW = nn.fit(X,y, 0)

			if i == 0:
				acc_delW = delW
			else:
				# Accumulate weight change for all training points per epoch
				for j, dW in enumerate(delW):
					acc_delW[j] += dW

		nn.update_weights(acc_delW)  # Apply accumulated weights at end of epoch.




if __name__ == "__main__":
    main()
