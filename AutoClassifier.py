"""
AutoClassifier.py
using Python 3.5/6
Used for problem 1 in HW 4. It can be ran through a terminal using the command
"python3 AutoClassifier.py" or "python AutoClassifier.py" depending on how python
is setup. It will run as long as a folder called "cached_run" is created and the
pickled files "saved_data_3_1.p", "saved_data_3_2.p", and "saved_data_3_3.p"
are placed inside of that folder, and scripts "FeedForwardNN" and "Autoencoder"
are in the top directory.
"""
import numpy as np
import math
import random
import time
import pickle
import matplotlib.pyplot as plt
from FeedForwardNN import NN, save_data, eval_network, plot_errorrate
from Autoencoder import plot_features


def classification_network(fn_load, savedata, fn_save, retrain_output, params):
	# Perform the process of classification used in HW3 P1 with pretrained auto-encoders

	# Load previous data
	saved_data = pickle.load(open(fn_load, "rb"))
	data = saved_data["Data"]

	# Initialize the network
	epochs = params["epochs"]
	H = params["H"]
	f = params["f"]
	c = params["c"]
	LR = params["LR"]
	a = params["a"]

	nn = NN(H, [LR, LR], f, c, a)

	# Load previous weights into network
	nn.load_weights(saved_data["Weights"])

	if retrain_output:
		# Add new randomized classification output layer
		nn.pop_layer()
		nn.append_layer(c)
		freeze_lst = [0]  # Freeze hidden layer
		nn.freeze_layers(freeze_lst)
	nn.regularize = [False, False]  # Only perform LMS on network for training

	mini_batch_per = 0.2

	# Initialize sentinel structures
	x_shape = data["x_train"].shape
	last_time = time.time()
	hit_rates_train = []
	hit_rates_valid = []
	epoch_arr = []
	best_validation, best_epoch = 0, 0

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
			nn.update_weights(acc_delW, past_delW, 1)  # Apply accumulated weights at end of epoch.
		else:
			nn.update_weights(acc_delW, past_delW, 0)  # Apply accumulated weights at end of epoch.
			past_delW = acc_delW

		if (epoch % 10) == 0:
			# measure error on the validation set
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
	confusion_Matrix_training, train_correct = eval_network(nn, data, "train")
	confusion_Matrix_testing, test_correct = eval_network(nn, data, "test")
	print("params")
	print(H , LR, a)
	print("Training Accuracy: {}".format(train_correct/len(data["x_train"])))
	print(confusion_Matrix_training)
	print("Test Accuracy: {}".format(test_correct/len(data["x_test"])))
	print(confusion_Matrix_testing)

	# Plot error rate
	plot_errorrate(epoch_arr, hit_rates_train, hit_rates_valid)
	# Plot hidden layer features
	plot_features(nn.layers[0])
	plt.show()

	# Store the network, data, and parameters for future use.
	if savedata:
		save_data(fn_save, nn.best_layers, data, nn.LR, nn.a)

def main():
	# Using the network from HW3 P1 to classify network w/o further training.
	one = True
	if one:
		fn_load = "cached_run/saved_data_3_1.p"
		savedata = True
		fn_save = "3_4_c"
		params = {"epochs":0, "H":[100], "f":784, "c":10, "LR":0, "a":0.7}
		retrain_output = False
		classification_network(fn_load, savedata, fn_save, retrain_output, params)

	# Using the network from HW3 P2 to classify network. Output layer is randomized
	# and hidden layer is frozen
	two = True
	if two:
		fn_load = "cached_run/saved_data_3_2.p"
		savedata = True
		fn_save = "3_4_a"
		params = {"epochs":500, "H":[100], "f":784, "c":10, "LR":0.01, "a":0.5}
		retrain_output = True
		classification_network(fn_load, savedata, fn_save, retrain_output, params)

	# Using the network from HW4 P1 to classify network. Output layer is randomized
	# and hidden layer is frozen
	three = True
	if three:
		fn_load = "cached_run/saved_data_3_3.p"
		savedata = True
		fn_save = "3_4_b"
		params = {"epochs":500, "H":[100], "f":784, "c":10, "LR":0.05, "a":0.5}
		retrain_output = True
		classification_network(fn_load, savedata, fn_save, retrain_output, params)

if __name__ == "__main__":
    main()
