"""
data_load.py
using Python 3.5/6
Helper script for loading the data in HW 3. Not ran directly.
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

def data_load(fn):
	# Open text file and create and read in each line
	X_array = []
	with open(fn) as f:
		contents = f.readlines()
	# for each line
	for data in contents:
		parsedData = data.split("\t")  # Split via space deliminator
		parsedData[-1] = parsedData[-1][:-1]  #remove new-line character
		# Create data tuple by casting text to floats and add label
		row = np.zeros((784))
		for i, feature in enumerate(parsedData):
			row[i] = float(feature)
		X_array.append(row)

	return X_array

def label_load(fn):
	# Load labels as 1-hot array
	label_array = []
	with open(fn) as f:
		contents = f.readlines()
	# for each line
	for label in contents:
		l = np.array((0,0,0,0,0,0,0,0,0,0))
		label = label[:-1]  #remove new-line character
		l[int(label)] = 1  # Index to create one-hot array
		label_array.append(l)  # Convert to int

	return label_array

def create_train_test_split(x, y, percent):
	# Create Train/Test Split
	d_len = len(x)
	# Sample 80% of the indices
	train_index = random.sample(range(d_len - 1), math.ceil(percent*d_len))
	# Compute the remaining 20% of samples for the test set
	test_index = list(set(range(d_len)) - set(train_index))

	# Index out the train set from the total set
	x_train = x[train_index]
	y_train = y[train_index]
	# Index the test set
	x_test = x[test_index]
	y_test = y[test_index]
	# Repeat process to get the validation set.
	valid_percent = 0.125
	validation_index = random.sample(range(len(x_train) - 1), math.ceil(valid_percent*len(x_train)))
	# Compute the remaining 20% of samples for the test set
	train_index = list(set(range(len(x_train))) - set(validation_index))
	# Index the validation set.
	x_validate = x_train[validation_index]
	y_validate = y_train[validation_index]

	# Reindex the training set to remove the validation set.
	x_train = x[train_index]
	y_train = y[train_index]

	# create data dict to pass up to other functions
	return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "x_validate": x_validate, "y_validate": y_validate}

def preprocessData():
	# Set the filenames for the data.
	Xfn = "MNISTnumImages5000.txt"
	yfn = "MNISTnumLabels5000.txt"

	# Load the data Nand label files
	X = data_load(Xfn)
	y = label_load(yfn)
	ones = np.ones((len(X), 1))

	X = np.asarray(X)  # Convert to numpy array
	X = np.concatenate((ones, X), axis=1)  # Add bias term
	y = np.asarray(y)  # Convert to numpy array
	# Data already normalized.
	return X, y

if __name__ == "__main__":
	main()
