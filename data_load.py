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
    label_array = []
    with open(fn) as f:
        contents = f.readlines()
    # for each line
    for label in contents:
        label = label[:-1]  #remove new-line character
        label_array.append(int(label))  # Convert to int

    return label_array


def create_train_test_split(x, y, percent):
	# Create Train/Test Split
	d_len = len(x)
	# Sample 80% of the indices
	train_index = random.sample(range(d_len - 1), math.ceil(percent*d_len))
	# Compute the remaining 20% of samples for the test set
	test_index = list(set(range(d_len-1)) - set(train_index))

	# Index out the train set from the total set
	x_train = x[train_index]
	y_train = y[train_index]
	# Index the test set
	x_test = x[test_index]
	y_test = y[test_index]

	# create data dict to pass up to other functions
	return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def preprocessData():
    # Read in data
    Xfn = "MNISTnumImages5000.txt"
    yfn = "MNISTnumLabels5000.txt"

    # Pass data array into functions to append everything together
    X = data_load(Xfn)
    y = label_load(yfn)

    X = np.asarray(X)  # Convert to numpy array
    y = np.asarray(y).reshape(len(y), 1)  # Reshape to make a 5000x1 matrix

    # Data already normalized.
    return X, y

def main():
	preprocessData()

if __name__ == "__main__":
	main()
