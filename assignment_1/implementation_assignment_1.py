import numpy as np
import csv


#Linear regression part
with open("./housing_train.txt", "r") as f:
	matrix_train = np.array([line.split() for line in f]).astype(float)

with open("./housing_test.txt", "r") as f:
	matrix_test = np.array([line.split() for line in f]).astype(float)

# get X_train and Y_train from file
X_train = np.delete(matrix_train, 13, 1)
Y_train = np.delete(matrix_train, np.s_[0:13], 1)

# get X_test and Y_test from file
X_test = np.delete(matrix_test, 13, 1)
Y_test = np.delete(matrix_test, np.s_[0:13], 1)

# append column of 1s to X
X_train_dummy = np.insert(X_train, 0, 1, axis=1)
X_test_dummy = np.insert(X_test, 0, 1, axis=1)

#generate d random features and insert them
row_num_train = X_train.shape[0]
row_num_test = X_test.shape[0]

mu, sigma = 0, 0.1 	#mean and standard deviation
from copy import deepcopy
X_train_mod_d = deepcopy(X_train)
X_test_mod_d = deepcopy(X_test)

for d in range(0, 100):
	feature_train = np.random.normal(mu, sigma, row_num_train)
	feature_test = np.random.normal(mu, sigma, row_num_test)
	X_train_mod_d = np.insert(X_train_mod_d, 0, feature_train, axis=1)
	X_test_mod_d = np.insert(X_test_mod_d, 0, feature_test, axis=1)



# compute optimum weight vector
w_dummy = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train_dummy), X_train_dummy)), np.matmul(np.transpose(X_train_dummy), Y_train))
w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.matmul(np.transpose(X_train), Y_train))
w_mod_d = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train_mod_d), X_train_mod_d)), np.matmul(np.transpose(X_train_mod_d), Y_train))

# compute sum of squared error (SSE)
sse_train_dummy = 0
sse_test_dummy = 0
sse_train = 0
sse_test = 0
sse_train_mod_d = [()]
sse_test_mod_d = [()]

for idx, house in enumerate(X_train_dummy):
	prediction = np.dot(house, w_dummy)
	error = (Y_train[idx] - prediction)**2
	sse_train_dummy += error

for idx, house in enumerate(X_test_dummy):
	prediction = np.dot(house, w_dummy)
	error = (Y_test[idx] - prediction)**2
	sse_test_dummy += error

for idx, house in enumerate(X_train):
	prediction = np.dot(house, w)
	error = (Y_train[idx] - prediction) ** 2
	sse_train += error

for idx, house in enumerate(X_test):
	prediction = np.dot(house, w)
	error = (Y_test[idx] - prediction) ** 2
	sse_test += error

for d in range(0, 100):
	for idx, house in enumerate(X_train_mod_d):
		prediction = np.dot(house, w_mod_d)
		error = (Y_train[idx] - prediction) ** 2
		sse_train_mod_d += (d, error)

	for idx, house in enumerate(X_test_mod_d):
		prediction = np.dot(house, w_mod_d)
		error = (Y_test[idx] - prediction) ** 2
		sse_test_mod_d += (d, error)

# mean squared error (MSE)
mse_train_dummy = sse_train_dummy / X_train_dummy.shape[0]
mse_test_dummy = sse_test_dummy / X_test_dummy.shape[0]
mse_train = sse_train / X_train.shape[0]
mse_test = sse_test / X_test.shape[0]

for d in range(0, 100):
	mse_train_mod_d += (d, sse_train_mod_d / X_train_mod_d.shape[0])
	mse_test_mod_d += (d, sse_test_mod_d / X_test_mod_d.shape[0])

print("Training Results w/ dummy var:\nSSE: {}\tMSE: {}".format(float(sse_train_dummy), float(mse_train_dummy)))
print("Testing Results w/ dummy var:\nSSE: {}\tMSE: {}".format(float(sse_test_dummy), float(mse_test_dummy)))
print("Training Results:\nSSE: {}\tMSE: {}".format(float(sse_train), float(mse_train)))
print("Testing Results:\nSSE: {}\tMSE: {}".format(float(sse_test), float(mse_test)))
print("Training Results w/ {} additional features:\nSSE: {}\tMSE: {}".format(d, float(sse_train_mod_d), float(mse_train_mod_d)))
print("Testing Results w/ {} additional features:\nSSE: {}\tMSE: {}".format(d, float(sse_test_mod_d), float(mse_test_mod_d)))



#Logistic regression part

##	for row in test_data:

