import numpy as np

with open("./housing_train.txt", "r") as f:
	matrix_train = np.array([line.split() for line in f]).astype(float)

with open("./housing_test.txt", "r") as f:
	matrix_test = np.array([line.split() for line in f]).astype(float)

# get X_train and Y_train from file
X_train = np.delete(matrix_train, 13, 1)
Y_train = np.delete(matrix_train, [0,1,2,3,4,5,6,7,8,9,10,11,12], 1)

# get X_test and Y_test from file
X_test = np.delete(matrix_test, 13, 1)
Y_test = np.delete(matrix_test, [0,1,2,3,4,5,6,7,8,9,10,11,12], 1)

# append column of 1s to X
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)

# compute optimum weight vector
w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.matmul(np.transpose(X_train), Y_train))

# compute sum of squared error (SSE)
sse_train = 0
sse_test = 0

for idx, house in enumerate(X_train):
	prediction = np.dot(house, w)
	error = (Y_train[idx] - prediction)**2
	sse_train += error

for idx, house in enumerate(X_test):
	prediction = np.dot(house, w)
	error = (Y_test[idx] - prediction)**2
	sse_test += error

# mean squared error (MSE)
mse_train = sse_train / X_train.shape[0]
mse_test = sse_test / X_test.shape[0]

print("Training Results:\nSSE: {}\tMSE: {}".format(float(sse_train), float(mse_train)))
print("Testing Results:\nSSE: {}\tMSE: {}".format(float(sse_test), float(mse_test)))