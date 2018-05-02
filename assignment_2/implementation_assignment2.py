import csv
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin/"
from sklearn import model_selection, neighbors, preprocessing, tree
import sys


# Constants
TRAIN_DATA_FILE = "./knn_train.csv"
TEST_DATA_FILE = "./knn_test.csv"


def main():
	
	print("Loading data sets...")
	train_data = open_data(TRAIN_DATA_FILE)
	test_data = open_data(TEST_DATA_FILE)

	# data normalizer
	min_max_scaler = preprocessing.MinMaxScaler()

	print("Normalizing data...")
	X_train = train_data[:,1:]
	X_train = min_max_scaler.fit_transform(X_train)
	y_train = train_data[:,0]

	X_test = test_data[:,1:]
	X_test = min_max_scaler.fit_transform(X_test)
	y_test = test_data[:,0]

	# K-NN (Problem 1)

	# array of K values for multiple K-Nearest Neighbors
	K_list = [x for x in range(1, 3, 2)]

	# create and train classifiers for each value of K
	print("Creating classifier for K-Nearest Neighbors...")
	clf_list = []
	for k in K_list:

		clf = neighbors.KNeighborsClassifier(k)
		clf.fit(X_train, y_train)
		clf_list.append(clf)

	# get our model selection metrics: training err, test err,
	# and LOO cross-validation error
	print("Calculating model selection metrics...")
	tr_errors = []
	te_errors = []
	loo_errors = []
	loo = model_selection.LeaveOneOut()
	for idx, clf in enumerate(clf_list):

		print("\t clf {}".format(idx))

		# training error
		tr_errors.append((1 - clf.score(X_train, y_train))*len(y_train))

		# testing error
		te_errors.append((1 - clf.score(X_test, y_test))*len(y_test))

		# Leave-One-Out cross validation error
		loo_scores = []
		for train_idx, test_idx in loo.split(X_train):

			X_loo_train, X_loo_test = X_train[train_idx], X_train[test_idx]
			y_loo_train, y_loo_test = y_train[train_idx], y_train[test_idx]

			loo_scores.append(clf.score(X_loo_train, y_loo_train))

		loo_errors.append((1 - (sum(loo_scores)/len(loo_scores)))*len(y_loo_train))

	# sum all errors
	sum_errors = []
	for x in range(len(K_list)):
		sum_errors.append(tr_errors[x] + te_errors[x] + loo_errors[x])

	"""plt.plot(K_list, tr_errors, "r*")
	plt.plot(K_list, te_errors, "g^")
	plt.plot(K_list, loo_errors, "b--")
	plt.plot(K_list, sum_errors, "black")
	plt.ylabel("Number of errors")
	plt.xlabel("K")
	plt.legend(["Training Errors", "Testing Errors", "LOO Errors", "Sum of Errors"])
	plt.show()"""

	# Decision Trees (Problem 2, 3)

	tr_acc_list = []
	te_acc_list = []

	for d in range(1,7):

		clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=d)
		clf.fit(X_train, y_train)

		# calculate train and test errors
		tr_acc = clf.score(X_train, y_train)
		te_acc = clf.score(X_test, y_test)

		tr_acc_list.append(tr_acc)
		te_acc_list.append(te_acc)

		# graph decision stump
		data = tree.export_graphviz(clf, out_file=None)
		graph = graphviz.Source(data, format="png")
		graph.render("ds_{}".format(d))

	"""plt.plot([1,2,3,4,5,6], tr_acc_list, "r--")
	plt.plot([1,2,3,4,5,6], te_acc_list, "b--")
	plt.ylabel("Accuracy")
	plt.xlabel("Depth, d")
	plt.legend(["Training Accuracy", "Testing Accuracy"])
	plt.show()"""

def open_data(fn):
	return np.genfromtxt(fn, delimiter=",")

if __name__ == '__main__':
	main()