import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable


# Constants
TRAIN_DATA_BATCHES = [1, 2, 3, 4]
TEST_DATA_BATCHES = [5]
DATA_BATCH_NAME = "data_batch_"
DATA_BATCH_DIR = "./cifar-10-batches-py/"
LABEL_FILE_NAME = "batches.meta"
PLOT_SAVE_DIR = "./plots/"
BATCH_SIZE = 32
EPOCHS = 100


class Net2Sig(nn.Module):
	"""docstring for Net
	"""
	def __init__(self, dropout):
		super(Net2Sig, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 100)
		self.fc1_drop = nn.Dropout(dropout)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = nn.functional.sigmoid(self.fc1(x))
		x = self.fc1_drop(x)

		return nn.functional.log_softmax(self.fc2(x))

class Net3Sig(nn.Module):
	"""docstring for Net
	"""
	def __init__(self, dropout):
		super(Net3Sig, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 50)
		self.fc1_drop = nn.Dropout(dropout)
		self.fc2 = nn.Linear(50, 50)
		self.fc2_drop = nn.Dropout(dropout)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = nn.functional.sigmoid(self.fc1(x))
		x = self.fc1_drop(x)
		x = nn.functional.sigmoid(self.fc2(x))
		x = self.fc2_drop(x)

		return nn.functional.log_softmax(self.fc3(x))

class Net2Relu(nn.Module):
	"""docstring for Net
	"""
	def __init__(self, dropout):
		super(Net2Relu, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 100)
		self.fc1_drop = nn.Dropout(dropout)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = nn.functional.relu(self.fc1(x))
		x = self.fc1_drop(x)

		return nn.functional.log_softmax(self.fc2(x))

class Net3Relu(nn.Module):
	"""docstring for Net
	"""
	def __init__(self, dropout):
		super(Net3Relu, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 50)
		self.fc1_drop = nn.Dropout(dropout)
		self.fc2 = nn.Linear(50, 50)
		self.fc2_drop = nn.Dropout(dropout)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = nn.functional.relu(self.fc1(x))
		x = self.fc1_drop(x)
		x = nn.functional.relu(self.fc2(x))
		x = self.fc2_drop(x)

		return nn.functional.log_softmax(self.fc3(x))
		
def main():

	parser = argparse.ArgumentParser(
		description="Run a simple neural net on CIFAR-10.")
	parser.add_argument("--nn-type", default="2sig",
		choices=["2sig", "3sig", "2relu", "3relu"],
		help="the type of neural net to train on")
	parser.add_argument("--lr", type=float, default=0.01,
		help="the learning rate of training")
	parser.add_argument("--wd", type=float, default=0,
		help="the weight decay of the optimizer")
	parser.add_argument("--momentum", type=float, default=0,
		help="the momentum of the optimizer")
	parser.add_argument("--dropout", type=float, default=0.5,
		help="the dropout rate to use in the neural net")
	args = parser.parse_args()

	print("Loading CIFAR-10 dataset...")
	train_data, test_data, label_names = load_data()
	X_train = train_data["data"]
	y_train = train_data["labels"]
	X_test = test_data["data"]
	y_test = test_data["labels"]

	print("Normalizing training dataset...")
	X_train = normalize_data(X_train)

	print("Normalizing testing dataset...")
	X_test = normalize_data(X_test)

	print("Shuffling data...")
	X_train, y_train = shuffle(X_train, y_train, random_state=0)

	print("Training data shape: {}.\nTesting data shape: {}.".format(
		X_train.shape, X_test.shape))
	print("Creating tensors...")
	X_train = torch.stack([torch.Tensor(i) for i in X_train])
	y_train = [np.array(i).astype(np.int32) for i in y_train]
	y_train = torch.stack([torch.Tensor(i) for i in y_train])
	X_test = torch.stack([torch.Tensor(i) for i in X_test])
	y_test = [np.array(i).astype(np.int32) for i in y_test]
	y_test = torch.stack([torch.Tensor(i) for i in y_test])

	print("Generating DataLoader...")
	train_ds = torch.utils.data.TensorDataset(X_train, y_train)
	test_ds = torch.utils.data.TensorDataset(X_test, y_test)
	train_loader = torch.utils.data.DataLoader(train_ds,
		batch_size=BATCH_SIZE)
	test_loader = torch.utils.data.DataLoader(test_ds,
		batch_size=BATCH_SIZE)

	# select network architecture
	if args.nn_type == "2sig":
		model = Net2Sig(args.dropout)
	elif args.nn_type == "3sig":
		model = Net3Sig(args.dropout)
	elif args.nn_type == "2relu":
		model = Net2Relu(args.dropout)
	elif args.nn_type == "3relu":
		model = Net3Relu(args.dropout)

	if torch.cuda.is_available():
		model.cuda()

	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
		momentum=args.momentum, weight_decay=args.wd)

	print("Training neural net...")
	lossv, accv, t_loss, epoch_list = [], [], [], []
	for epoch in range(1, EPOCHS+1):
		model, optimizer, train_loader, loss = train(epoch, model, optimizer,
			train_loader)
		t_loss.append(loss)
		lossv, accv, model, test_loader = validate(lossv, accv, model,
			test_loader)
		epoch_list.append(epoch)

		# break if loss is not changing much
		avg_loss = sum(t_loss)/len(t_loss)
		if abs(avg_loss - loss) < 0.01*avg_loss and epoch > 10:
			break

	print("Plotting results...")
	fig = plt.figure(figsize=(10,6))
	ax = plt.subplot(2, 1, 1)
	plt.plot(epoch_list, t_loss)
	plt.xlabel("epochs")
	plt.ylabel("negative loglikelihood")
	plt.title("Training Loss")
	ax.text(0.05, 0.95, "Final loss: {}".format(t_loss[-1]),
		transform=ax.transAxes, fontsize=14, verticalalignment="top",
		bbox={"boxstyle":"round", "alpha":0.5})

	ax = plt.subplot(2, 1, 2)
	plt.plot(epoch_list, accv)
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.title("Validation Accuracy")
	ax.text(0.05, 0.95, "Final accuracy: {}%".format(accv[-1]),
		transform=ax.transAxes, fontsize=14, verticalalignment="top",
		bbox={"boxstyle":"round", "alpha":0.5})
	
	plt.tight_layout()
	plt.savefig("{}nn-type_{}_lr_{}_dropout_{}_momentum_{}_wd_{}.png".format(
		PLOT_SAVE_DIR, args.nn_type, args.lr, args.dropout, args.momentum,
		args.wd))
	plt.close(fig)

def load_data():
	"""Loads CIFAR-10 images into training and testing dictionaries

	Dictionaries have two elements: data and labels.  The data element is
	a numpy array of 10,000 32x32 color images in uint8S.  The labels element
	is a list of numbers corresponding to the labels for each of the images.

	Args:
		(none)

	Returns:
		training dictionary
		test dictionary
		list where each index defines a name of a label
	"""
	train_data = {"data":[], "labels":[]}
	test_data = {"data":[], "labels":[]}
	for x in TRAIN_DATA_BATCHES:

		with open(DATA_BATCH_DIR + DATA_BATCH_NAME + str(x), "rb") as f:

			temp_dict = pickle.load(f, encoding="bytes")

			if len(train_data["data"]) == 0:
				train_data["data"] = temp_dict[b"data"]
				train_data["labels"] = temp_dict[b"labels"]

			else:
				train_data["data"] = np.concatenate([train_data["data"], 
					temp_dict[b"data"]])
				train_data["labels"] = np.concatenate([train_data["labels"], 
					temp_dict[b"labels"]])

	for x in TEST_DATA_BATCHES:

		with open(DATA_BATCH_DIR + DATA_BATCH_NAME + str(x), "rb") as f:

			temp_dict = pickle.load(f, encoding="bytes")

			if len(test_data["data"]) == 0:
				test_data["data"] = temp_dict[b"data"]
				test_data["labels"] = temp_dict[b"labels"]

			else:
				test_data["data"] = np.concatenate([test_data["data"],
					temp_dict[b"data"]])
				test_data["labels"] = np.concatenate([test_data["labels"],
					temp_dict[b"labels"]])

	label_names = []
	with open(DATA_BATCH_DIR + LABEL_FILE_NAME, "rb") as f:

		temp_list = pickle.load(f, encoding="bytes")
		label_names += temp_list

	return train_data, test_data, label_names

def normalize_data(dataset, d_scalar=255.0):
	"""Normalizes a numpy array of data given a division scalar

	Args:
		dataset - a numpy array of the data to be normalized

	Returns:
		normalized dataset
	"""
	return np.divide(dataset, d_scalar)

def train(epoch, model, optimizer, train_loader, log_interval=100):
	"""
	"""
	model.train()
	final_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		if torch.cuda.is_available():
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = nn.functional.nll_loss(output, target.long())
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
		final_loss = loss.data[0]
	return model, optimizer, train_loader, final_loss

def validate(loss_vector, accuracy_vector, model, test_loader):
	"""
	"""
	model.eval()
	val_loss, correct = 0, 0
	for data, target in test_loader:
		if torch.cuda.is_available():
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		val_loss += nn.functional.nll_loss(output, target.long()).data[0]
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data.long()).cpu().sum()

	val_loss /= len(test_loader)
	loss_vector.append(val_loss)

	accuracy = 100. * correct / len(test_loader.dataset)
	accuracy_vector.append(accuracy)
	
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(test_loader.dataset), accuracy))
	return loss_vector, accuracy_vector, model, test_loader

if __name__ == '__main__':
	main()