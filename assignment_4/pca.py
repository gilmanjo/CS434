from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
import numpy as np


# Constants
DATA_LOC = "./"
PLOT_LOC = "./plots/"


def main():
	
	data = load_data("data.txt")
	data = np.array(data)
	# Labels (every 1000 samples): 1, 3, 4, 7, 8, 9

	# scale data with 0 mean
	mu = data.mean(axis=0)
	data = np.subtract(data, mu)

	# fit using PCA
	covariance_matrix = np.cov(data.T)
	eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

	# sort values/vectors
	idx = np.argsort(eigenvalues)[::-1]
	eigenvectors = eigenvectors[:,idx]
	eigenvalues = eigenvalues[idx]
	print(eigenvalues[:10])
	eigenvectors = eigenvectors[:,:10]
	eigenvectors = eigenvectors.T

	# project data into 10 dimensions using eigenvectors
	new_data = np.dot(eigenvectors, data.T)

	# plot mean image
	data = np.add(data, mu)
	
	mean_sample = np.mean(data, axis=0)
	plt.imshow(np.reshape(mean_sample, (28,28)))
	plt.show()

	# plot top 10 eigenvectors
	
	for x in range(10):
		plt.subplot(2, 5, x + 1)
		plt.imshow(np.reshape(eigenvectors[x].real, (28,28)))

	plt.tight_layout()
	plt.show()

	# find samples with greatest values in each reduced dimension
	image_idxs = np.argmin(new_data, axis=1)

	# plot with corresponding eigenvector
	fig = plt.figure(figsize=(25,5))
	for x in range(10):
		plt.subplot(2, 10, x + 1)
		plt.imshow(np.reshape(data[image_idxs[x]], (28,28)))
		plt.subplot(2, 10, x + 11)
		plt.imshow(np.reshape(eigenvectors[x].real, (28,28)))

	plt.tight_layout()
	plt.show()

def load_data(fn):

	data = []
	with open(DATA_LOC + fn, "r") as f:

		for line in f:

			vector = [int(i) for i in line.split(",")]
			data.append(vector)

	return data

if __name__ == '__main__':
	main()