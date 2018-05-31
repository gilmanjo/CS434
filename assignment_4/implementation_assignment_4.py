import matplotlib.pyplot as plt
import numpy as np
import random
import sys


# Constants
DATA_LOC = "./"
PLOT_LOC = "./plots/"
MAX_ITER = 20

def main():
	
	data = load_data("data.txt")
	# Labels (every 1000 samples): 1, 3, 4, 7, 8, 9

	for i in range(2, 11):
		for j in range(10):
			kmeans(data, i, plot=True)
	"""results = []
	for i in range(2,11):
		results.append(kmeans(data, i))

	fig = plt.figure(figsize=(10,6))
	ax = plt.subplot(1, 1, 1)
	plt.plot(np.arange(2, 11), results)
	plt.xlabel("k")
	plt.ylabel("final sum of squared error (SSE)")
	plt.title("K-Means Clustering Survey")
	plt.savefig("{}k_survey.png".format(PLOT_LOC))
	plt.close(fig)"""

def load_data(fn):

	data = []
	with open(DATA_LOC + fn, "r") as f:

		for line in f:

			vector = [int(i) for i in line.split(",")]
			data.append(vector)


	return data

def kmeans(data, k, plot=False):

	print("Performing K-Means for k = {}".format(k))

	# initialize random centroids
	centroids = []
	for i in range(k):
		centroids.append(np.random.random_integers(255, size=(len(data[0]))))

	# init array to bookkeep change in centroid location
	deltas = [1 for x in range(k)]

	# init array to keep track of labels
	labels = [-1 for x in range(len(data))]

	# init k clusters
	clusters = [[] for x in range(k)]
	
	# run k-means until the centroids stop shifting or limit reached
	sse_history = []
	i = 0
	while i < MAX_ITER:

		sse = 0

		# check euclidean distance between each sample and centroid
		for s_idx, sample in enumerate(data):

			min_distance = np.inf

			for c_idx, centroid in enumerate(centroids):

				distance = np.linalg.norm(centroid - sample)

				# if this the lowest distance, assign sample to cluster
				if distance < min_distance:

					min_distance = distance
					labels[s_idx] = c_idx

			# add sample to corresponding cluster
			clusters[labels[s_idx]].append(sample)

			# add to sse
			sse += np.linalg.norm(centroids[labels[s_idx]] - sample)**2

		# generate new centroids from avg of assigned samples
		for j in range(k):

			new_centroid = np.sum(clusters[j], axis=0) / len(clusters[j])
			new_delta = np.linalg.norm(new_centroid - centroids[j])
			centroids[j] = new_centroid
			deltas[j] = new_delta

		sse_history.append(sse)
		i += 1
		print("Iteration: {}\tSSE: {}".format(i, sse))

	if plot:
		fig = plt.figure(figsize=(10,6))
		ax = plt.subplot(1, 1, 1)
		plt.plot(np.arange(i), sse_history)
		plt.xlabel("iterations")
		plt.ylabel("sum of squared error (SSE)")
		plt.title("K-Means Clustering (k = {})".format(k))
		ax.text(0.05, 0.95, "Final SSE: {}".format(sse),
			transform=ax.transAxes, fontsize=14, verticalalignment="top",
			bbox={"boxstyle":"round", "alpha":0.5})
		
		plt.tight_layout()
		plt.savefig("{}k_{}_sse_{}.png".format(PLOT_LOC, k, sse))
		plt.close(fig)

	return sse

if __name__ == '__main__':
	main()