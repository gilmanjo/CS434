# CS 434: Machine Learning & Data Mining

An upper-division course at Oregon State University in supervised and unsupervised learning.

There are 4 implementation assignments in the course.  Here's a brief overview of each of them!  More detailsto be added when I get around to polishing this repo up.

---

### Assignment 1: Linear Regression & Logistic Regression with Regularization

Comparing training and testing accuracy over iterations of batch gradient descent.  Performing linear regression on the Boston Housing dataset of housing prices in the Boston suburbs.  Logistic regression (as a classifier) on USPS handwritten digits 4 and 9.

Here's an example of the logistic regression at work:
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_1/p21_plot.png "Figure 1")

### Assignment 2: K-Nearest Neighbors & Decision Stumps/Trees

In this assignment I looked at diagnoses of breast cancer given a range of features provided by the Winsconsin Diagnostic Breast Cancer dataset.  Leave-One-Out cross-validation is implemented to perform model selection.  The result of LOO ended up being pretty minimal.

A decision tree is also implemented via scikit-learn.  Is there a single-feature that provides significant information to whether a tumor is malignant or benign?

The result of using different values of k in the algorithm:
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_2/fig1.png "Figure 2")

Experimenting with different decision tree depths to obtain optimal performance:
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_2/fig8.png "Figure 3")

### Assignment 3: Neural Nets

A multi-layer perceptron for classification of the CIFAR-10 dataset.  Nothing sophisticated, just a few two-layer and three-layer networks experimenting between sigmoid and ReLu activation.  The performance is not comparable anything top papers are getting, but it's a nice exercise in a fairly difficult classification problem.

Example training loss and validation accuracy with a 3-layer, ReLu-activated network with a learning rate of 0.01, 10% dropout, 0.1 momentum, and 0.001 weight decay.
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_3/plots/nn-type_3relu_lr_0.01_dropout_0.1_momentum_0.1_wd_0.001.png "Figure 4")

### Assignment 4: K-Means & Principal Component Analysis

Unsupervised learning assignment.  Dataset is 6000 greyscale images of handwritten images.  I wrote an implementation of the K-Means algorithm to test varying numbers clusters and how they affect the K-Means objective.  For each K-Means test, ten runs with random initialization are done and the best results are used.

Increasing the value of k increased performance significantly up to the maximum test value of 10.
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_4/plots/all_k.png "Figure 5")

Finally some PCA is done to analyze the geometry of the digits.  I use NumPy to pull out the ten most significant eigenvectors and reduce the dataset to just ten dimensions.  A comparison is done on the images with the highest weights for each of the ten dimensions.  Very interesting results!

The "mean" handwritten digit:
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_4/plots/mean_image.png "Figure 6")

The bottom rows shows our top eigenvectors, the top shows the sample image with the greatest value for the dimension corresponding to the eigenvector below it.  (yellow area):
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_4/plots/eigen_vs_image.png "Figure 7")

The bottom rows shows our top eigenvectors, the top shows the sample image with the lowest value for the dimension corresponding to the eigenvector below it.  (purple area):
![alt text](https://github.com/gilmanjo/CS434/blob/master/assignment_4/plots/eigen_vs_image_min.png "Figure 8")