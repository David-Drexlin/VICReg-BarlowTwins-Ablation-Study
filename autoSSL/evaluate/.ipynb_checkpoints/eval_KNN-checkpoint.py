from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

def eval_KNN(root_path, X, y, fname, samples):
    """
    This function plots the nearest neighbors for a given number of randomly chosen samples.

    Parameters:
    root_path (str): The root path of the image directory.
    X (np.ndarray): The array representing the test set features.
    y (np.ndarray): The array representing the test set labels.
    fname (list): List of image filenames.
    samples (int): The number of images to plot nearest neighbors for.

    Returns:
    None
    """

    random_indices = random.sample(range(X.shape[0]), samples)
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(X)

    for idx in random_indices:
        distances, indices = nbrs.kneighbors(X[idx].reshape(1, -1))
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('Nearest neighbors for: '+str(fname[idx][0]), fontsize=16)
        example_label = y[idx]
    
        # Iterate over the neighbors
        for i, index in enumerate(indices[0]):
            image_path = root_path + fname[index]  # Extract the filename from the tuple and prepend the root path
            img = mpimg.imread(image_path)
            label = y[index]
        
            # Choose label color based on match with example
            color = 'red' if label != example_label else 'black'
        
            # Plot image and label
            axs[i//3, i%3].imshow(img)
            axs[i//3, i%3].set_title('Label: '+str(label), color=color)
            axs[i//3, i%3].axis('off')

        plt.tight_layout()
        plt.show()


def eval_linear(X_embedding, y_train, X_test_embedding, y_test):
    '''
    This function trains a linear model using Stochastic Gradient Descent (SGD) 
    and evaluates its accuracy on a test dataset.

    Parameters:
    X_train_embedding (numpy.ndarray or pandas.DataFrame): Training data embedding
    y_train (numpy.ndarray or pandas.Series): Labels for the training data
    X_test (numpy.ndarray or pandas.DataFrame): Test data
    y_test (numpy.ndarray or pandas.Series): Labels for the test data

    Returns:
    float: The accuracy of the model on the test data
    '''

    # Initialize the SGDClassifier
    clf = SGDClassifier()

    # Fit the model on the training data
    clf.fit(X_embedding, y_train)

    # Use the trained model to predict labels for the test data
    X_test_predicted = clf.predict(X_test_embedding)

    # Calculate and return the accuracy of the model on the test data
    accuracy = accuracy_score(y_test, X_test_predicted)
    return accuracy

eval_linear(X_test_embedding, y_test, X_test_embedding, y_test) 