from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

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