import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def embed(x, embedding_model, device):
    embedding_model.eval()
    embedding_model.to(device)
    x = x.float().to(device)   # remove the unsqueeze operation
    return embedding_model(x).detach().cpu().numpy()


def eval_linear(pipe_data, models, device='cuda', split=None, test=None):
    '''
    Parameters:
    pipe_data (PipeDataset): The whole data as a PipeDataset object
    models (dict/torch.nn.Module): Model dict returned by `pipe_collate` or a single model to create embeddings
    device (str): Device to perform computations on. Default is 'cuda' if available.
    split (float, optional): The ratio of samples to include in the train split.
    test (PipeDataset, optional): The test data as a PipeDataset object
    '''

    # Use split parameter to divide data into train and test if test is not provided
    if split is not None and test is None:
        train_data, test_data = pipe_data.split(split)
    else:
        train_data = pipe_data
        test_data = test

    # Extract features and labels from train and test data
    print("Load the training and testing dataset")
    X_train, y_train = train_data.array[0], train_data.array[1]
    X_test, y_test = test_data.array[0], test_data.array[1]

    if isinstance(models, torch.nn.Module):
        models = {'name': ['model_0'], 'model': [models], 'address': None}

    # Initialize the results list
    results = []

    # Iterate over models and calculate accuracy
    for i, embedding_model in enumerate(tqdm(models['model'])):
        # Get the embeddings for train and test data
        X_train_embedding = [embed(x, embedding_model, device) for x in DataLoader(X_train, batch_size=128)]
        X_train_embedding = np.concatenate(X_train_embedding)

        X_test_embedding = [embed(x, embedding_model, device) for x in DataLoader(X_test, batch_size=128)]
        X_test_embedding = np.concatenate(X_test_embedding)

        # Initialize the SGDClassifier and train it
        clf = SGDClassifier()
        clf.fit(X_train_embedding, y_train)

        # Predict labels for the test data and calculate accuracy
        X_test_predicted = clf.predict(X_test_embedding)
        accuracy = accuracy_score(y_test, X_test_predicted)
        namee=models["name"][i]
        # Append the accuracy to the results
        results.append((namee, accuracy))
        
    if models['address'] is not None:
        df = pd.read_csv(models['address'])
        df['linear_accuracy'] = [result[1] for result in results]
        df.to_csv(models['address'], index=False)
    
    return results