import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
 
from torch.utils.data import DataLoader
from tqdm import tqdm
def embed(x, embedding_model, device):
        embedding_model.eval()
        embedding_model.to(device)
        x = x.float().to(device)   # remove the unsqueeze operation
        return embedding_model(x).detach().cpu().numpy()

def eval_KNN(pipe_data, embedding_model, device='cuda', split=None, test=None):
    '''
    This function trains a KNN model and evaluates its accuracy on a test dataset.

    Parameters:
    pipe_data (PipeDataset): The whole data as a PipeDataset object
    embedding_model (torch.nn.Module): Model to create embeddings
    device (str): Device to perform computations on. Default is 'cuda' if available.
    split (float, optional): The ratio of samples to include in the train split.
    test (PipeDataset, optional): The test data as a PipeDataset object

    Returns:
    float: The accuracy of the model on the test data
    '''

    # Use split parameter to divide data into train and test if test is not provided
    if split is not None and test is None:
        train_data, test_data = pipe_data.split(split)
    else:
        train_data = pipe_data
        test_data = test

    # Extract features and labels from train and test data
    print("Load the training dataset to array")
    X_train, y_train = train_data.array[0], train_data.array[1]
    print("Load the testing dataset to array")
    X_test, y_test = test_data.array[0], test_data.array[1]

    # Get the embeddings for train and test data
    X_train_embedding = []
    print("embedding the training dataset")
    for x in tqdm(DataLoader(X_train, batch_size=128)):
        X_train_embedding.append(embed(x, embedding_model, device))
    X_train_embedding = np.concatenate(X_train_embedding)
    print("embedding the test dataset")
    X_test_embedding = []
    for x in tqdm(DataLoader(X_test, batch_size=128)):
        X_test_embedding.append(embed(x, embedding_model, device))
    X_test_embedding = np.concatenate(X_test_embedding)

    # Initialize the KNN Classifier
    knn = KNeighborsClassifier()
    print("Training in downstream")
    # Fit the model on the training data
    knn.fit(X_train_embedding, y_train)

    # Use the trained model to predict labels for the test data
    X_test_predicted = knn.predict(X_test_embedding)
    accuracy = accuracy_score(y_test, X_test_predicted)
    return accuracy
