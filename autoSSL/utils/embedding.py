import torch
# Embedding
def embedding_feature(X, embedding_model, device):
    """
    This function computes the embeddings of a given input using a pre-trained model.

    Parameters:
    X (numpy.ndarray): The input data for which the embeddings should be calculated. 
                       It should be a 2D numpy array where each row corresponds to an instance.

    embedding_model (torch.nn.Module): A PyTorch model which is used to compute the embeddings. 
                                       The model should have a forward method which takes in a 
                                       PyTorch tensor and outputs the corresponding embeddings.

    device (torch.device): The device (CPU or GPU) where the computations will be performed.

    Returns:
    numpy.ndarray: The embeddings of the input data. It will be a 2D numpy array where 
                   each row corresponds to the embeddings of an instance in the input.
    """
    
    # ensure the model is in eval mode
    embedding_model.eval()
    # Move the model to the appropriate device
    embedding_model.to(device)
    # Move input data to the appropriate device
    X = torch.from_numpy(X).float()
    X_device = X.to(device)
    
    # Compute embeddings, detach it from the computation graph and move it to cpu
    X_embedding = embedding_model(X_device).detach().cpu().numpy()
    
    return X_embedding.reshape(X_embedding.shape[0], -1)