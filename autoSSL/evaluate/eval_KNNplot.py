from sklearn.neighbors import NearestNeighbors
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np  
 
def embed(x, embedding_model, device):
        embedding_model.eval()
        embedding_model.to(device)
        x = x.float().to(device)   # remove the unsqueeze operation
        return embedding_model(x).detach().cpu().numpy()

 
 
def eval_KNNplot(pipe_data_test, embedding_model, samples=2, device='cuda'):
    """
    This function plots the nearest neighbors for a given number of randomly chosen samples.

    Parameters:
    pipe_data_test (PipeDataset): The test data as a PipeDataset object
    embedding_model (torch.nn.Module): Model to create embeddings
    device (str): Device to perform computations on. Default is 'cuda' if available.
    samples (int): The number of images to plot nearest neighbors for.

    Returns:
    None
    """

    root_path=pipe_data_test.dir
    # Extract features and labels from test data
    print("Load the testing dataset to array")
    X_test, y_test, fname = pipe_data_test.array[0], pipe_data_test.array[1], pipe_data_test.array[2] 

    # Get the embeddings for test data
    print("embedding the test dataset") 
    X_test_embedding = []
    for x in tqdm(DataLoader(X_test, batch_size=128)):
        X_test_embedding.append(embed(x, embedding_model, device))
    X_test_embedding = np.concatenate(X_test_embedding)

    random_indices = random.sample(range(X_test_embedding.shape[0]), samples)
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(X_test_embedding)

    for idx in random_indices:
        distances, indices = nbrs.kneighbors(X_test_embedding[idx].reshape(1, -1))
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        print(fname[idx])
        fig.suptitle('Nearest neighbors for: '+str(fname[idx][0]), fontsize=16)
        example_label = y_test[idx]
    
        # Iterate over the neighbors
        for i, index in enumerate(indices[0]):
            image_path = root_path+fname[index]  # Extract the filename from the tuple
            img = mpimg.imread(image_path)
            label = y_test[index]
        
            # Choose label color based on match with example
            color = 'red' if label != example_label else 'black'
        
            # Plot image and label
            axs[i//3, i%3].imshow(img)
            axs[i//3, i%3].set_title('Label: '+str(label), color=color)
            axs[i//3, i%3].axis('off')

        plt.tight_layout()
        plt.show()
