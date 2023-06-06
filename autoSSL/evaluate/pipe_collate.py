import os
import glob
import re
import pandas as pd
import torch
import yaml
from autoSSL.models import pipe_model

def pipe_collate(address, reg):
    # Define the directory to search
    search_dir = address
    
    # Use * as a wildcard to match all files starting with "batch_"
    pattern = os.path.join(search_dir, "batch_*")
    
    # Find matching directories
    matching_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    
    # Filter directories based on the regular expression
    regex = re.compile(reg)
    matching_dirs = [d for d in matching_dirs if regex.search(os.path.basename(d))]
    
    # Initialize lists to store column data
    dir_names = []
    ckpt_paths = []
    config_paths = []
    log_paths = []
    
    model_list = []  # list to store loaded models
    
    # Loop through all matching directories
    for dir_path in matching_dirs:
        dir_name = os.path.basename(dir_path)
        dir_names.append(dir_name)

        # Find the checkpoint file with maximum epoch
        checkpoint_files = glob.glob(os.path.join(dir_path, "*.ckpt"))
        max_epoch = -1
        ckpt_path = None
        for file in checkpoint_files:
            # extract epoch number from filename
            epoch_str = file.split('=')[1].split('-')[0]
            epoch = int(epoch_str)
            if epoch > max_epoch:
                max_epoch = epoch
                ckpt_path = file
        ckpt_paths.append(ckpt_path)

        # Generate the config file path and log file path
        config_path = os.path.join(dir_path, "config.yaml")
        config_paths.append(config_path)
        log_paths.append(os.path.join(dir_path, dir_name+".csv"))

        # Load model and add to model_list
        checkpoint = torch.load(ckpt_path)
        print("")
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            
        model = pipe_model(config=config) 
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model_list.append(model)
    
    # Create a pandas DataFrame
    df = pd.DataFrame({
        'dir_name': dir_names,
        'ckpt_path': ckpt_paths,
        'config_path': config_paths,
        'log_path': log_paths
    })

    # Save the DataFrame as a CSV file
    csv_path = os.path.join(search_dir, reg+".csv")
    print(f"Collating the models' (evaluating) information to {csv_path}")
    df.to_csv(csv_path, index=False)

    return {'name': dir_names, 'model': model_list, 'address': csv_path}
