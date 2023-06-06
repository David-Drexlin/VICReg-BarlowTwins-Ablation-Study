import os
import glob
import yaml
import pytorch_lightning as pl
from autoSSL.utils import ck_callback, join_dir,ContinuousCSVLogger

class Trainer(pl.Trainer):
    def __init__(self, config, model_mode, extra_epoch=0):
        self.config = config
        self.model_mode = model_mode
        self.extra_epoch = extra_epoch

        # Define the path for the config and checkpoint
        self.config["log_dir"] = join_dir(config["checkpoint_dir"], config["experiment"], config["name"])

        # Create directory if not exists
        os.makedirs(self.config["log_dir"], exist_ok=True)

        # Initialize or load config depending on the mode
        if self.model_mode == "start":
            self.initialize_config()
        elif self.model_mode == "continue":
            self.load_config()

        super().__init__(
            max_epochs=self.config["max_epochs"],
            accelerator=self.config["device"],
            callbacks=[ck_callback(self.config["log_dir"])],
            logger=ContinuousCSVLogger(save_dir=self.config["log_dir"], name=self.config["name"]),
        )

    def initialize_config(self):
        # Save the config
        with open(join_dir(self.config["log_dir"], "config.yaml"), 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def load_config(self):
        # Load the config
        with open(join_dir(self.config["log_dir"], "config.yaml"), 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        # Update max_epochs and save updated config
        print(f"Previous max epoch: {self.config['max_epochs']}. Continue to train for extra {self.extra_epoch} epochs.")
        self.config["max_epochs"] += self.extra_epoch
        with open(join_dir(self.config["log_dir"], "config.yaml"), 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def fit(self, model, dataloader, ckpt_path=None):
        if self.model_mode == "continue" and ckpt_path == "latest":
            # Get the most recent checkpoint file based on the epoch number in the filename
            checkpoint_files = glob.glob(join_dir(self.config["log_dir"], "*.ckpt"))
            max_epoch = -1
            for file in checkpoint_files:
                # extract epoch number from filename
                epoch_str = file.split('=')[1].split('-')[0]
                epoch = int(epoch_str)
                if epoch > max_epoch:
                    max_epoch = epoch
                    ckpt_path = file
            print(f"Loading checkpoint from: {ckpt_path}")

        super().fit(model, dataloader, ckpt_path=ckpt_path)
