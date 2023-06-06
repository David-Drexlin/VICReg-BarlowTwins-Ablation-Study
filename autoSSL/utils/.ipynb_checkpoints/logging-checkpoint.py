from pytorch_lightning.loggers import CSVLogger
import os
import csv

class ContinuousCSVLogger(CSVLogger):

    def __init__(self, save_dir, name="default"):
        super().__init__(save_dir, name, version=None)
  
    def log_metrics(self, metrics, step=None):
        filepath = os.path.join(self.save_dir, f"{self.name}.csv")

        if not os.path.isfile(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()

        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writerow(metrics)


from pytorch_lightning.callbacks import ModelCheckpoint


def ck_callback(checkpoint_dir):
    checkpoint_callback = ModelCheckpoint(
       monitor='train_loss',
       dirpath=checkpoint_dir,
       filename='checkpoints-{epoch:02d}-{train_loss:.2f}',
       save_top_k=3,
       mode='min',)
    return checkpoint_callback