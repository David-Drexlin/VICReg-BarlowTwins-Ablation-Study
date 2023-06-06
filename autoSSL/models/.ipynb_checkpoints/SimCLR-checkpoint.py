
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from autoSSL.models.Backbone import pipe_backbone

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(pl.LightningModule):
    def __init__(self, backbone="resnet18"):
        super().__init__()
 
        self.backbone, self.out_dim = pipe_backbone(backbone)
        self.projection_head = SimCLRProjectionHead(self.out_dim, 2048, 2048)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
