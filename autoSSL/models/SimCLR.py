import pytorch_lightning as pl
import torch
from torch import nn
from autoSSL.models.Backbone import pipe_backbone
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

class SimCLR(pl.LightningModule):
    def __init__(self, backbone="resnet18", stop_gradient=False, prjhead_dim=2048):
        super().__init__()

        self.backbone, self.out_dim = pipe_backbone(backbone)
        self.prjhead_dim = prjhead_dim
        if self.prjhead_dim:
            self.projection_head = SimCLRProjectionHead(self.out_dim, self.prjhead_dim, self.prjhead_dim)
        self.criterion = NTXentLoss()
        self.stop_gradient = stop_gradient

    def forward(self, x, stop_gradient=False):
        x = self.backbone(x).flatten(start_dim=1)
        if self.prjhead_dim:
            x = self.projection_head(x)
        if stop_gradient:
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1, self.stop_gradient)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
