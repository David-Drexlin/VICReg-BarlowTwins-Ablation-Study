import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimSiamTransform
from autoSSL.models.Backbone import pipe_backbone

class SimSiam(pl.LightningModule):
    def __init__(self, backbone="resnet18", stop_gradient=False, prjhead_dim=512):
        super().__init__()
        self.backbone, self.out_dim = pipe_backbone(backbone)
        self.prjhead_dim = prjhead_dim
        if self.prjhead_dim:
            self.projection_head = SimSiamProjectionHead(self.out_dim, self.prjhead_dim, 128)
            self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        else:
            self.prediction_head = SimSiamPredictionHead(self.out_dim, 64, 128)    
        self.criterion = NegativeCosineSimilarity()
        self.stop_gradient = stop_gradient

    def forward(self, x, stop_gradient=False):
        f = self.backbone(x).flatten(start_dim=1)
        if self.prjhead_dim:
            f = self.projection_head(f)
        p = self.prediction_head(f)
        if stop_gradient:
            f = f.detach()
        return f, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
