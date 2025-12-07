from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
from torch.nn import functional as F
from pytorch_lightning.loggers import WandbLogger
from models.common import LinearProbeMultiClass
from pytorch_lightning.callbacks import Callback
from train_simclr import SimCLR
from models import SimCLR
import wandb
from PIL import Image
import torchvision.transforms as T
import pandas as pd




class Evaluator(pl.LightningModule):
    """
    Evaluates a self-supervised learning backbone using the standard evaluation protocol of a linear probe.

    Example::

        # pretrained model
        backbone = SimCLR.load_from_checkpoint(PATH, strict=False)

        # dataset + transforms
        dm = ImageFolderDataModule(data_dir='.')

        # finetuner
        evaluator = Evaluator(backbone, in_features=512)

        # train
        trainer = pl.Trainer()
        trainer.fit(evaluator, dm)

        # test
        trainer.test(datamodule=dm)
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 512,
        max_epochs: int = 100,
        log_table = None,
        dropout: float = 0.,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.0,
        finetune: bool = False, # it should be False!        
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
        """
        
        """
        the input image of 64X64 is reduced by the encoder to 512, thats why in_features = 512
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.final_lr = final_lr
        self.finetune = finetune

        self.backbone = backbone
        self.log_table = log_table # wandB log table
        self.linear_probe = LinearProbeMultiClass(input_dim=in_features, dropout=dropout)  
        self.loss_fc = torch.nn.CrossEntropyLoss()    
        
        """
        finetune will train the backbone and the linear probe together
        requires_grad is used to freeze or unfreeze the backbone model
        
        """
        
        # Determine whether to finetune the weights in backbone.
        # backbone.parameters() gets all the hyper parameters of the backbone.
        for param in self.backbone.parameters():
            param.requires_grad = self.finetune 

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)
       
    
    def load_pretrained(self,url):
        return self.load_from_checkpoint(url, strict=False)
        
    # this is where the weights are frozen before the very first epoch.
    def on_train_epoch_start(self) -> None:
        if self.finetune:
            self.backbone.train()
        else:
            self.backbone.eval()


    def training_step(self, batch, batch_idx):
        loss, probs, y = self.shared_step(batch)
        self.train_acc(probs, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self.shared_step(batch)
        self.val_acc(probs, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc,prog_bar=True) 

        return loss

    
    def test_step(self, batch, batch_idx):
        '''
        The Accuracy() function takes into account the predicted probabilities 
        for each class and selects the class with the highest probability as the 
        predicted class label. It then compares the predicted labels with the true
        labels to compute the accuracy.
        '''


        loss, probs, y = self.shared_step(batch) # this loss is the mean of each mini-batch   

        self.test_acc(probs, y)
        print(self.test_acc)
        self.log('test_loss', loss, sync_dist=True,prog_bar=True)
        self.log('test_acc', self.test_acc,prog_bar=True)
        return loss
    

    
    def shared_step(self, batch):
        x, y = batch # x- img as as np array or tensors, y-labels in tensors (0,1)
        feats = self.backbone(x) # shape: [128,512]
        feats = feats.view(feats.size(0), -1) # does not change the shape or size of the tensor, checked
        logits = self.linear_probe(feats).squeeze() # shape: [128, 10] becz 10 output classes

        # softmax is internally applied in torch' CrossEntropyLoss()
        copied_logits = logits.clone()  # Create a copy of logits

        probs = torch.nn.functional.softmax(copied_logits, dim=1)  # dim=1 is the label dim, shape: [128,10]
        loss = self.loss_fc(logits, y)
        return loss, probs, y

    def configure_optimizers(self):
        # Train entire network if finetune is True.
        if self.finetune:
            params = self.parameters()
        else:
            params = self.linear_probe.parameters()

        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.max_epochs,
                eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]