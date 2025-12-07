from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy

from models.common import LinearProbeMultiClass


class Intermediate(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        layer: str,
        in_features: int,
        max_epochs: int,
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

        super().__init__()
        self.layer = layer
        self._features = None
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
        self._register_hook()
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

    def _register_hook(self):
        def hook_fn(module, input, output):
            if output.dim() > 2:
                # remove cls token
                output = output[:, 1:, :].mean(dim=1)
            self._features = output

        # Traverse the layer path string and register the hook
        target_layer = self.backbone
        for part in self.layer.split('.'):
            target_layer = getattr(target_layer, part)
        self._handle = target_layer.register_forward_hook(hook_fn)

    def on_fit_end(self):
        # Clean up hook
        self._handle.remove()

    def on_fit_start(self):
        if hasattr(self, '_handle'):
            self._handle.remove()
        self._register_hook()
        
    # this is where the weights are frozen before the very first epoch.
    def on_train_epoch_start(self) -> None:
        if self.finetune:
            self.backbone.train()
        else:
            self.backbone.eval()

    def forward(self, x):
        _ = self.backbone(x)  # triggers the hook
        return self._features


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
        feats = self(x) # shape: [128,512] CALLS FORWARD FUNCTION!!
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