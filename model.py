"""Models for facial keypoint detection"""

import torch
from torch._C import device
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models




class ResBlock(nn.Module):

    def __init__(self,n_channels):

        super().__init__()


        self.conv1=nn.Conv2d(n_channels,n_channels,3,1,1)
        self.bn1=nn.BatchNorm2d(n_channels)

        self.conv2=nn.Conv2d(n_channels,n_channels,3,1,1)
        self.bn2=nn.BatchNorm2d(n_channels)

        self.act=nn.LeakyReLU()

    def forward(self,x):

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.act(out)

        out=self.conv2(out)
        out=self.bn2(out)

        out+=x
        out=self.act(out)

        return out


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self,hparams,logger=None,train_set=None,val_set=None):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        #KeypointModel, self
        super().__init__()
        #self.hparams = hparams

        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
    
        self.hparams=hparams
        self.data={
            'train': train_set,
            'val': val_set
        }
        if logger is not None:
            self.logger=logger
        
        self.conv1=nn.Conv2d(1,3,1)
        self.wide_resnet=models.wide_resnet50_2(pretrained=True)
        for param in self.wide_resnet.parameters():
            param.requires_grad = False
        self.fc1=nn.Linear(1000,512)
        self.bn1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,256)
        self.bn2=nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,30)

        self.dp=nn.Dropout(0.3)
        self.act=nn.LeakyReLU()
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        out=self.conv1(x)
        out=self.wide_resnet(out)
        out=self.act(self.bn1(self.fc1(out)))
        out=self.dp(out)
        out=self.act(self.bn2(self.fc2(out)))
        out=self.dp(out)
        out=self.fc3(out)

        return out

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch["image"], batch["keypoints"]
    
    
        # forward pass
        out = self.forward(images).view(-1,15,2)
        targets=targets.view(-1,15,2)

        # loss
        loss = F.mse_loss(out, targets)

        return loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        logs={'train_loss':loss} 
        
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        loss= self.general_step(batch, batch_idx, "val")
        logs={'val_loss':loss}

        return {'loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('val_loss',avg_loss, self.current_epoch)

        return {'val_loss': avg_loss}

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('train_loss',avg_loss, self.current_epoch)

        return {'avg_loss_train': avg_loss}

    
    def configure_optimizers(self):
       
        optim=torch.optim.Adam(self.parameters(),self.hparams['learning_rate'])
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optim,self.hparams['lr_decay']),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': self.hparams['lr_freq']}


        return [optim], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    

