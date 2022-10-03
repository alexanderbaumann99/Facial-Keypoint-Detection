import os
import torch
from torchvision import transforms
from data.facial_keypoints_dataset import FacialKeypointsDataset
from model import KeypointModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping



if __name__=="__main__":

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_url = 'https://vision.in.tum.de/webshare/g/i2dl/facial_keypoints.zip'
    data_root = os.path.join(os.getcwd(), "datasets")

    train_dataset = FacialKeypointsDataset(
        train=True,
        transform=transforms.ToTensor(),
        root=data_root,
        download_url=download_url
    )
    val_dataset = FacialKeypointsDataset(
        train=False,
        transform=transforms.ToTensor(),
        root=data_root,
    )
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))

    hparam={'learning_rate':1e-4,
            'batch_size': 32,
            'lr_decay': 1,
            'lr_freq': 2
    }

    logger = TensorBoardLogger(save_dir='lightning_logs')
    model=KeypointModel(hparam,train_dataset,val_dataset)
    early_stop=EarlyStopping(monitor='train_loss',patience=6)

    trainer=pl.Trainer(max_epochs=30,
                       gpus=-1,
                       callbacks=[early_stop],
                       logger=logger
                       )

    trainer.fit(model,train_dataset,val_dataset)

    trainer.save_checkpoint("saved_model/facial_model.ckpt")
    
 

