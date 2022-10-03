import os
from pytorch_lightning.core.hooks import ModelHooks
import torch
from torchvision import transforms
from data.facial_keypoints_dataset import FacialKeypointsDataset
from model import KeypointModel
from utils import show_pred,evaluate_model


if __name__=="__main__":

    data_root = os.path.join(os.getcwd(), "datasets")
    val_dataset = FacialKeypointsDataset(
        train=False,
        transform=transforms.ToTensor(),
        root=data_root,
    )

       
    model=KeypointModel.load_from_checkpoint(checkpoint_path="saved_model/facial_model.ckpt")
    model.eval()
    print('Evaluation Score',evaluate_model(model,val_dataset))
    show_pred(val_dataset,model,2)

 

