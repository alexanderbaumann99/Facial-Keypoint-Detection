import matplotlib.pyplot as plt
import torch
from torch._C import device
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def show_all_keypoints(image, keypoints, pred_kpts=None):
    """Show image with predicted keypoints"""
    image = (image.clone() * 255).view(96, 96)
    plt.imshow(image, cmap='gray')
    keypoints = keypoints.clone() * 48 + 48
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=200, marker='.', c='m')
    if pred_kpts is not None:
        pred_kpts = pred_kpts.clone() * 48 + 48
        plt.scatter(pred_kpts[:, 0], pred_kpts[:, 1], s=200, marker='.', c='r')
    plt.show()



def show_pred(dataset,model,num_samples=3):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  
    for i,batch in enumerate(dataloader):
        if i<=num_samples:
            print(i)
            image, _ = batch["image"], batch["keypoints"]
            pred = model(image).view(15,2)
            image=image.cpu()
            pred=pred.cpu().detach()
            key_pts = dataset[i]["keypoints"]
            print('PREDICTION')
            show_all_keypoints(image,pred)
            print('GROUND TRUTH')
            show_all_keypoints(image, key_pts)



def show_keypoints(dataset, num_samples=3):

    
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    for i in range(num_samples):
        image = dataset[i]["image"]
        key_pts = dataset[i]["keypoints"]
        show_all_keypoints(image, key_pts)


def evaluate_model(model, dataset):
    
    criterion = torch.nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0
    for batch in dataloader:
        image, keypoints = batch["image"], batch["keypoints"]
        predicted_keypoints = model(image).view(-1,15,2)
        loss += criterion(
            torch.squeeze(keypoints),
            torch.squeeze(predicted_keypoints)
        ).item()
    return 1.0 / (2 * (loss/len(dataloader)))
