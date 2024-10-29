""" Trains a DeepLabv3 model from a configuration file """

import os

import torch
import yaml

from models import DeepLabWrapper
from utils import get_dataloader, Trainer
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    with open('config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # create an output directory for the model if one doesn't exist
    os.makedirs('runs', exist_ok=True)

    # create dataloaders
    dataloaders = get_dataloader(config['DATA_PATH'],
                                 batch_size=config['BATCH_SIZE'],
                                 resize_shape=(config['IMG_HEIGHT'], config['IMG_WIDTH']))
    
    print('dataloaders', dataloaders)
    # create the model
    model = DeepLabWrapper(backbone=config['BACKBONE'], num_mask_channels=config['NUM_MASK_CHANNELS'])

    # train the model
    
    #class_weights = torch.tensor([0.25,0.5,0.1,0.15]).float()
    class_weights = torch.tensor([0.235,0.56,0.065,0.14]).float()

    # Move class weights to the appropriate device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    # Initialize the criterion with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    #criterion = torch.nn.CrossEntropyLoss(reduction='mean')   
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.89, 0.998), lr=1e-4, weight_decay=1e-4)

    trainer = Trainer(model, dataloaders, criterion, optimizer,
                      num_epochs=config['NUM_EPOCHS'],
                      is_inception=config['IS_INCEPTION'])
    trainer.train()

    # save the model
    model_path = config.get('SAVE_MODEL_PATH', f'models/{config["BACKBONE"]}_v1.{config["NUM_EPOCHS"]}.pth')
    model.save_model(model_path)
