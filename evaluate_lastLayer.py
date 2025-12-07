from argparse import ArgumentParser
from ast import arg
from platform import architecture
import ast
import pytorch_lightning as pl
import torch
import os
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf warnings set to silent in terminal

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path
sys.path.append(parent_dir)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import resnet18, resnet34
from models import archs
from models.archs import resnets, resnet_3b
from models.archs.resnet_3b import resnet_3blocks
from models.archs.resnet_2b import resnet_2blocks
from models.archs.resnet_1b import resnet_1block
import collections.abc as container_abcs
from models import SimCLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
import csv
import wandb
import collections
import pandas as pd
from models.vit_contrastive import VisionTransformer, Backbone, LitClassifier, ViTConfigExtended 
import transformers
from datamodules.dataHandler import OrientationRecognition
from models.evaluator import Evaluator


# wandb table to log results with UI
columns = ["IMAGE", "ACTUAL_LABEL", "PREDICTED_LABEL", "PROBABILITY", "CONFIDENCE", "LOSS", "PATH", "VIEWPOINT"]
log_table = wandb.Table(columns)


def cli_main():
    parser = create_argparser()
    args = parser.parse_args()
   

    validation(args)


def validation(args):


    # Load data
    dm = OrientationRecognition(
            data_dir=args.data_dir,
            batch_size=128,
            pin_memory=False,
            num_workers=16,
            val_split=args.val_split,
            drop_last=False,
            shuffle=args.shuffle, # initial - True
            train_dir=args.train_dir,
            img_res=args.img_res,
        )
    
    print("dataloader loaded successfully")
    
    print("shuffle is - ", dm.shuffle)
    

    model = init_model(args)
    
    if args.model == 'vit' or args.model == "untrained_vit" or args.model == 'simclr' or args.model == 'ae' or args.model == 'untrained_r18_2b':
        feature_dim = 512
    else:
        feature_dim = get_model_output_size(model, dm.dims)
    
    # dm.prepare_data()
    # dm.setup()
    evaluator = Evaluator(model, in_features=feature_dim, max_epochs=args.max_epochs, log_table=log_table)
    
    #print(evaluator)
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')

    callbacks = [model_checkpoint]

    # create a new logger  for wandB
    logger = WandbLogger(save_dir=f"/data/lpandey/LOGS/eval/{args.model}", name=args.exp_name, project=f"{args.project_name}", log_model="all")

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks
    )

    trainer.fit(evaluator, datamodule=dm)

    # log average test acc accross all the classes
    if args.test_indi_class is False:
        print("Logging Average Test Acc accross all the classes!")
        trainer.test(datamodule=dm)
    # log test acc for each class separately
    else:
        
        print("Testing Individual Class and loggin each Test Score separately!")

        num_classes = args.num_classes # [0,10,20,...,90]
        for i in range(num_classes):
            # initialize test dataset
            dm.setup(stage='test')
            dataset_test = dm.dataset_test
            # store samples from each subdir or class
            subset_samples = []
            subset_class_names = i
            # Filter the samples to create a new subset with the desired classes or folders
            subset_samples = [sample for sample in dataset_test.samples if sample[1] == subset_class_names]
            # update datamodule samples with subset samples
            dm.dataset_test.samples = subset_samples
            # test model
            test_score = trainer.test(datamodule=dm)
            # Extract the 'test_acc' value from the dictionary
            test_acc = test_score[0]['test_acc']
            # Log the test accuracy
            wandb.log({'Test Accuracy': test_acc})
 

def init_model(args):
    if args.model == 'pixels':
        model = nn.Flatten()
    elif args.model == 'simclr':
        model = SimCLR.load_from_checkpoint(args.model_path)    
    elif args.model == 'ae':
        model = AE.load_from_checkpoint(args.model_path).encoder
    elif args.model == 'supervised':
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r18':
        model = resnet18(pretrained=False)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r34':
        model = resnet34(pretrained=False)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r18_3b':
        model = resnet_3blocks(pretrained=False)
        model.fc = nn.Identity()
        print("Model selected - untrained resnet 18 : 3 blocks")
    elif args.model == 'untrained_r18_2b':
        model = resnet_2blocks(pretrained=False)
        model.fc = nn.Identity()
        print("Model selected - untrained resnet 18 : 2 blocks")
    elif args.model == 'untrained_r18_1b':
        model = resnet_1block(pretrained=False)
        model.fc = nn.Identity()
        print("Model selected - untrained resnet 18 : 1 block")

    elif args.model == 'vit':
        model = LitClassifier.load_from_checkpoint(args.model_path).backbone
        model.fc = nn.Identity()

    elif args.model == 'untrained_vit':
        configuration = ViTConfigExtended()
        configuration.num_hidden_layers = 3
        configuration.num_attention_heads = 3
        # print configuration parameters of ViT
        print('image_size - ', configuration.image_size)
        print('patch_size - ', configuration.patch_size)
        print('num_classes - ', configuration.num_classes)
        print('hidden_size - ', configuration.hidden_size)
        print('intermediate_size - ', configuration.intermediate_size)
        print('num_hidden_layers - ', configuration.num_hidden_layers)
        print('num_attention_heads - ', configuration.num_attention_heads)
        
        # pass the configuration parameters to get backbone
        backbone = Backbone('vit', configuration)
        model = LitClassifier(backbone).backbone
        model.fc = nn.Identity()
    return model



def get_model_output_size(model, input_size) -> int:
    """ Returns the output activation size of the encoder. """
    with torch.no_grad():
        if isinstance(input_size, int):
            x = model(torch.zeros(1, input_size))
        else:
            x = model(torch.zeros(1, *input_size))
        return x.view(1, -1).size(1)



def create_argparser():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="directory containing dataset")
    parser.add_argument("--test_indi_class", type=bool, default=False, help='Log Test Accuracy for individual classes if True')
    parser.add_argument("--num_classes", type=int, default=10, help="num of classes or subdirs in the test set")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--model", type=str, choices=['pixels', 'supervised', 'simclr', 'untrained_r18', 'untrained_r34', 'untrained_r18_3b', 'untrained_r18_2b', 'untrained_r18_1b', 'untrained_vit', 'ae', 'byol', 'vae', 'barlowTwins', 'vit', 'sit', 'cpc', 'individual_vit_heads', 'videomae'])
    parser.add_argument("--model_path", type=str, help="stored model checkpoint")
    parser.add_argument("--max_epochs", default=100, type=int, help="Max number of epochs to train.")
    parser.add_argument("--project_name", type=str, help="project_name") # for wandb dashboard and logging
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle images for training") # for wandb dashboard and logging
    parser.add_argument("--val_split", type=float, default=0.1, help="validation samples split ratio")
    parser.add_argument("--train_dir", type=str, default=None, help="a specific train dir")
    parser.add_argument("--img_res", type=int, default=64, help="select the image resolution of test images to train and test the linear probe with")
    return parser


if __name__ == "__main__":
    cli_main()
    # finish wandb operation
    wandb.finish()
    
    print("ALL DONE")
