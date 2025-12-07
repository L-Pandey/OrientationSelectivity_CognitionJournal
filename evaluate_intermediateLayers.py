from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import os
import torch.nn as nn
import sys
from itertools import islice
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf warnings set to silent in terminal

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path
sys.path.append(parent_dir)

from pytorch_lightning.callbacks import ModelCheckpoint

from models import SimCLR
from pytorch_lightning.loggers import WandbLogger

import wandb
from models.vit_contrastive import Backbone, LitClassifier, ViTConfigExtended 
from datamodules.dataHandler import OrientationRecognition
from models.intermediate import Intermediate


def cli_main():
    parser = create_argparser()
    args = parser.parse_args()
   

    validation(args)

def get_layers():

    vit_layer_dict = {
        'layer1':'model.model.transformer.layers.0.1',
        'layer2':'model.model.transformer.layers.1.1',
        'layer3':'model.model.transformer.layers.2.1',
        'layer4':'model.model.transformer.layers.3.1',
        'layer5':'model.model.transformer.layers.4.1',
        'layer6':'model.model.mlp_head',
    }

    return vit_layer_dict


def get_selected_layers(vit_layer_dict, total_layers):
    # Always include the last layer (mlp_head)
    keys = list(vit_layer_dict.keys())

    # Always keep mlp_head
    mlp_key = keys[-1]

    # Take total_layers - 1 transformer layers, then add mlp_head
    selected_keys = keys[:total_layers - 1] + [mlp_key]

    # Return the corresponding subset of the dictionary
    return {k: vit_layer_dict[k] for k in selected_keys}

def infer_layer_feature_dim(model, layer, img_size=(3, 64, 64), device='cuda'):
    dummy_input = torch.randn(1, *img_size)

    device = next(model.parameters()).device  # get the model's device (e.g., cuda)
    dummy_input = dummy_input.to(device)     # move dummy_input to that device
    feature = None

    def hook_fn(module, input, output):
        nonlocal feature
        if output.dim() > 2:
            output = output[:, 1:, :].mean(dim=1)
        feature = output

    target = model
    for part in layer.split('.'):
        target = getattr(target, part)
    handle = target.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    handle.remove()
    return feature.shape[1]  # output dim


def validation(args):

    # Load data
    dm = OrientationRecognition(
            data_dir=args.data_dir,
            batch_size=128,
            pin_memory=False,
            num_workers=16,
            val_split=args.val_split,
            drop_last=False,
            shuffle=args.shuffle,
            train_dir=None,
            img_res=args.img_res,
        )
    
    print("[INFO] Dataloader loaded successfully")
    
    print("[INFO] Shuffle is set to ", dm.shuffle)
    

    model = init_model(args)

    
    # train and test each intermediate layer of ViTs
    vit_layer_dict = get_layers()

    selected_layers = get_selected_layers(vit_layer_dict, args.total_model_layers)


    for _, layer in islice(selected_layers.items(), args.total_model_layers):
        print("[INFO] Probing layer ", layer)

        in_features = infer_layer_feature_dim(model, layer, device='cuda') 
        print("[INFO] Feature dimension for layer ", in_features)

        evaluator = Intermediate(
            backbone=model, 
            layer=layer,
            max_epochs=args.max_epochs,
            in_features=in_features
            )
    
        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')

        callbacks = [model_checkpoint]

        # create a new logger  for wandB
        logger = WandbLogger(
        save_dir=f"/data/lpandey/LOGS/eval/{args.model}",
        name=f"{args.exp_name}_layer_{layer}",
        project=f"{args.project_name}",
        log_model="all"
        )
       
        trainer = pl.Trainer(
            gpus=1,
            logger=logger,
            max_epochs=args.max_epochs,
            callbacks=callbacks
        )

        trainer.fit(evaluator, datamodule=dm)

        # log average test acc accross all the classes
        print("[INFO] Logging Average Test Acc accross all the classes!")
        trainer.test(datamodule=dm)

        # finish wandb operation
        wandb.finish()

 

def init_model(args):
    if args.model == 'pixels':
        model = nn.Flatten()
    elif args.model == 'simclr':
        model = SimCLR.load_from_checkpoint(args.model_path)    
    elif args.model == 'vit':
        model = LitClassifier.load_from_checkpoint(args.model_path).backbone
        model.fc = nn.Identity()
    elif args.model == 'untrained_vit':
        configuration = ViTConfigExtended()
        configuration.num_hidden_layers = args.total_model_layers
        configuration.num_attention_heads = args.total_model_layers
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


def create_argparser():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="directory containing train n test datasets")
    parser.add_argument("--exp_name", type=str, help="wandb experiment name")
    parser.add_argument("--model", type=str, choices=['pixels', 'simclr', 'vit', 'untrained_vit'])
    parser.add_argument("--total_model_layers", type=int, help="total number of layers in the model, for e.x - 1,2,3,4,5,6,...")
    parser.add_argument("--model_path", type=str, help="stored model checkpoint")
    parser.add_argument("--max_epochs", default=100, type=int, help="Max number of epochs to train.")
    parser.add_argument("--project_name", type=str, help="wandb project_name") # for wandb dashboard and logging
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle images for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="validation samples split ratio")
    parser.add_argument("--img_res", type=int, default=64, help="select the image resolution of test images to train and test the linear probe with")

    return parser


if __name__ == "__main__":
    cli_main()
    
    print("ALL DONE")
