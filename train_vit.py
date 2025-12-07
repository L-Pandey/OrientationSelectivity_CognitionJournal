# Handle warnings
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


# LIBRARIES
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

# Pytorch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules import ImagePairsDataModule
from datamodules.transforms import PixelScrambleTransform, get_torch_transforms
import torch
import torchvision.transforms as transforms

# model
from models.vit import Backbone, configuration, LitClassifier
from args import get_args


# Freeze model layers
def freeze_layers(model, args):
    ############ transformer backbone ###############  
    for i, (name, param) in enumerate(model.named_parameters()):
        if args.finetune_ckpt_freeze_layers == 1:

            # freeze the first transformer block only
            if "transformer.layers.0" in name:
                param.requires_grad = False 

        elif args.finetune_ckpt_freeze_layers == 2:

            # freeze the first two transformer blocks
            if "transformer.layers.0" in name:
                param.requires_grad = False
            if "transformer.layers.1" in name:
                param.requires_grad = False

    # print non frozen layers - 
    print("[INFO] Active layers (Trainable layers) are :: ")
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if param.requires_grad == True:
    #         print(i, name)

def cli_main():

    # get arguments
    args = get_args()

    # single gpu training
    args.gpus = 1

    # set seed value
    pl.seed_everything(args.seed_val)
    torch.manual_seed(args.seed_val)


    # assign heads and hidden layers
    # heads and hidden_layers are same.
    configuration.num_attention_heads = args.head
    configuration.num_hidden_layers = args.head

    # assing image size and patch size
    configuration.image_size = args.image_size
    configuration.patch_size = args.patch_size

    print("[INFO] Number of ATTENTION HEADS :: ", configuration.num_attention_heads)
    print("[INFO] Number of HIDDEN LAYERS :: ", configuration.num_hidden_layers)
    print("[INFO] Image Size :: ", configuration.image_size)
    print("[INFO] Patch Size :: ", configuration.patch_size)
    print("[INFO] Temporal Window Size :: ", args.window_size)

    # get data transform
    trans = get_torch_transforms(args)

    if args.temporal:
        dm = ImagePairsDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_frames = args.shuffle_frames,
            shuffle_temporalWindows = args.shuffle_temporalWindows,
            dataloader_shuffle = args.dataloader_shuffle,
            drop_last=False,
            val_split=args.val_split,
            window_size=args.window_size,
            dataset_size=args.dataset_size,
            gpus=args.gpus,
            transform=trans,
        )

    dm.setup("fit")
    args.num_samples = len(dm.train_dataset)

    # setup model for training from scratch (no finetuning)
    if args.finetune_ckpt == 'NA': 
        print("[INFO] Training from scratch (no finetune)")
        backbone = Backbone('vit', configuration)
        model = LitClassifier(backbone=backbone, 
                              window_size=args.window_size, 
                              loss_ver=args.loss_ver, 
                              )
    
    # setup model for finetuning
    else:
        print("[INFO] Finetuning model saved at :: ", args.finetune_ckpt)
        backbone = Backbone('vit', configuration)
        model = LitClassifier(backbone=backbone, 
                              window_size=args.window_size, 
                              loss_ver=args.loss_ver, 
                              gpus=args.gpus,
                              batch_size=args.batch_size,
                              num_samples=args.num_samples,
                              max_epochs=args.max_epochs)
        
        model = model.load_from_checkpoint(args.finetune_ckpt)

        # condition to decide if layers should be frozen
        if args.finetune_ckpt_freeze_layers > 0:
            print("[INFO] Freezing layers")
            freeze_layers(model, args)
        else:
            print("[INFO] All layers are trainable")
    
    print("[INFO] Shuffle (frames) set to :: ", dm.shuffle_frames)

    print("[INFO] Shuffle (temporal windows) set to :: ", dm.shuffle_temporalWindows)

    print("[INFO] Train dataloader shuffle set to :: ", dm.dataloader_shuffle)

    print("[INFO] Passing through transformations :: {}".format(dm.transform))

    print("[INFO] Loss function version :: {}".format(model.loss_ver))


    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger(args.log_path, name=f"{args.exp_name}")
   
    trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            logger=logger,
            sync_batchnorm=True if args.gpus > 1 else False,
            callbacks=callbacks,
        )

    if args.print_model:
        print(model)
    
    # train model
    trainer.fit(model, datamodule=dm)




if __name__ == '__main__':
    cli_main()