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
from datamodules.transforms import PixelScrambleTransform
import torch
import torchvision.transforms as transforms

# model
from models.vit import Backbone, configuration, LitClassifier

def create_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Max number of epochs to train."
    )
    parser.add_argument(
        "--val_split",
        default=0.1,
        type=float,
        help="Percent (float) of samples to use for the validation split."
    )
    #The action set to store_true will store the argument as True , if present
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporally ordered image pairs."
    )
    parser.add_argument(
        "--window_size",
        default=3,
        type=int,
        help="Size of sliding window for sampling temporally ordered image pairs."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--seed_val",
        type=int,
        default=0,
        help="SEED VALUE"
    )
    parser.add_argument(
        "--shuffle_frames",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--shuffle_temporalWindows",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--dataloader_shuffle",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--image_size",
        default=64,
        type=int,
        help="supported images :: 224X224 and 64X64"
    )
    parser.add_argument(
        "--patch_size",
        default=8,
        type=int,
        help="Square patch size"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=1,
        help="number of attention heads"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="num of training samples to use from dataset. -1 = entire dataset"
    )
    parser.add_argument(
        "--transforms",
        type=str,
        choices=['transform_gB', 'transform_grayScale', 'transform_randomCrop', 
                 'transform_randomHFlip', 'transform_cj', 'transform_all', 'transform_resize', 
                 'transform_none', 'transform_cropped_resize', 'transform_randomCropWithProb',
                 'transform_spatialScrambling', 'transform_combo', 'transform_randomHFlip_highProb', 'transform_randomHFlip_lowProb'],
        default='None',
        help="data augmentation transform"
    )

    parser.add_argument(
        "--print_model",
        action="store_true",
        help="display backbone"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="BATCH SIZE"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="dataset directory"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="NUM OF WORKERS"
    )
    parser.add_argument(
        "--resize_dims",
        type=int,
        choices=[64,128,224,256],
        default=64,
        help="resize the image to a desired resolution if transform_resize is selected"
    )
    parser.add_argument(
        "--loss_ver",
        type=str,
        choices=['v0','v1', 'v2'],
        default='v0',
        help="select btw CLTT loss version 0 and loss version 1. Same objectives but different implementations"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='/data/lpandey/LOGS/VIT_Time',
        help="the checkpoints for the trained models will be saved in this dir"
    )
    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default='NA',
        help="path of the saved model checkpoint that you wish to finetune"
    )
    parser.add_argument(
        "--finetune_ckpt_freeze_layers",
        type=int,
        default=0,
        help="number of transformer layers you wish to freeze when finetuning a saved checkpoint"
    )
    parser.add_argument(
        "--spatialTransform_patchsize",
        type=int,
        default=1,
        help="patch size used for spatial scrambling"
    )
    
    return parser


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

    parser = create_argparser()

    # model args
    args = parser.parse_args()
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


    # data transforms
    if args.transforms == 'transform_gB':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_grayScale':
        trans = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomCrop':
        trans = transforms.Compose([
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomHFlip':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomHFlip_highProb':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomHFlip_lowProb':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_cj':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_all':
        # gb
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
        ], p=0.5),
        # grayscale 
        transforms.RandomGrayscale(p=0.2),
        # crop
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        # hflip
        transforms.RandomHorizontalFlip(p=0.5),
        # cj
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        # resize to desired resolution
        transforms.Resize((args.resize_dims, args.resize_dims)),
        # final transform
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_none':
        trans = transforms.Compose([
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_resize':
        trans = transforms.Compose([
        transforms.Resize((args.resize_dims, args.resize_dims)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_cropped_resize': # custom transform class
        trans = transforms.Compose([
            CenterCropLongDimension(),
            transforms.Resize((args.resize_dims, args.resize_dims)),
            transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomCropWithProb':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0), ratio=(0.75, 1.3333)),
        ], p=0.5),
        transforms.ToTensor()
])
    elif args.transforms == 'transform_spatialScrambling':
        trans = transforms.Compose([
        PixelScrambleTransform(args.spatialTransform_patchsize, (args.image_size, args.image_size)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_combo':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        ])


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