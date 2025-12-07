import argparse

def get_args():
    parser = argparse.ArgumentParser()
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
        choices=['gB', 'gs', 'rc', 'rhf', 'cj', 'all', 
        'resize', 'none', 'cropped_resize', 'spatialScrambling', 
        'rhf_highProb', 'rhf_lowProb'],
        default='none',
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
    
    return parser.parse_args()