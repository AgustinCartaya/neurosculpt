"""Train the 2D StarGAN v2 (HUNT3 <-> HUNT4 domains) on sagittal slices, guided by brain masks.

Slices are read from paths.get_images_folder_2D() (created by the 2D slice preprocessing).
The first 620 matching patients are used for training and the rest for the debug samples.
The thesis model is the EMA checkpoint at iteration 65200 (training was resumed several times,
e.g. --resume_iter 58000).

    python scripts/train_stargan.py
    python scripts/train_stargan.py --resume_iter 58000
    python scripts/train_stargan.py --mode sample --resume_iter 65200
"""
import argparse

import numpy as np
import torch
from torch.backends import cudnn

from brainpred import paths, splits
from brainpred.data.stargan_2d import DataLoaderImagesStarGAN2D, InputFetcher
from brainpred.methods.stargan_v2.predict import METHOD_NAME, get_results_layout
from brainpred.methods.stargan_v2.solver import Solver

N_TRAIN = 620  # NOTE: the 3D methods use 621 (splits.get_train_ids()); the StarGAN was trained with 620.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # model arguments
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--n_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=3, help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=2, help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=2, help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000, help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--total_iters', type=int, default=100000, help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10, help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'], help='This argument is used in solver')
    parser.add_argument('--seed', type=int, default=777, help='Seed for random number generator')

    # directories (default: checkpoints in <checkpoints_dir>/StarGAN_V2_sagital, images in <results_dir>/StarGAN_V2_sagital/training)
    parser.add_argument('--sample_dir', type=str, default=None, help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving network checkpoints')
    parser.add_argument('--result_dir', type=str, default=None, help='Directory for saving generated images')

    # step size
    parser.add_argument('--print_every', type=int, default=80)
    parser.add_argument('--sample_every', type=int, default=80)
    parser.add_argument('--save_every', type=int, default=400)
    args = parser.parse_args()

    # resolved after parsing so that --help works without a config file
    training_folder = f"{get_results_layout().get_images_path()}/training"
    args.sample_dir = args.sample_dir or f"{training_folder}/samples"
    args.checkpoint_dir = args.checkpoint_dir or paths.get_checkpoints_folder(METHOD_NAME)
    args.result_dir = args.result_dir or f"{training_folder}/results"
    return args


def make_loader(args, ids, batch_size, mode):
    return DataLoaderImagesStarGAN2D(img_size=[args.img_size, args.img_size], n_channels=args.n_channels,
                                     path=paths.get_images_folder_2D(), img_sufix=paths.INAME2D_T1, ids=ids,
                                     layers=list(range(paths.LAYER_START_2D, paths.LAYER_END_2D)), batch_size=batch_size,
                                     possible_table=None, shuffle=True, channels_first=True, mode=mode, dtype=np.float32)


def main():
    args = parse_args()
    print("CUDA available:", torch.cuda.is_available())
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)
    matching_ids = splits.get_matching_ids()
    data_val_fetcher = InputFetcher(make_loader(args, matching_ids[N_TRAIN:], args.val_batch_size, "val"), args.latent_dim)

    if args.mode == 'train':
        data_train_fetcher = InputFetcher(make_loader(args, matching_ids[:N_TRAIN], args.batch_size, "train"), args.latent_dim)
        solver.train(data_train_fetcher, data_val_fetcher, mask_name=paths.INAME2D_BMASK)
    else:
        solver.sample(data_val_fetcher, mask_name=paths.INAME2D_BMASK)


if __name__ == "__main__":
    main()
