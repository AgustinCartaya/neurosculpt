"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
sys.path.append(base_path)


from munch import Munch
from torch.backends import cudnn
import torch
import numpy as np
import project_images_loader as pil
from data_loaders.data_loaders_2D.data_loader_stargan_2D_images import DataLoaderImagesStarGAN2D
from data_loaders.data_loaders_2D.data_loader_stargan_2D_input_fetcher import InputFetcher

from core.solver import Solver


# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    # print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    images_path = pil.get_images_folder_2D(absolute=True)
    matching_ids = pil.get_matching_ids()
    layers = [slide for  slide in range(pil.LAYER_START_2D, pil.LAYER_END_2D)]
    server = False

    if args.mode == 'train':

        data_train = DataLoaderImagesStarGAN2D(img_size=[args.img_size, args.img_size],
                                               n_channels=args.n_channels,
                                               path=images_path,
                                                img_sufix=pil.INAME2D_T1,
                                                ids=matching_ids[:620],
                                                layers=layers,
                                                batch_size=args.batch_size,
                                                possible_table=None,
                                                shuffle=True,
                                                server=server,
                                                channels_first=True,
                                                mode="train",
                                                dtype=np.float32)
 
        data_val = DataLoaderImagesStarGAN2D(img_size=[args.img_size, args.img_size],
                                               n_channels=args.n_channels,
                                               path=images_path,
                                                img_sufix=pil.INAME2D_T1,
                                                ids=matching_ids[620:],
                                                layers=layers,
                                                batch_size=args.val_batch_size,
                                                possible_table=None,
                                                shuffle=True,
                                                server=server,
                                                channels_first=True,
                                                mode="val",
                                                dtype=np.float32)
        
        data_train_fetcher = InputFetcher(data_train, args.latent_dim)
        data_val_fetcher = InputFetcher(data_val, args.latent_dim)
        
        solver.train(data_train_fetcher, data_val_fetcher, mask_name=pil.INAME2D_BMASK)

    elif args.mode == 'sample':
        data_val = DataLoaderImagesStarGAN2D(img_size=[args.img_size, args.img_size],
                                            n_channels=args.n_channels,
                                            path=images_path,
                                            img_sufix=pil.INAME2D_T1,
                                            ids=matching_ids[620:],
                                            layers=layers,
                                            batch_size=args.val_batch_size,
                                            possible_table=None,
                                            shuffle=True,
                                            server=server,
                                            channels_first=True,
                                            mode="val",
                                            dtype=np.float32)
        
        data_val_fetcher = InputFetcher(data_val, args.latent_dim)
    
        solver.sample(data_val_fetcher, mask_name=pil.INAME2D_BMASK)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--n_channels', type=int, default=1,
                        help='Image channels')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=3,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=2,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=2,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=58000,
                        help='Iterations to resume training/testing')
    

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=4,
                        help='Batch size for validation')
    

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'sample', 'eval'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--sample_dir', type=str, default=f'{current_path}/expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default=f'{current_path}/expr/checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--result_dir', type=str, default=f'{current_path}/expr/results',
                        help='Directory for saving generated images and videos')

    # step size
    parser.add_argument('--print_every', type=int, default=80)
    parser.add_argument('--sample_every', type=int, default=80)
    parser.add_argument('--save_every', type=int, default=400)

    
    args = parser.parse_args()

    main(args)

