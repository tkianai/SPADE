#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python run_server.py --label_nc 19 --load_size 512 --crop_size 512 --no_instance --name celebA --which_epoch 20 --dataset_mode custom --label_dir images --image_dir images