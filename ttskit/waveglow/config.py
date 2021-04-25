#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2021/2/3
"""
config
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

config = {
    "train_config": {
        "fp16_run": False,
        "output_directory": "../models/waveglow/samples/checkpoint",
        "epochs": 100000,
        "learning_rate": 1e-4,
        "sigma": 1.0,
        "iters_per_checkpoint": 10,
        "batch_size": 2,
        "seed": 1234,
        "checkpoint_path": "",
        "with_tensorboard": True,
        "dataloader_num_workers": 4,
        "dataloader_shuffle": True,
        "dataloader_pin_memory": True
    },
    "data_config": {
        "training_files": "../data/samples/metadata.csv",
        "segment_length": 8000,
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0
    },
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321"
    },

    "waveglow_config": {
        "n_mel_channels": 80,
        "n_flows": 12,
        "n_group": 8,
        "n_early_every": 4,
        "n_early_size": 2,
        "WN_config": {
            "n_layers": 8,  # 4 缩小参数量
            "n_channels": 256,  # 128 缩小参数量
            "kernel_size": 3
        }
    }
}

if __name__ == "__main__":
    logger.info(__file__)
