#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/7
"""
waveglow_inference
"""
from pathlib import Path
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--waveglow_path', default='../models/waveglow/samples/checkpoint/waveglow-000000.pt',
                        type=str,
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('--is_simple', type=int, default=1,
                        help='是否简易模式。')
    parser.add_argument('-i', "--input_path", default='../data/samples/wav', type=str)
    parser.add_argument('-o', "--output_path", default='../models/waveglow/samples/test/waveglow-000000', type=str)
    parser.add_argument("-c", "--config_path", default='../models/waveglow/samples/metadata/config.json', type=str)
    parser.add_argument('--kwargs', type=str, default=r'{"denoiser_strength":0.1,"sigma":1}',
                        help='Waveglow kwargs json')
    parser.add_argument("--cuda", type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument("--save_model_path", type=str, default='../models/waveglow/samples/waveglow-000000.model.pt',
                        help='Save model for torch load')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import time
import json
from scipy.io import wavfile
import shutil

import torch
import librosa
from tqdm import tqdm
import numpy as np

# from mellotron.layers import TacotronSTFT
from waveglow.mel2samp import MAX_WAV_VALUE, Mel2Samp, load_wav_to_torch
from waveglow.inference import Denoiser


# 要把glow所在目录包含进来，否则导致glow缺失报错。

def main(input_path, waveglow_path, config_path, output_path, save_model_path, is_simple=1, **kwargs):
    denoiser_strength = kwargs.get('denoiser_strength', 0)
    sigma = kwargs.get('sigma', 1.0)

    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if save_model_path:
        torch.save(waveglow, save_model_path)

    denoiser = Denoiser(waveglow).cuda()

    # waveglow = torch.load('../waveglow_v5_model.pt', map_location='cuda')
    with open(config_path) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    input_path = str(input_path)
    if os.path.isfile(input_path) and input_path.endswith('txt'):
        audio_path_lst = [w.strip() for w in open(input_path, encoding='utf8')]
    elif os.path.isdir(input_path):
        audio_path_lst = [w for w in Path(input_path).glob('**/*') if w.is_file() and w.name.endswith(('mp3', 'wav'))]
    else:
        audio_path_lst = [input_path]

    if is_simple:
        audio_path_lst = np.random.choice(audio_path_lst, min(10, len(audio_path_lst)), replace=False)

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    for audio_path in tqdm(audio_path_lst, 'waveglow', ncols=100):
        audio_path = Path(audio_path)
        cur_time = time.strftime('%Y%m%d-%H%M%S')
        audio_name = f'waveglow_{cur_time}_{audio_path.name}'
        outpath = output_dir.joinpath(audio_name)
        name_cnt = 2
        while outpath.is_file():
            outpath = output_dir.joinpath(f'{audio_path.stem}-{name_cnt}{audio_path.suffix}')
            name_cnt += 1

        shutil.copyfile(audio_path, outpath)

        # 用mellotron的模块等价的方法生成频谱
        # audio_norm, sr = librosa.load(str(audio_path), sr=None)
        # audio_norm = torch.from_numpy(audio_norm).unsqueeze(0)
        # stft = TacotronSTFT(mel_fmax=8000.0)
        # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # mel = stft.mel_spectrogram(audio_norm)
        # mel = torch.autograd.Variable(mel.cuda())

        audio, sr = load_wav_to_torch(audio_path, sr_force=data_config['sampling_rate'])
        mel = mel2samp.get_mel(audio)
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)

        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        outpath = output_dir.joinpath(f'{outpath.name}.waveglow.wav')
        wavfile.write(outpath, data_config['sampling_rate'], audio)


if __name__ == "__main__":
    args = parse_args()

    if args.is_simple:
        workdir = Path(args.waveglow_path).parent.parent
        model_stem = Path(args.waveglow_path).stem
        input_path = workdir.joinpath('metadata', 'train.txt')
        waveglow_path = args.waveglow_path
        output_path = workdir.joinpath('test', model_stem)
        config_path = workdir.joinpath('metadata', 'config.json')
        save_model_path = workdir.joinpath(f'{model_stem}.{workdir.stem}.pt')
    else:
        input_path = args.input_path
        waveglow_path = args.waveglow_path
        output_path = args.output_path
        config_path = args.config_path
        save_model_path = args.save_model_path

    args_kwargs = json.loads(args.kwargs)
    main(input_path=input_path,
         waveglow_path=waveglow_path,
         output_path=output_path,
         config_path=config_path,
         save_model_path=save_model_path,
         **args_kwargs)
