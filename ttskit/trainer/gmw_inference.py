#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/8
"""
demo_inference

整个batch的文本等控制数据的准备。
合成器推理单位batch的文本。
声码器推理单位batch的频谱。
如果模型没有load，则自动load。
保存日志和数据。
"""
from pathlib import Path
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)


def parse_args():
    parser = argparse.ArgumentParser(description='声音编码器、语音合成器和声码器推理')
    parser.add_argument('--mellotron_path', type=str,
                        default=r"../models/mellotron/kuangdd-rtvc/mellotron.kuangdd-rtvc.pt",
                        help='Mellotron model file path')
    parser.add_argument('--waveglow_path', type=str, default='../models/waveglow/kuangdd/waveglow.kuangdd.pt',
                        help='WaveGlow model file path')
    parser.add_argument('--mellotron_hparams', type=str, default=r"../models/mellotron/samples/metadata/hparams.json",
                        help='Mellotron hparams json file path')
    parser.add_argument('--is_simple', type=int, default=1,
                        help='是否简易模式。')
    parser.add_argument('--waveglow_kwargs', type=str, default=r'{"denoiser_strength":0.1,"sigma":1}',
                        help='Waveglow kwargs json')
    parser.add_argument('--device', type=str, default='', help='Use device to inference')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='Input file path or text')
    parser.add_argument('--input', type=str, default=r"../models/mellotron/samples/metadata/validation.txt",
                        help='Input file path or text')
    parser.add_argument('--output', type=str,
                        default=r"../models/mellotron/samples/test/mellotron-000000.samples.waveglow-000000.samples",
                        help='Output file path or dir')
    parser.add_argument("--cuda", type=str, default='0,1,2,3', help='Set CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import re
import json
import functools
import multiprocessing as mp
import traceback
import tempfile

import time

import numpy as np
import pydub
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import aukit
import unidecode
import yaml
import librosa

from waveglow import inference as waveglow
from melgan import inference as melgan
from mellotron import inference as mellotron
from utils.argutils import locals2dict

from mellotron.layers import TacotronSTFT
from mellotron.hparams import create_hparams

# 用griffinlim声码器
_hparams = create_hparams()
_stft = TacotronSTFT(
    _hparams.filter_length, _hparams.hop_length, _hparams.win_length,
    _hparams.n_mel_channels, _hparams.sampling_rate, _hparams.mel_fmin,
    _hparams.mel_fmax)

_use_waveglow = 0

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
filename_formatter_re = re.compile(r'[\s\\/:*?"<>|\']+')


def process_one(kwargs: dict):
    try:
        kwargs['code'] = 'success'
        return kwargs
    except Exception as e:
        traceback.print_exc()
        kwargs['code'] = f'{e}'
        return kwargs


def run_process(n_proc=1, **kwargs):
    kwargs_lst = []
    for kw in tqdm(kwargs, 'kwargs', ncols=100):
        kwargs_lst.append(kw)

    if n_proc <= 1:
        with tempfile.TemporaryFile('w+t', encoding='utf8') as fout:
            for kw in tqdm(kwargs_lst, 'process-{}'.format(n_proc), ncols=100):
                outs = process_one(kw)
                for out in outs:
                    fout.write(f'{json.dumps(out, ensure_ascii=False)}\n')
    else:
        func = functools.partial(process_one)
        job = mp.Pool(n_proc).imap(func, kwargs_lst)
        with tempfile.TemporaryFile('w+t', encoding='utf8') as fout:
            for outs in tqdm(job, 'process-{}'.format(n_proc), ncols=100, total=len(kwargs_lst)):
                for out in outs:
                    fout.write(f'{json.dumps(out, ensure_ascii=False)}\n')


def plot_mel_alignment_gate_audio(mel, alignment, gate, audio, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(alignment, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(gate)), gate, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(gate))
    axes[3].scatter(range(len(audio)), audio, alpha=0.5, color='blue', marker='.', s=1)
    axes[3].set_xlim(0, len(audio))

    axes[0].set_title("mel")
    axes[1].set_title("alignment")
    axes[2].set_title("gate")
    axes[3].set_title("audio")

    plt.tight_layout()


def load_models(args):
    global _use_waveglow
    if args.waveglow_path and args.waveglow_path not in {'_', 'gf', 'griffinlim'}:
        waveglow.load_waveglow_torch(args.waveglow_path)
        _use_waveglow = 1

    if args.mellotron_path:
        mellotron.load_mellotron_torch(args.mellotron_path)


def transform_mellotron_input_data(dataloader, text, speaker='', audio='', device=''):
    if not device:
        device = _device

    text_data, mel_data, speaker_data, f0_data = dataloader.get_data_train_v2([audio, text, speaker])
    text_data = text_data[None, :].long().to(device)
    style_data = 0
    speaker_data = speaker_data.to(device)
    f0_data = f0_data
    mel_data = mel_data[None].to(device)

    # text_data = torch.LongTensor(phkit.chinese_text_to_sequence(text, cleaner_names='hanzi'))[None, :].to(device)
    # style_data = 0
    #
    # hex_idx = hashlib.md5(speaker.encode('utf8')).hexdigest()
    # out = (np.array([int(w, 16) for w in hex_idx])[None] - 7) / 10  # -0.7~0.8
    # speaker_data = torch.FloatTensor(out).to(device)
    # # speaker_data = torch.zeros([1], dtype=torch.long).to(device)
    # f0_data = None
    return text_data, style_data, speaker_data, f0_data, mel_data


def hello():
    waveglow.load_waveglow_torch('../models/waveglow/waveglow_v5_model.pt')
    # melgan.load_melgan_model(r'E:\githup\zhrtvc\models\vocoder\saved_models\melgan\melgan_multi_speaker.pt',
    #                          args_path=r'E:\githup\zhrtvc\models\vocoder\saved_models\melgan\args.yml')
    melgan.load_melgan_torch('../models/melgan/melgan_multi_speaker_model.pt')

    # mellotron.load_mellotron_model(r'E:\githup\zhrtvc\models\mellotron\samples\checkpoint\checkpoint-000000.pt',
    #                                hparams_path=r'E:\githup\zhrtvc\models\mellotron\samples\metadata\hparams.yml')
    #
    # torch.save(mellotron._model, '../models/mellotron/mellotron_samples_model.pt')
    mellotron.load_mellotron_torch('../models/mellotron/mellotron_samples_model.pt')

    # text, mel, speaker, f0
    text = torch.randint(0, 100, [4, 50]).cuda()
    style = 0  # torch.rand(4, 80, 400).cuda()
    speaker = torch.randint(0, 10, [4]).cuda()
    f0 = None  # torch.rand(4, 400)

    mels = mellotron.generate_mel(text=text, style=style, speaker=speaker, f0=f0)

    for mel in mels:
        print(mel.shape)

    mel = torch.rand(4, 80, 400).cuda()

    wav = waveglow.generate_wave(mel)
    print(wav.shape)


if __name__ == "__main__":
    logger.info(__file__)

    if not args.device:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        _device = args.device

    if args.is_simple:
        workdir = Path(args.mellotron_path).parent
        mellotron_stem = Path(args.mellotron_path).stem
        waveglow_stem = Path(args.waveglow_path).stem

        mellotron_hparams_path = workdir.joinpath('metadata', 'hparams.json').__str__()
        texts_path = workdir.joinpath('metadata', 'validation.txt').__str__()
        output_dir = workdir.joinpath('test', f'{mellotron_stem}.{waveglow_stem}').__str__()
    else:
        mellotron_hparams_path = args.mellotron_hparams
        texts_path = args.input
        output_dir = args.output

    # 模型导入
    load_models(args)

    mellotron_hparams = mellotron.create_hparams(open(mellotron_hparams_path, encoding='utf8').read())
    dataloader = mellotron.TextMelLoader(audiopaths_and_text='', hparams=mellotron_hparams, speaker_ids=None,
                                         mode='test')
    waveglow_kwargs = json.loads(args.waveglow_kwargs)
    # 模型测试
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = os.path.join(tmpdir, 'audio_example.wav')
        pydub.AudioSegment.silent(3000, frame_rate=args.sampling_rate).export(audio, format='wav')

        text = '这是个试水的例子。'
        speaker = 'speaker'
        text_data, style_data, speaker_data, f0_data, mel_data = transform_mellotron_input_data(
            dataloader=dataloader, text=text, speaker=speaker, audio=audio, device=_device)

        mels, mels_postnet, gates, alignments = mellotron.generate_mel(text_data, style_data, speaker_data, f0_data)

        if _use_waveglow:
            wavs = waveglow.generate_wave(mel=mels_postnet, **waveglow_kwargs)
        else:
            wavs = _stft.griffin_lim(mels_postnet)
            wav_output = wavs.squeeze().cpu().numpy()
            aukit.save_wav(wav_output, os.path.join(tmpdir, 'demo_example.wav'), sr=args.sampling_rate)

    print('Test success done.')

    # 模型推理

    if os.path.isfile(texts_path):
        text_inputs = [w.strip() for w in open(texts_path, encoding='utf8')]
        if args.is_simple:
            text_inputs = np.random.choice(text_inputs, min(len(text_inputs), 10), replace=False)
    else:
        text_inputs = [texts_path]

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    audio_lst, text_lst, speaker_lst = [], [], []
    for text_input in text_inputs:
        # print('Running: {}'.format(text_input))
        audio, text, speaker = text_input.split('\t')
        audio_lst.append(audio)
        text_lst.append(text)
        speaker_lst.append(speaker)

    # np.random.shuffle(audio_lst)
    # np.random.shuffle(text_lst)
    # np.random.shuffle(speaker_lst)

    for text_input in tqdm(zip(audio_lst, text_lst, speaker_lst), 'TTS', total=len(audio_lst), ncols=100):
        # for text_input in tqdm(text_inputs, 'TTS', ncols=100):
        # print('Running: {}'.format(text_input))
        audio, text, speaker = text_input  # .split('\t')
        text_data, style_data, speaker_data, f0_data, mel_data = transform_mellotron_input_data(
            dataloader=dataloader, text=text, speaker=speaker, audio=audio, device=_device)

        mels, mels_postnet, gates, alignments = mellotron.generate_mel(text_data, style_data, speaker_data, f0_data)

        out_gate = gates.cpu().numpy()[0]
        end_idx = np.argmax(out_gate > 0.2) or out_gate.shape[0]

        mels_postnet = mels_postnet[:, :, :end_idx]
        if _use_waveglow:
            wavs = waveglow.generate_wave(mel=mels_postnet, **waveglow_kwargs)
        else:
            wavs = _stft.griffin_lim(mels_postnet, n_iters=5)

        # 保存数据
        cur_text = filename_formatter_re.sub('', unidecode.unidecode(text))[:15]
        cur_time = time.strftime('%Y%m%d-%H%M%S')
        outpath = os.path.join(output_dir, "demo_{}_{}_out.wav".format(cur_time, cur_text))

        wav_output = wavs.squeeze(0).cpu().numpy()
        aukit.save_wav(wav_output, outpath, sr=args.sampling_rate)

        if isinstance(audio, (Path, str)) and Path(audio).is_file():
            # # 原声
            # refpath_raw = os.path.join(output_dir, "demo_{}_{}_ref_copy.wav".format(cur_time, cur_text))
            # shutil.copyfile(audio, refpath_raw)

            # 重采样
            wav_input, sr = aukit.load_wav(audio, with_sr=True)
            wav_input = librosa.resample(wav_input, sr, args.sampling_rate)
            refpath = os.path.join(output_dir, "demo_{}_{}_ref.wav".format(cur_time, cur_text))
            aukit.save_wav(wav_input, refpath, sr=args.sampling_rate)

            # # 声码器
            # wavs_ref = waveglow.generate_wave(mel=mel_data, **waveglow_kwargs)
            # outpath_ref = os.path.join(output_dir, "demo_{}_{}_ref_waveglow.wav".format(cur_time, cur_text))
            # wav_output_ref = wavs_ref.squeeze(0).cpu().numpy()
            # aukit.save_wav(wav_output_ref, outpath_ref, sr=args.sampling_rate)

        fig_path = os.path.join(output_dir, "demo_{}_{}_fig.jpg".format(cur_time, cur_text))

        plot_mel_alignment_gate_audio(mel=mels_postnet.squeeze(0).cpu().numpy(),
                                      alignment=alignments.squeeze(0).cpu().numpy(),
                                      gate=gates.squeeze(0).cpu().numpy(),
                                      audio=wav_output[::args.sampling_rate // 1000])
        plt.savefig(fig_path)
        plt.close()

        yml_path = os.path.join(output_dir, "demo_{}_{}_info.yml".format(cur_time, cur_text))
        info_dict = locals2dict(locals())
        with open(yml_path, 'wt', encoding='utf8') as fout:
            yaml.dump(info_dict, fout, encoding='utf-8', allow_unicode=True)

        log_path = os.path.join(output_dir, "info_dict.txt".format(cur_time))
        with open(log_path, 'at', encoding='utf8') as fout:
            fout.write('{}\n'.format(json.dumps(info_dict, ensure_ascii=False)))
