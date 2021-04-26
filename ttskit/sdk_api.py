# author: kuangdd
# date: 2021/4/23
"""
### sdk_api
语音合成SDK接口。
本地函数式地调用语音合成。

+ 简单使用
```python
from ttskit import sdk_api

wav = sdk_api.tts_sdk('文本', audio='1')
```
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import os
import argparse
import json
import tempfile
import base64
import numpy as np
import torch
import aukit
import tqdm
import requests

from waveglow import inference as waveglow
from mellotron import inference as mellotron
from mellotron.layers import TacotronSTFT
from mellotron.hparams import create_hparams

_home_dir = os.path.dirname(os.path.abspath(__file__))

# 用griffinlim声码器
_hparams = create_hparams()
_stft = TacotronSTFT(
    _hparams.filter_length, _hparams.hop_length, _hparams.win_length,
    _hparams.n_mel_channels, _hparams.sampling_rate, _hparams.mel_fmin,
    _hparams.mel_fmax)

_use_waveglow = 0
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

_mellotron_path = os.path.join(_home_dir, 'resource', 'model', 'mellotron.kuangdd-rtvc.pt')
_waveglow_path = os.path.join(_home_dir, 'resource', 'model', 'waveglow.kuangdd.pt')
_ge2e_path = os.path.join(_home_dir, 'resource', 'model', 'ge2e.kuangdd.pt')
_mellotron_hparams_path = os.path.join(_home_dir, 'resource', 'model', 'mellotron_hparams.json')
_reference_audio_tar_path = os.path.join(_home_dir, 'resource', 'reference_audio.tar')
_audio_tar_path = os.path.join(_home_dir, 'resource', 'audio.tar')
_dataloader = None

_reference_audio_list = []

tmp = os.path.splitext(_audio_tar_path)[0]
if os.path.isdir(tmp):
    tmp = list(sorted([*Path(tmp).glob('*.wav'), *Path(tmp).glob('*.mp3')]))
    _reference_audio_list.extend(tmp)

tmp = os.path.splitext(_reference_audio_tar_path)[0]
if os.path.isdir(tmp):
    tmp = list(sorted([*Path(tmp).glob('*.wav'), *Path(tmp).glob('*.mp3')]))
    _reference_audio_list.extend(tmp)

_reference_audio_list = [w.__str__() for w in _reference_audio_list]
_reference_audio_dict = {os.path.basename(w).split('-')[1]: w for w in _reference_audio_list}


def download_resource():
    global _audio_tar_path, _reference_audio_tar_path
    global _ge2e_path, _mellotron_path, _mellotron_hparams_path, _waveglow_path
    for fpath in [_audio_tar_path, _reference_audio_tar_path]:
        flag = download_data(fpath)
        if flag or not os.path.isdir(fpath[:-4]):
            ex_tar(fpath)

    for fpath in [_ge2e_path, _mellotron_hparams_path, _mellotron_path, _waveglow_path]:
        download_data(fpath)


def download_data(fpath):
    url_prefix = 'http://www.kddbot.com:11000/data/'
    url_info_prefix = 'http://www.kddbot.com:11000/data_info/'

    fname = os.path.relpath(fpath, _home_dir).replace('\\', '/')
    fname_key = fname.replace('/', ';').replace('resource', 'ttskit_resource')
    url = f'{url_prefix}{fname_key}'

    url_info = f'{url_info_prefix}{fname_key}'
    res = requests.get(url_info)
    if res.status_code == 200:
        fsize = res.json()['file_size']
    else:
        logger.info(f'Download <{fname}> failed!!!')
        logger.info(f'Download url: {url}')
        logger.info(f'Download failed! Please check!')
        return

    if os.path.isfile(fpath):
        if os.path.getsize(fpath) == fsize:
            logger.info(f'File <{fname}> exists.')
            return
        else:
            logger.info(f'File <{fname}> exists but size not match!')
            logger.info(f'Local size {os.path.getsize(fpath)} != {fsize} Url size. Re download.')

    logger.info(f'Downloading <{fname}> start.')
    logger.info(f'Downloading url: {url}')

    res = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'wb') as fout:
        for chunk in tqdm.tqdm(res.iter_content(chunk_size=1024),
                               fname, mininterval=1, unit='KB', total=fsize // 1024):
            if chunk:
                fout.write(chunk)

    logger.info(f'Downloaded <{fname}> done.')
    logger.info(f'Downloaded file: {fpath}')
    return True


def ex_tar(inpath):
    """"""
    import tarfile
    outdir = os.path.dirname(inpath)
    with tarfile.open(inpath, 'r') as fz:
        # fz.gettarinfo()
        for fname in tqdm.tqdm(fz.getnames(), os.path.basename(inpath), ncols=100, mininterval=1):
            fz.extract(fname, outdir)


def load_models(mellotron_path=_mellotron_path,
                waveglow_path=_waveglow_path,
                ge2e_path=_ge2e_path,
                mellotron_hparams_path=_mellotron_hparams_path,
                **kwargs):
    global _use_waveglow
    global _dataloader

    if (mellotron_path == _mellotron_path
            and waveglow_path == _waveglow_path
            and ge2e_path == _ge2e_path
            and mellotron_hparams_path == _mellotron_hparams_path):
        download_resource()

    if _dataloader is not None:
        return
    if waveglow_path and waveglow_path not in {'_', 'gf', 'griffinlim'}:
        waveglow.load_waveglow_torch(waveglow_path)
        _use_waveglow = 1

    if mellotron_path:
        mellotron.load_mellotron_torch(mellotron_path)

    mellotron_hparams = mellotron.create_hparams(open(mellotron_hparams_path, encoding='utf8').read())
    mellotron_hparams.encoder_model_fpath = ge2e_path
    _dataloader = mellotron.TextMelLoader(audiopaths_and_text='',
                                          hparams=mellotron_hparams,
                                          speaker_ids=None,
                                          mode='test')
    return _dataloader


def transform_mellotron_input_data(dataloader, text, speaker='', audio='', device=''):
    if not device:
        device = _device

    text_data, mel_data, speaker_data, f0_data = dataloader.get_data_train_v2([audio, text, speaker])
    text_data = text_data[None, :].long().to(device)
    style_data = 0
    speaker_data = speaker_data.to(device)
    f0_data = f0_data
    mel_data = mel_data[None].to(device)

    return text_data, style_data, speaker_data, f0_data, mel_data


def tts_sdk(text, speaker='biaobei', audio='0', **kwargs):
    global _dataloader
    if _dataloader is None:
        load_models(**kwargs)

    if str(audio).isdigit():
        audio = _reference_audio_list[(int(audio) - 1) % len(_reference_audio_list)]
    elif os.path.isfile(audio):
        audio = str(audio)
    elif isinstance(audio, bytes):
        tmp_audio = tempfile.TemporaryFile(suffix='.wav')
        tmp_audio.write(audio)
        audio = tmp_audio.name
    elif isinstance(audio, str) and len(audio) >= 100:
        tmp_audio = tempfile.TemporaryFile(suffix='.wav')
        tmp_audio.write(base64.standard_b64decode(audio))
        audio = tmp_audio.name
    elif speaker in _reference_audio_dict:
        audio = _reference_audio_dict[speaker]
    else:
        raise AssertionError
    text_data, style_data, speaker_data, f0_data, mel_data = transform_mellotron_input_data(
        dataloader=_dataloader, text=text, speaker=speaker, audio=audio, device=_device)

    mels, mels_postnet, gates, alignments = mellotron.generate_mel(text_data, style_data, speaker_data, f0_data)

    out_gate = gates.cpu().numpy()[0]
    end_idx = np.argmax(out_gate > 0.2) or np.argmax(out_gate) or out_gate.shape[0]

    mels_postnet = mels_postnet[:, :, :end_idx]
    if _use_waveglow:
        wavs = waveglow.generate_wave(mel=mels_postnet, **kwargs)
    else:
        wavs = _stft.griffin_lim(mels_postnet, n_iters=5)

    wav_output = wavs.squeeze(0).cpu().numpy()

    output = kwargs.get('output', '')
    if output.startswith('play'):
        aukit.play_sound(wav_output, sr=_stft.sampling_rate)
    if output.endswith('.wav'):
        aukit.save_wav(wav_output, output, sr=_stft.sampling_rate)
    wav_output = aukit.anything2bytes(wav_output, sr=_stft.sampling_rate)
    return wav_output


if __name__ == "__main__":
    logger.info(__file__)
    load_models(mellotron_path=_mellotron_path,
                waveglow_path=_waveglow_path,
                ge2e_path=_ge2e_path,
                mellotron_hparams_path=_mellotron_hparams_path)
    wav = tts_sdk(text='这是个示例', speaker='biaobei', audio='0')
