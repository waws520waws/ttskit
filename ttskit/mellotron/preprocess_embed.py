# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/10/9
"""
preprocess_embed
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys

sys.path.append(str(Path(__file__).absolute().parent.parent))

from functools import partial
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from hparams import create_hparams

hp = create_hparams()

import aukit
from encoder import inference as encoder


def embed_utterance(src, skip_existing=True, encoder_model_fpath=Path()):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    wav_fpath, embed_fpath = src

    if skip_existing and embed_fpath.is_file():
        return

    wav = aukit.load_wav(wav_fpath, sr=hp.sampling_rate)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(n_processes, txt_fpath, skip_existing=False, encoder_model_fpath=Path()):
    # Embed the utterances in separate threads
    # 000000	F:\github\zhrtvc\data\samples_ssml\Aiyue/000003.wav	<speak><phoneme alphabet="py" ph="bao2 ma3">宝马</phoneme>。</speak>	Aiyue
    npy_dir = Path(txt_fpath).parent.joinpath('npy')
    ids = []
    with open(txt_fpath, encoding='utf8') as fin:
        for line in fin:
            embed_fpath = npy_dir.joinpath(line.split('\t')[0], 'embed.npy')
            wav_fpath = line.split('\t')[1]
            ids.append((wav_fpath, embed_fpath))

    if n_processes <= 1:
        for index in tqdm(ids, ncols=50):
            try:  # 防止少数错误语音导致生成数据失败。
                embed_utterance(index, skip_existing=skip_existing, encoder_model_fpath=encoder_model_fpath)
            except Exception as e:
                logger.info('Error! The <{}> audio load failed! {}'.format(index, e))
                logger.info('=' * 50)
    else:
        func = partial(embed_utterance, skip_existing=skip_existing, encoder_model_fpath=encoder_model_fpath)
        job = Pool(n_processes).imap(func, ids)
        list(tqdm(job, "Embedding", len(ids), unit="utterances", ncols=50))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="把语音信号转为语音表示向量。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input_fpath", type=Path,
                        default=Path(r'../../data/SV2TTS/mellotron/samples/train.txt'),
                        help="文本路径。")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("-s", "--skip_existing", type=bool, default=True,
                        help="Whether to overwrite existing files with the same name. ")
    parser.add_argument("-n", "--n_processes", type=int, default=0,
                        help="进程数。")
    parser.add_argument("--hparams", type=str, default="",
                        help="Hyperparameter overrides as a json string.")
    args = parser.parse_args()

    # Preprocess the dataset
    create_embeddings(
        n_processes=args.n_processes,
        txt_fpath=args.input_fpath,
        skip_existing=args.skip_existing,
        encoder_model_fpath=args.encoder_model_fpath)
