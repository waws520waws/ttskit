#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/4/14
"""
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
from tqdm import tqdm
import collections as clt
import os
import re
import json
import numpy as np
import shutil

from data_utils import TextMelLoader
from hparams import create_hparams

hp = create_hparams()

metadata_path = None
text_mel_loader = None
output_dir = None


def format_index(index):
    return '{:06d}'.format(index)


def process_one(index, skip_existing=False):
    global text_mel_loader
    global metadata_path
    global output_dir
    if text_mel_loader is None:
        text_mel_loader = TextMelLoader(metadata_path, hparams=hp, mode='preprocess')
        fpath = output_dir.joinpath('speaker_ids.json')
        speaker_ids = text_mel_loader.speaker_ids
        json.dump(speaker_ids, open(fpath, 'wt', encoding='utf8'), indent=4, ensure_ascii=False)

    onedir = output_dir.joinpath('npy', format_index(index))
    onedir.mkdir(exist_ok=True, parents=True)
    tpath = onedir.joinpath("text.npy")
    mpath = onedir.joinpath("mel.npy")
    spath = onedir.joinpath("speaker.npy")
    fpath = onedir.joinpath("f0.npy")

    if skip_existing and all([f.is_file() for f in [tpath, mpath, spath, fpath]]):
        return

    text, mel, speaker_id, f0 = text_mel_loader[index]

    np.save(tpath, text.numpy(), allow_pickle=False)
    np.save(mpath, mel.numpy(), allow_pickle=False)
    np.save(spath, speaker_id.numpy(), allow_pickle=False)
    np.save(fpath, f0.numpy(), allow_pickle=False)
    return index


def process_many(n_processes, skip_existing=False):
    # Embed the utterances in separate threads
    ids = list(range(len(text_mel_loader)))
    with open(output_dir.joinpath('train.txt'), 'wt', encoding='utf8') as fout:
        for num, idx in enumerate(tqdm(ids)):
            tmp = text_mel_loader.audiopaths_and_text[idx]
            fout.write('{}\t{}\n'.format(format_index(idx), '\t'.join(tmp).strip()))

    with open(output_dir.joinpath('validation.txt'), 'wt', encoding='utf8') as fout:
        val_ids = np.random.choice(ids, min(len(ids), hp.batch_size * 2), replace=False)
        for idx in tqdm(val_ids):
            tmp = text_mel_loader.audiopaths_and_text[idx]
            fout.write('{}\t{}\n'.format(format_index(idx), '\t'.join(tmp).strip()))

    if n_processes == 0:
        for index in tqdm(ids):
            try:  # 防止少数错误语音导致生成数据失败。
                process_one(index, skip_existing=skip_existing)
            except Exception as e:
                logger.info('Error! The <{}> audio load failed! {}'.format(index, e))
                tmp = text_mel_loader.audiopaths_and_text[index]
                logger.info('{}\t{}\n'.format(format_index(index), '\t'.join(tmp).strip()))
                logger.info('=' * 50)
    else:
        func = partial(process_one, skip_existing=skip_existing)
        job = Pool(n_processes).imap(func, ids)
        list(tqdm(job, "Embedding", len(ids), unit="utterances"))


if __name__ == "__main__":
    import argparse
    try:
        from setproctitle import setproctitle
        setproctitle('zhrtvc-mellotron-preprocess')
    except ImportError:
        pass


    parser = argparse.ArgumentParser(
        description="预处理训练数据，保存为numpy的npy格式，训练的时候直接从本地load数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--metadata_path", type=str, default=r'../../data/samples/metadata.csv',
                        help="metadata file path")
    # 每行数据格式：语音文件路径\t文本\t说话人名称\n，样例：aliaudio/Aibao/005397.mp3	他走近钢琴并开始演奏“祖国从哪里开始”。	aibao

    parser.add_argument("-o", "--output_dir", type=Path,
                        default=Path(r'../../data/SV2TTS/mellotron/samples'),
                        help="Path to the output directory")
    parser.add_argument("-n", "--n_processes", type=int, default=0,
                        help="Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", type=bool, default=True,
                        help="Whether to overwrite existing files with the same name. ")
    parser.add_argument("--hparams", type=str, default="",
                        help="Hyperparameter overrides as a comma-separated list of name-value pairs")
    args = parser.parse_args()

    metadata_path = args.metadata_path
    text_mel_loader = TextMelLoader(metadata_path, hparams=hp, mode='preprocess')

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    fpath = output_dir.joinpath('speaker_ids.json')
    speaker_ids = text_mel_loader.speaker_ids
    json.dump(speaker_ids, open(fpath, 'wt', encoding='utf8'), indent=4, ensure_ascii=False)

    # Preprocess the dataset
    process_many(args.n_processes, skip_existing=args.skip_existing)
