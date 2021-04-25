from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys

sys.path.append(str(Path(__file__).absolute().parent.parent))

import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa
import hashlib
import traceback
from pathlib import Path

from mellotron import layers
from mellotron.utils import load_wav_to_torch, load_filepaths_and_text, load_filepaths_and_text_train
from mellotron.text import text_to_sequence, cmudict
from mellotron.yin import compute_yin

from encoder import inference as encoder


def transform_embed(wav, encoder_model_fpath=Path()):
    # from encoder import inference as encoder
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    wav_ = encoder.preprocess_wav(wav)
    # Take segment
    segment_length = 2 * encoder.sampling_rate  # 随机选取2秒语音生成语音表示向量
    if len(wav_) > segment_length:
        max_audio_start = len(wav_) - segment_length
        audio_start = random.randint(0, max_audio_start)
        wav_ = wav_[audio_start:audio_start + segment_length]

    embed = encoder.embed_utterance(wav_)
    return embed


def transform_text(text, text_cleaners):
    return text_to_sequence(text, text_cleaners)


def transform_mel(wav, stft=None):
    audio_norm = torch.FloatTensor(wav[None].astype(np.float32))
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec.cpu().numpy()


def transform_speaker(speaker, speaker_ids=None):
    if speaker_ids is None:
        # 一个说话人名字对应唯一的一个说话人向量
        hex_idx = hashlib.md5(speaker.encode('utf8')).hexdigest()
        out = (np.array([int(w, 16) for w in hex_idx])[None] - 7) / 10  # -0.7~0.8
        return out
    else:
        speaker_ids = speaker_ids or {}
        out = np.array([speaker_ids.get(speaker, 0)])
        return out


def transform_f0(wav, hparams):
    sampling_rate = hparams.sampling_rate
    frame_length = hparams.filter_length
    hop_length = hparams.hop_length
    f0_min = hparams.f0_min
    f0_max = hparams.f0_max
    harm_thresh = hparams.harm_thresh

    f0, harmonic_rates, argmins, times = compute_yin(
        wav, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad

    f0 = np.array(f0, dtype=np.float32)
    return f0


def transform_data_train(hparams, text_data, mel_data, speaker_data, f0_data, embed_data=None):
    """
    把数据转为训练需要的形式，模式控制。
    """
    tmp = hparams.train_mode.split('-')
    if tmp[0] == 'train':
        if len(tmp) == 2:
            mode = tmp[1]
        else:
            mode = True
    else:
        mode = False

    if isinstance(text_data, np.ndarray):
        text = torch.from_numpy(text_data)  # (86,)
    else:
        text = text_data

    if isinstance(mel_data, np.ndarray):
        mel = torch.from_numpy(mel_data)  # (80, 397)
    else:
        mel = mel_data

    if isinstance(speaker_data, np.ndarray):
        speaker = speaker_data  # (1,)
    else:
        speaker = speaker_data.cpu().numpy()

    if isinstance(f0_data, np.ndarray):
        f0 = f0_data  # (1, 395)
    elif f0_data is not None:
        f0 = f0_data.cpu().numpy()

    if isinstance(embed_data, np.ndarray):
        embed = embed_data  # (256,)
    elif embed_data is not None:
        embed = embed_data.cpu().numpy()

    if mode == 'f01':
        # 用f0数据。
        f0 = f0[:, :mel.shape[1]]
    elif mode == 'f02':
        # 用f0的均值代替f0，简化f0。
        f0 = f0.flatten()
        f0_value = np.mean(f0[f0 > 10])
        f0 = np.ones((1, mel.shape[1])) * f0_value
    elif mode == 'f03':
        # 用零向量填充f0。
        f0 = np.zeros((1, mel.shape[1]))
    elif mode == 'f04':
        # 不用f0。
        f0 = None
    elif mode == 'f05s02':
        # 音色控制，用发音人id，等距分配，speaker_id设置为0。
        f0_value = speaker[0] / hparams.n_speakers
        f0 = np.ones((1, mel.shape[1])) * f0_value
        speaker = speaker * 0
    elif mode == 'f06s02':
        # 音色控制，用降维的音频表示向量控制音色，speaker_id设置为0。
        embed = embed[::embed.shape[0] // hparams.prenet_f0_dim]
        embed = embed if embed.shape[0] == hparams.prenet_f0_dim else embed[:hparams.prenet_f0_dim]
        f0 = np.tile(embed, (mel.shape[1], 1)).T
        speaker = speaker * 0
    elif mode == 'gst':
        f0 = None
        speaker = speaker * 0
    elif mode == 'tacotron':
        f0 = None
        speaker = speaker * 0
    elif mode == 'mspk':
        # 发音人用md5的数字生成的向量表示，不用f0。
        f0 = None
    elif mode == 'rtvc':
        f0 = None
    else:
        # 默认：不用f0。
        f0 = None

    if isinstance(f0, np.ndarray):
        f0 = torch.from_numpy(f0)
    if isinstance(speaker, np.ndarray):
        speaker = torch.from_numpy(speaker)
    return (text, mel, speaker, f0)


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, speaker_ids=None, mode='train'):
        self.hparams = hparams
        tmp = mode.split('-')
        if tmp[0] == 'train':
            self.audiopaths_and_text = load_filepaths_and_text_train(audiopaths_and_text, split='\t')
            if len(tmp) == 2:
                self.mode = tmp[1]
            else:
                self.mode = True
        else:
            if isinstance(audiopaths_and_text, (str, Path)) and os.path.isfile(audiopaths_and_text):
                self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text, split='\t')
            else:
                self.audiopaths_and_text = ['audiopath', 'text', 'speaker']
            self.mode = False
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.p_arpabet = hparams.p_arpabet
        self.max_decoder_steps = hparams.max_decoder_steps

        self.f0_dim = hparams.prenet_f0_dim  # f0的维度设置
        self.encoder_model_fpath = hparams.encoder_model_fpath

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.speaker_ids = speaker_ids

        if self.speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

        # random.seed(1234)
        # random.shuffle(self.audiopaths_and_text)

        self.ids = set(range(len(self.audiopaths_and_text)))

    def get_data_train(self, data_dir):
        onedir = Path(data_dir)
        tpath = onedir.joinpath("text.npy")
        text_data = np.load(tpath)
        mpath = onedir.joinpath("mel.npy")
        mel_data = np.load(mpath)
        spath = onedir.joinpath("speaker.npy")
        speaker_data = np.load(spath)
        fpath = onedir.joinpath("f0.npy")
        f0_data = np.load(fpath)
        epath = onedir.joinpath("embed.npy")
        if epath.is_file():
            embed_data = np.load(epath)
        else:
            embed_data = None

        out = transform_data_train(
            hparams=self.hparams,
            text_data=text_data,
            mel_data=mel_data,
            speaker_data=speaker_data,
            f0_data=f0_data,
            embed_data=embed_data)
        return out

    def get_data_train_v2(self, data_dir):
        (text_data, mel_data, speaker_data, f0_data) = self.get_data(data_dir)
        assert mel_data.shape[1] < self.max_decoder_steps
        embed_data = np.zeros(256)  # 临时
        out = transform_data_train(
            hparams=self.hparams,
            text_data=text_data,
            mel_data=mel_data,
            speaker_data=speaker_data,
            f0_data=f0_data,
            embed_data=embed_data)
        return out

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[-1] if len(x) >= 3 else '0' for x in audiopaths_and_text]))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        # f0 = f0[:, :melspec.size(1)]

        # 用零向量替换F0
        # f0 = torch.zeros(1, melspec.shape[1], dtype=torch.float)
        # return melspec, f0
        return f0

    def get_embed(self, wav):
        # from encoder import inference as encoder
        if not encoder.is_loaded():
            encoder.load_model(self.encoder_model_fpath, device='cpu')
            # 用cpu避免以下报错。
            # "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the ‘spawn’ start method"

        wav_ = encoder.preprocess_wav(wav)
        # Take segment
        segment_length = 2 * encoder.sampling_rate  # 随机选取2秒语音生成语音表示向量
        if len(wav_) > segment_length:
            max_audio_start = len(wav_) - segment_length
            audio_start = random.randint(0, max_audio_start)
            wav_ = wav_[audio_start:audio_start + segment_length]

        embed = encoder.embed_utterance(wav_)
        return embed

    def get_data(self, audiopath_and_text):
        audiopath, text, speaker = audiopath_and_text

        text = self.get_text(text)

        audio_norm, sampling_rate = load_wav_to_torch(audiopath, sr_force=self.stft.sampling_rate)
        audio = audio_norm * self.max_wav_value
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))

        mel = self.get_mel(audio_norm)

        if self.hparams.prenet_f0_dim > 0:
            f0 = self.get_f0(audio.cpu().numpy().astype('int16'), sampling_rate,
                             self.filter_length, self.hop_length, self.f0_min,
                             self.f0_max, self.harm_thresh)
            f0 = torch.from_numpy(f0)[None]
        else:
            f0 = None

        if self.hparams.train_mode.endswith('rtvc'):
            embed = self.get_embed(audio_norm.cpu().numpy())
            speaker = torch.from_numpy(embed)[None]
        else:
            speaker = self.get_speaker(speaker)

        return (text, mel, speaker, f0)

    def get_speaker(self, speaker):
        if self.hparams.train_mode.endswith('mspk'):
            # 一个说话人名字对应唯一的一个说话人向量
            hex_idx = hashlib.md5(speaker.encode('utf8')).hexdigest()
            out = (np.array([int(w, 16) for w in hex_idx])[None] - 7) / 10
            return torch.FloatTensor(out)
        else:
            return torch.IntTensor([self.speaker_ids[speaker]])

    def get_mel(self, wav):
        audio_norm = wav.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        # melspec = linearspectrogram_torch(audio_norm)  # 用aukit的频谱生成方案
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners))  # self.cmudict, self.p_arpabet))

        return text_norm

    def __getitem__(self, index):
        if self.mode:
            tmp = index
            if tmp not in self.ids:
                tmp = np.random.choice(list(self.ids))
            while True:
                try:  # 模型训练模式容错。
                    # out = self.get_data_train(self.audiopaths_and_text[tmp][0])
                    out = self.get_data_train_v2(self.audiopaths_and_text[tmp])
                    # if tmp != index:
                    #     logger.info(
                    #         'The index <{}> loaded success! <Train>\n{}\n'.format(tmp, '-' * 50))
                    return out
                except:
                    logger.info(
                        'The index <{}> loaded failed! <Train>'.format(index, tmp))
                    traceback.print_exc()
                    self.ids.discard(tmp)
                    tmp = np.random.choice(list(self.ids))
        else:
            try:  # 数据预处理模式容错。
                out = self.get_data(self.audiopaths_and_text[index])
                return out
            except Exception as e:
                logger.info('The index <{}> loaded failed! <Preprocess>'.format(index))
                traceback.print_exc()
                return

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step, mode='train'):
        self.n_frames_per_step = n_frames_per_step
        self.train_mode = mode == 'train'

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # 推理模式不要排序
        if not self.train_mode:
            input_lengths = torch.LongTensor([len(x[0]) for x in batch])
            ids_sorted_decreasing = list(range(len(batch)))

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        try:
            num_f0s = batch[0][3].size(0)  # 获取f0s的维度。
        except:
            num_f0s = 1
        try:
            num_speaker_ids = batch[0][2].size(1)  # 获取num_speaker_ids的维度。
        except:
            num_speaker_ids = 0

        # include mel padded, gate padded and speaker ids
        # mel频谱pad很小的负数才是静音，pad数字0是很大声的噪声
        # 如果pad为很小的负数，训练不起来，原因未知
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        # mel_padded = torch.ones_like(mel_padded) * -10
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        if num_speaker_ids == 0:
            speaker_ids = torch.LongTensor(len(batch))
        else:
            speaker_ids = torch.FloatTensor(len(batch), num_speaker_ids)

        f0_padded = torch.FloatTensor(len(batch), num_f0s, max_target_len)
        f0_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            f0 = batch[ids_sorted_decreasing[i]][3]
            if isinstance(f0, torch.Tensor):
                f0_padded[i, :, :f0.size(1)] = f0
            else:
                f0_padded = f0

        # fixme 为了推理能够用batch
        input_lengths = torch.ones_like(input_lengths) * max_input_len

        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded,
                        output_lengths, speaker_ids, f0_padded)
        return model_inputs
