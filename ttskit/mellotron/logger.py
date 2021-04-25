import random
import json
import torch
import numpy as np
from tensorboardX import SummaryWriter

from .plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from .plotting_utils import plot_gate_outputs_to_numpy
# from .utils import inv_linearspectrogram, default_hparams
from .text import sequence_to_text
from .layers import TacotronSTFT


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, hparams=None):
        super(Tacotron2Logger, self).__init__(logdir, max_queue=100, filename_suffix='.tensorboard')
        self.stft = TacotronSTFT(**{k: v for k, v in hparams.items()})

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration, x):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y
        text_inputs = x[0]
        speaker_ids = x[5]
        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        gate_target = gate_targets[idx].data.cpu().numpy()
        gate_output = torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_target,
                gate_output),
            iteration, dataformats='HWC')

        # 记录一下合成的语音效果。
        end_idx = np.argmax(gate_output > 0.5) or gate_output.shape[0]
        mel = mel_outputs[idx][:, :end_idx].unsqueeze(0)
        audio_predicted = self.stft.griffin_lim(mel)[0]
        self.add_audio(
            'audio_predicted',
            audio_predicted,
            iteration, sample_rate=self.stft.sampling_rate
        )
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')

        end_idx = np.argmax(gate_target > 0.5) or gate_target.shape[0]
        mel = mel_targets[idx][:, :end_idx].unsqueeze(0)
        audio_target = self.stft.griffin_lim(mel)[0]
        self.add_audio(
            'audio_target',
            audio_target,
            iteration, sample_rate=self.stft.sampling_rate
        )

        # spk = int(speaker_ids[idx].data.cpu().numpy().flatten()[0])
        spk = ' '.join([f'{w:.2g}' for w in speaker_ids[idx].data.cpu().numpy().flatten()])  # speaker_ids可能是向量
        ph_ids = text_inputs[idx].data.cpu().numpy().flatten()
        phs_text = sequence_to_text(ph_ids)
        phs_size = len(ph_ids)
        reduced_loss = float(reduced_loss)
        audt_duration = int(len(audio_target) / (self.stft.sampling_rate / 1000))
        audp_duration = int(len(audio_predicted) / (self.stft.sampling_rate / 1000))
        spect_shape = mel_targets[idx].data.cpu().numpy().shape
        specp_shape = mel_outputs[idx].data.cpu().numpy().shape
        align_shape = alignments[idx].data.cpu().numpy().T.shape
        out_text = dict(speaker=spk, phonemes=phs_text, phonemes_size=phs_size, validation_loss=reduced_loss,
                        audio_target_ms=audt_duration, audio_predicted_ms=audp_duration,
                        spectrogram_target_shape=str(spect_shape), spectrogram_predicted_shape=str(specp_shape),
                        alignment_shape=str(align_shape))
        out_text = json.dumps(out_text, indent=4, ensure_ascii=False)
        out_text = f'<pre>{out_text}</pre>'  # 支持html标签
        self.add_text(
            'text',
            out_text,
            iteration
        )
