# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
import yaml
import traceback
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

# =====START: ADDED FOR DISTRIBUTED======
from .distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
# =====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from .glow import WaveGlow, WaveGlowLoss
from .mel2samp import Mel2Samp

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def keep_n_checkpoints(info_path, checkpoint_info, n_keep):
    """
    一个txt文本文件记录保存的checkpoint路径和必要信息。
    删除指定参数最小或最大的checkpoint。
    白名单规则额外保留。
    """
    if os.path.isfile(info_path):
        info_lst = yaml.load(open(info_path, encoding='utf8'))
    else:
        info_lst = []
    info_lst.insert(0, checkpoint_info)
    while len(info_lst) > n_keep:
        info_rm = info_lst.pop()
        path_rm = os.path.join(os.path.dirname(info_path), info_rm['name'])
        if os.path.isfile(path_rm):
            os.remove(path_rm)
    yaml.dump(info_lst, open(info_path, 'wt', encoding='utf8'))


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    try:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    except:
        traceback.print_exc()

    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, waveglow_config):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).to(_device)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard, waveglow_config, dist_config, data_config, train_config, **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    # =====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config).to(_device)

    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    # =====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1
    iteration_start = iteration
    trainset = Mel2Samp(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset,
                              num_workers=train_config.get('dataloader_num_workers', 8),
                              shuffle=train_config.get('dataloader_shuffle', True),
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=train_config.get('dataloader_pin_memory', False),
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(
            os.path.join(os.path.dirname(output_directory), 'tensorboard'),
            filename_suffix='.tensorboard')

    with open(Path(output_directory).parent.joinpath('metadata', 'train.txt'), 'wt', encoding='utf8') as fout:
        for line in trainset.audio_files:
            fpath = os.path.abspath(line)
            fout.write(f'{fpath}\n')

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch-{epoch}", ncols=100)):
            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.to(_device))
            audio = torch.autograd.Variable(audio.to(_device))
            outputs = model((mel, audio))

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # print("{}:\t{:.9f}".format(iteration, reduced_loss))
            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            if (iteration % iters_per_checkpoint == 0) or (iteration == iteration_start):
                if rank == 0:
                    checkpoint_path = "{}/waveglow-{:06d}.pt".format(output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, waveglow_config=waveglow_config)

                    info_path = os.path.join(output_directory, 'info.yml')
                    checkpoint_info = {'name': os.path.basename(checkpoint_path),
                                       'iteration': iteration,
                                       'loss': reduced_loss}
                    keep_n_checkpoints(info_path, checkpoint_info, 5)

                    if with_tensorboard:
                        # outputs[0].shape: torch.Size([1, 8, 1000])
                        with torch.no_grad():
                            d = model.infer(mel.data[0].unsqueeze(0), sigma=sigma)
                            d = d.cpu().squeeze()
                            pred_audio = (d - d.min()) * 1.98 / (d.max() - d.min()) - 0.99

                            logger.add_audio(
                                "generated/iteration-{}.wav".format(iteration),
                                pred_audio,
                                iteration,
                                sample_rate=trainset.sampling_rate,
                            )

                            true_audio = audio.data[0].squeeze()
                            logger.add_audio(
                                "original/iteration-{}.wav".format(iteration),
                                true_audio,
                                iteration,
                                sample_rate=trainset.sampling_rate,
                            )

                            # 查看频谱，直观了解生成语音的情况
                            mel_output = trainset.get_mel(pred_audio.cpu())
                            logger.add_image(
                                "generated/iteration-{}.png".format(iteration),
                                plot_spectrogram_to_numpy(mel_output.data.cpu().numpy()),
                                iteration, dataformats='HWC')

                            mel_input = mel.data[0]
                            logger.add_image(
                                "original/iteration-{}.png".format(iteration),
                                plot_spectrogram_to_numpy(mel_input.data.cpu().numpy()),
                                iteration, dataformats='HWC')

            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    # global data_config
    data_config = config["data_config"]
    # global dist_config
    dist_config = config["dist_config"]
    # global waveglow_config
    waveglow_config = config["waveglow_config"]

    metadata_dir = Path(train_config["output_directory"]).parent.joinpath('metadata')
    metadata_dir.mkdir(exist_ok=True, parents=True)

    shutil.copyfile(args.config, metadata_dir.joinpath('config.json'))
    shutil.copyfile(data_config['training_files'], metadata_dir.joinpath('train.txt'))

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name,
          waveglow_config=waveglow_config,
          dist_config=dist_config,
          data_config=data_config,
          train_config=train_config,
          **train_config)
