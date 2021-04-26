# 中文语音克隆

## 使用指引

主要做语音合成器Mellotron，声码器WaveGlow。

新版GMW版本的语音克隆框架，用ge2e(encoder)-mellotron-waveglow的模块（简称GMW），运行更简单，效果更稳定和合成语音更加优质。

旧版ESV版本是基于项目Real-Time-Voice-Cloning改造为中文支持的版本，用encoder-synthesizer-vocoder的模块（简称ESV），运行比较复杂。

建议使用GMW版本开发，本项目重点维护GMW版本。

### 容器环境Docker
镜像基于ubuntu18.04，python环境是python3.7版本，用anaconda的环境。
必要依赖已经安装好，TensorFlow和Torch可以根据自己的实际情况安装。

```
# 执行路径为Dockerfile文件所在目录的路径
# 构建镜像
sudo docker build -t ubuntu/zhrtvc .

# 打开交互环境的容器
# 用-v参数设置挂载数据路径
sudo docker run -it -v [current absulte dir path]:/home/zhrtvc ubuntu/zhrtvc
```

### 安装依赖环境

建议用python3.7的环境，requirements.txt所列依赖支持于python3.7环境。

建议用zhrtvc/makefile.py来安装依赖包，如果有依赖包没有成功安装，再单独处理不能成功安装的依赖包。

执行：

```
python makefile.py

或者：
python makefile.py [requirement.txt的路径]

注意：默认安装CPU版本torch：
pip install torch==1.7.0+cpu torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

注意：

- GMW版本的依赖：requirements_gmw.txt
- ESV版本的依赖：requirements_esv.txt
- 全部版本适用的依赖：requirements.txt


### 语音合成器mellotron

#### 1. 处理语料。

生成用于训练mellotron的数据。

- **语料格式**

```markdown
|--datasets_root
   |--dataset1
      |--audio_dir1
      |--audio_dir2
      |--metadata.csv
```

- **metadata.csv**

一行描述一个音频文件。

每一行的数据格式：

```markdown
音频文件相对路径\t文本内容\n
```


- 例如：

```markdown
aishell/S0093/BAC009S0093W0368.mp3  有 着 对 美 和 品质 感 执着 的 追求
```

- 注意：

文本可以是汉字、拼音，汉字可以是分词后的汉字序列。

#### 2. 训练mellotron模型。

用处理好的数据训练mellotron的模型。

```
执行：

python mellotron_train.py

说明：

usage: mellotron_train.py [-h] [-i INPUT_DIRECTORY] [-o OUTPUT_DIRECTORY]
                          [-l LOG_DIRECTORY] [-c CHECKPOINT_PATH]
                          [--warm_start] [--n_gpus N_GPUS] [--rank RANK]
                          [--group_name GROUP_NAME]
                          [--hparams_json HPARAMS_JSON]
                          [--hparams_level HPARAMS_LEVEL] [--cuda CUDA]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIRECTORY, --input_directory INPUT_DIRECTORY
                        directory to save checkpoints
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory to save checkpoints
  -l LOG_DIRECTORY, --log_directory LOG_DIRECTORY
                        directory to save tensorboard logs
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        checkpoint path
  --warm_start          load model weights only, ignore specified layers
  --n_gpus N_GPUS       number of gpus
  --rank RANK           rank of current gpu
  --group_name GROUP_NAME
                        Distributed group name
  --hparams_json HPARAMS_JSON
                        comma separated name=value pairs
  --hparams_level HPARAMS_LEVEL
                        hparams scale
  --cuda CUDA           设置CUDA_VISIBLE_DEVICES
```

- 注意

如果多个数据一起用，可以用绝对路径表示，汇总到一个metadata.csv文件，便于训练。


#### 3. 应用mellotron模型。

```markdown
执行：
python mellotron_inference.py

说明：
usage: mellotron_inference.py [-h] [-m CHECKPOINT_PATH]
                              [--is_simple IS_SIMPLE] [-s SPEAKER_PATH]
                              [-a AUDIO_PATH] [-t TEXT_PATH] [-o OUT_DIR]
                              [-p PLAY] [--n_gpus N_GPUS]
                              [--hparams_path HPARAMS_PATH]
                              [-e ENCODER_MODEL_FPATH]
                              [--save_model_path SAVE_MODEL_PATH]
                              [--cuda CUDA]

optional arguments:
  -h, --help            show this help message and exit
  -m CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        模型路径。
  --is_simple IS_SIMPLE
                        是否简易模式。
  -s SPEAKER_PATH, --speaker_path SPEAKER_PATH
                        发音人映射表路径。
  -a AUDIO_PATH, --audio_path AUDIO_PATH
                        参考音频路径。
  -t TEXT_PATH, --text_path TEXT_PATH
                        文本路径。
  -o OUT_DIR, --out_dir OUT_DIR
                        保存合成的数据路径。
  -p PLAY, --play PLAY  是否合成语音后自动播放语音。
  --n_gpus N_GPUS       number of gpus
  --hparams_path HPARAMS_PATH
                        comma separated name=value pairs
  -e ENCODER_MODEL_FPATH, --encoder_model_fpath ENCODER_MODEL_FPATH
                        Path your trained encoder model.
  --save_model_path SAVE_MODEL_PATH
                        保存模型为可以直接torch.load的格式
  --cuda CUDA           设置CUDA_VISIBLE_DEVICES

```

### 语音合成器waveglow

#### 1. 处理语料。

生成用于训练waveglow的数据。

方法同处理mellotron的数据方法。

因为训练声码器只需要音频即可，不需要文本和发音人的标注，故可以任意指定文本和发音人，格式如训练mellotron的数据格式即可。

#### 2. 训练waveglow模型。

```
执行：
python waveglow_train.py

说明：
usage: waveglow_train.py [-h] [-c CONFIG] [-r RANK] [-g GROUP_NAME]
                         [--cuda CUDA]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        JSON file for configuration
  -r RANK, --rank RANK  rank of process for distributed
  -g GROUP_NAME, --group_name GROUP_NAME
                        name of group for distributed
  --cuda CUDA           Set CUDA_VISIBLE_DEVICES

```

#### 3. 应用waveglow模型。
```
执行：
python waveglow_inference.py

说明：
usage: waveglow_train.py [-h] [-c CONFIG] [-r RANK] [-g GROUP_NAME]
                         [--cuda CUDA]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        JSON file for configuration
  -r RANK, --rank RANK  rank of process for distributed
  -g GROUP_NAME, --group_name GROUP_NAME
                        name of group for distributed
  --cuda CUDA           Set CUDA_VISIBLE_DEVICES

(base) E:\github-kuangdd\zhrtvc\zhrtvc>python waveglow_inference.py --help
usage: waveglow_inference.py [-h] [-w WAVEGLOW_PATH] [--is_simple IS_SIMPLE]
                             [-i INPUT_PATH] [-o OUTPUT_PATH] [-c CONFIG_PATH]
                             [--kwargs KWARGS] [--cuda CUDA]
                             [--save_model_path SAVE_MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -w WAVEGLOW_PATH, --waveglow_path WAVEGLOW_PATH
                        Path to waveglow decoder checkpoint with model
  --is_simple IS_SIMPLE
                        是否简易模式。
  -i INPUT_PATH, --input_path INPUT_PATH
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
  -c CONFIG_PATH, --config_path CONFIG_PATH
  --kwargs KWARGS       Waveglow kwargs json
  --cuda CUDA           Set CUDA_VISIBLE_DEVICES
  --save_model_path SAVE_MODEL_PATH
                        Save model for torch load

```

## 参考项目

- **Real-Time Voice Cloning**
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. Feel free to check [my thesis](https://matheo.uliege.be/handle/2268.2/6801) if you're curious or if you're looking for info I haven't documented yet (don't hesitate to make an issue for that too). Mostly I would recommend giving a quick look to the figures beyond the introduction.

SV2TTS is a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.

**Video demonstration** (click the picture):

[![Toolbox demo](https://i.imgur.com/Ixy13b7.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |

## Wiki
- **How it all works** (coming soon!)
- [**Training models yourself**](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training)
- **Training with other data/languages** (coming soon! - see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/30#issuecomment-507864097) for now)
- [**TODO and planned features**](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/TODO-&-planned-features) 

## 版本记录

### v1.4.23
