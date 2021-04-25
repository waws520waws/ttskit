![Mellotron](mellotron_logo.png "Mellotron")

### 语音克隆走起

#### 简化跑模型方法
直接跑train.py即可，传入metadata.csv文件，设置好参数即可跑起。


#### 解码端音色控制语音合成修改要点

* 命令行执行指令：
```commandline
# 生成语音表示向量
python preprocess_embed.py -i ../../data/SV2TTS/mellotron/aliaudio/train.txt -n 4

# 训练模型
python train.py -i ../../data/SV2TTS/mellotron/aliaudio -o ../../models/mellotron/aliaudio-f06s02 --hparams {\"batch_size\":64,\"iters_per_checkpoint\":5000}
```


* 把控制音色的部分放到解码端。
    + 用1维向量表示一个发音人，模仿F0的方法控制音色。
        - 每个发音人分配一个数字，可随机分配，可等距分配。
        - 给每个发音人找一个代表音频，音频转为向量，降到1维。
    + 用3维向量表示一个发音人，音色用颜色表示。
        - 随机，等距。
        - 代表音频转为向量，降维到3维。
        - 当前音频转为向量，按照特定方法降维到3维。
    + 用N维向量表示一个发音人，音色精确表示。
        - 代表音频转为向量，降维到N维。
        - 当前音频转为向量，按照特定方法降维到N维。
        - 当前音频转为向量，取前N维即可。

* 语音转为向量，降维。
    + 实验了基于投影和基于成分的降维方法，分别是等距映射、t-SNE、UMAP、主成分分析、奇异值分解、独立成分分析。
    + 基于投影的降维对向量之间距离的变换尺度很大，相似的更加相似，不相似的很不相似。
    + 基于成分的降维方法比较能反映原数据特点。
    + 训练集内的数据，降维到3维依然有效果。
    + 对训练集外的数据降维效果不好。
    + 用等距采样的方法降维，降维到8维能反映整体数据特点，降维到16维更加精准。

#### 多发音人语音合成修改要点

* 频谱生成采用aukit的方案的。

* 文本处理采用phkit的方案的。

* 基频F0方案修改。
    + 基频采用零向量。
        - 在训练的时候，传入和语音时长对应长度的零向量。
        - 在合成的时候，预估合成语音的时长，传入时长较长的零向量作为基频即可。
    + 在模型中去除基频模块。
        - 基频传入None类型，不用基频。
        - 合成的时候也传入None即可。
    + 基频取平均值作为代表。
        - 在训练的时候，取基频的平均值，用均值填充基频向量。
        - 在合成的时候，先预估说话人语音基频，生成基频向量传入。
        
* 说话人表示延用一个说话人一个向量的方法，使用mellotron的方案。

* 训练频谱采用线性频谱。
    + 用aukit的默认参数生成频谱。
    + Griffinlim声码器用线性频谱。
    + 神经网络的声码器用线性频谱转的mel频谱。

* 不用GST模块。

#### 使用方法

* 预处理语音和文本数据，生成训练模型的可用numpy直接load的文件。
    + 执行mellotron目录下的preprocess.py脚本，可命令行设置参数执行，或者修改默认参数执行。
    + 执行preprocess.py后，会生成存放频谱、文本编码等数据的目录，同时生成train.txt文本文件。

* 训练语音合成器模型。
    + 执行mellotron目录下的train.py脚本，可命令行设置参数执行，或者修改默认参数执行。
    + 执行train.py后，会从train.txt的文本读取数据，不断训练模型。

* 用训练好的语音合成器模型合成频谱，并合成语音。
    + 执行synthesize.py脚本，设置文本和说话人。
    + 用Griffinlim声码器合成语音。



### Rafael Valle\*, Jason Li\*, Ryan Prenger and Bryan Catanzaro
In our recent [paper] we propose Mellotron: a multispeaker voice synthesis model
based on Tacotron 2 GST that can make a voice emote and sing without emotive or
singing training data. 

By explicitly conditioning on rhythm and continuous pitch
contours from an audio signal or music score, Mellotron is able to generate
speech in a variety of styles ranging from read speech to expressive speech,
from slow drawls to rap and from monotonous voice to singing voice.

Visit our [website] for audio samples.

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/mellotron.git`
2. CD into this repo: `cd mellotron`
3. Initialize submodule: `git submodule init; git submodule update`
4. Install [PyTorch]
5. Install [Apex]
6. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. Update the filelists inside the filelists folder to point to your data
2. `python train.py --output_directory=outdir --log_directory=logdir`
3. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the speaker embedding layer is [ignored]

1. Download our published [Mellotron] model trained on LibriTTS
2. `python train.py --output_directory=outdir --log_directory=logdir -c models/mellotron_libritts.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. `jupyter notebook --ip=127.0.0.1 --port=31337`
2. Load inference.ipynb 
3. (optional) Download our published [WaveGlow](https://drive.google.com/open?id=1Rm5rV5XaWWiUbIpg5385l5sh68z2bVOE) model

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft), 
[Chengqi Deng](https://github.com/KinglittleQ/GST-Tacotron),
[Patrice Guyot](https://github.com/patriceguyot/Yin), as described in our code.

[ignored]: https://github.com/NVIDIA/mellotron/blob/master/hparams.py#L22
[paper]: https://arxiv.org/abs/1910.11997
[WaveGlow]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[Mellotron]: https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI
[pytorch]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/Mellotron
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
