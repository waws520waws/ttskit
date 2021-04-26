

![ttskit](ttskit.png "ttskit")



## ttskit

Text To Speech Toolkit: 语音合成工具箱。



### 安装



```

pip install -U ttskit

```



- 注意

    * 可能需另外安装的依赖包：torch, pyaudio, sounddevice。

    * pyaudio暂不支持python37以上版本直接pip安装，需要下载whl文件安装，下载路径：https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

    * sounddevice依赖pyaudio。

    * ttskit的默认音频采样率为22.5k。



### 快速使用

```

import ttskit



ttskit.tts('这是个示例', audio='1')

```



### 版本

v0.1.1



### sdk_api

语音合成SDK接口。

本地函数式地调用语音合成。



+ 简单使用

```python

from ttskit import sdk_api



wav = sdk_api.tts_sdk('文本', audio='1')

```



### cli_api

语音合成命令行接口。

用命令行调用语音合成。



+ 简单使用

```python

from ttskit import cli_api



args = cli_api.parse_args()

cli_api.tts_cli(args)

# 命令行交互模式使用语音合成。

```



+ 使用说明

```

usage: cli_api.py [-h] [-i INTERACTION] [-t TEXT] [-s SPEAKER] [-a AUDIO]

                  [-o OUTPUT] [-m MELLOTRON_PATH] [-w WAVEGLOW_PATH]

                  [-g GE2E_PATH]

                  [--mellotron_hparams_path MELLOTRON_HPARAMS_PATH]

                  [--waveglow_kwargs_json WAVEGLOW_KWARGS_JSON]



语音合成



optional arguments:

  -h, --help            show this help message and exit

  -i INTERACTION, --interaction INTERACTION

                        是否交互，如果1则交互，如果0则不交互。交互模式下：如果不输入文本或发音人，则为随机。如果输入文本为exit

                        ，则退出。

  -t TEXT, --text TEXT  Input text content

  -s SPEAKER, --speaker SPEAKER

                        Input speaker name

  -a AUDIO, --audio AUDIO

                        Input audio path or audio index

  -o OUTPUT, --output OUTPUT

                        Output audio path. 如果play开头，则播放合成语音；如果.wav结尾，则保存语音。

  -m MELLOTRON_PATH, --mellotron_path MELLOTRON_PATH

                        Mellotron model file path

  -w WAVEGLOW_PATH, --waveglow_path WAVEGLOW_PATH

                        WaveGlow model file path

  -g GE2E_PATH, --ge2e_path GE2E_PATH

                        Ge2e model file path

  --mellotron_hparams_path MELLOTRON_HPARAMS_PATH

                        Mellotron hparams json file path

  --waveglow_kwargs_json WAVEGLOW_KWARGS_JSON

                        Waveglow kwargs json

```





### web_api

语音合成WEB接口。

构建简单的语音合成服务。



+ 简单使用

```python

from ttskit import web_api



web_api.app.run(host='0.0.0.0', port=2718, debug=False)

# 用POST或GET方法请求：http://localhost:2718/tts，传入参数text、audio、speaker。

# 例如GET方法请求：http://localhost:2718/tts?text=这是个例子&audio=2

```



+ 使用说明



### resource

模型数据等资源。

audio

model

reference_audio


### encoder

声音编码器。



### mellotron

语音合成器。



### waveglow

声码器。



### 历史版本



#### v0.1.0

- 初始版。

