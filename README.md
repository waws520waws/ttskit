

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

v0.1.0



### sdk_api

语音合成SDK接口。

本地函数式地调用语音合成。



### cli_api

语音合成命令行接口。

用命令行调用语音合成。



### web_api

语音合成WEB接口。

构建简单的语音合成服务。



### resource

模型数据等资源。

__init__.py

audio

model

reference_audio

reference_audio.tar



### encoder

声音编码器。



### mellotron

语音合成器。



### waveglow

声码器。



### 历史版本



#### v0.1.0

- 初始版。

