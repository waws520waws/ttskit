"""
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
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sdk_api
import cli_api
import web_api
import encoder
import mellotron
import waveglow
import resource
from sdk_api import tts_sdk as tts

__version__ = "0.1.1"

version_doc = """
### 版本
v{}
""".format(__version__)

history_doc = """
### 历史版本

#### v0.1.0
- 初始版。
"""

readme_docs = [__doc__, version_doc,
               sdk_api.__doc__, cli_api.__doc__, web_api.__doc__,
               resource.__doc__, encoder.__doc__, mellotron.__doc__, waveglow.__doc__,
               history_doc]
