# author: kuangdd
# date: 2021/4/25
"""
### resource
模型数据等资源。
__init__.py
audio
model
reference_audio
reference_audio.tar
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
    for line in sorted(Path(__file__).parent.glob('*')):
        print(line.name)
