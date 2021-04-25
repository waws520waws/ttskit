# author: kuangdd
# date: 2021/4/23
"""
### cli_api
语音合成命令行接口。
用命令行调用语音合成。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import argparse
import json

import sdk_api

mellotron_path = sdk_api._mellotron_path
waveglow_path = sdk_api._waveglow_path
ge2e_path = sdk_api._ge2e_path
mellotron_hparams_path = sdk_api._mellotron_hparams_path


def parse_args():
    parser = argparse.ArgumentParser(description='语音合成')
    parser.add_argument('-i', '--input', type=str, default="audio\ttext\tspeaker",
                        help='Input file path or text')
    parser.add_argument('-t', '--text', type=str, default="欢迎使用语音合成接口。",
                        help='Input text content')
    parser.add_argument('-s', '--speaker', type=str, default="biaobei",
                        help='Input speaker name')
    parser.add_argument('-a', '--audio', type=str, default="0",
                        help='Input audio path')
    parser.add_argument('-o', '--output', type=str, default='play',
                        help='Output audio path')
    parser.add_argument('-m', '--mellotron_path', type=str, default=mellotron_path,
                        help='Mellotron model file path')
    parser.add_argument('-w', '--waveglow_path', type=str, default=waveglow_path,
                        help='WaveGlow model file path')
    parser.add_argument('-g', '--ge2e_path', type=str, default=ge2e_path,
                        help='Ge2e model file path')
    parser.add_argument('--mellotron_hparams_path', type=str, default=mellotron_hparams_path,
                        help='Mellotron hparams json file path')
    parser.add_argument('--waveglow_kwargs_json', type=str, default=r'{"denoiser_strength":1,"sigma":1}',
                        help='Waveglow kwargs json')

    args = parser.parse_args()
    return args


def tts_cli(args):
    sdk_api.load_models(mellotron_path=args.mellotron_path,
                        waveglow_path=args.waveglow_path,
                        ge2e_path=args.ge2e_path,
                        mellotron_hparams_path=args.mellotron_hparams_path)
    waveglow_kwargs = json.loads(args.waveglow_kwargs_json)
    wav_output = sdk_api.tts_sdk(text=args.text,
                                 speaker=args.speaker,
                                 audio=args.audio,
                                 output=args.output,
                                 **waveglow_kwargs)
    return wav_output


if __name__ == "__main__":
    logger.info(__file__)
    sdk_api.download_resource()
    args = parse_args()
    tts_cli(args)
