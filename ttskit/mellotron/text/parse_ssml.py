# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/9/23
"""
parse_ssml

SSML格式：
1.文本首尾分别是：<speak>、</speak>
2.拼音标注格式：<phoneme alphabet="py" ph="pin1 yin1">拼音</phoneme>
3.样例：
<speak><phoneme alphabet="py" ph="gan4 ma2 a5 ni3">干嘛啊你</phoneme><phoneme alphabet="py" ph="you4 lai2">又来</phoneme><phoneme alphabet="py" ph="gou1 da5 shei2">勾搭谁</phoneme>。</speak>
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import re
import json

_speak_re = re.compile(r'^<speak>(.+?)</speak>$')
_phoneme_re = re.compile(r'<phoneme alphabet="py" ph="(.+?)">(.+?)</phoneme>')
_none_pinyin = None
_default_errors = lambda x: x

def convert_ssml(ssml_text, errors=None):
    """把ssml格式的文本转为汉字拼音列表。"""
    if errors is None:
        errors = _default_errors
    han_lst, pin_lst = [], []
    f = _speak_re.search(ssml_text)
    if f:
        s = f.group(1)
        si = 0
        for w in _phoneme_re.finditer(s):
            se = w.span()[0]
            if se > si:
                han_lst.extend(s[si: se])
                pin_lst.extend([errors(p) for p in s[si: se]])
            si = w.span()[1]

            p = w.group(1).split()
            h = list(w.group(2))
            assert len(p) == len(h)
            pin_lst.extend(p)
            han_lst.extend(h)
        else:
            se = len(s)
            if se > si:
                han_lst.extend(s[si: se])
                pin_lst.extend([errors(p) for p in s[si: se]])
    else:
        for w in ssml_text:
            han_lst.append(w)
            pin_lst.append(errors(w))
    outs = [(h, p) for h, p in zip(han_lst, pin_lst)]
    return outs


def parse_ssml(ssml_text):
    """解析ssml，用OrderDict不合适。"""
    import xmltodict
    outs = []

    def deep_parse(obj):
        if isinstance(obj, dict):
            if '#text' in obj:
                han_lst = list(obj['#text'])
                if '@ph' in obj:
                    pin_lst = obj['@ph'].split()
                    assert len(han_lst) == len(pin_lst)
                    for h, p in zip(han_lst, pin_lst):
                        outs.append((h, p))
                else:
                    for h in han_lst:
                        outs.append((h, None))

        if isinstance(obj, list):
            for w in obj:
                if isinstance(w, dict):
                    deep_parse(w)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    deep_parse(v)

    dt = xmltodict.parse(ssml_text)
    print(json.dumps(dt, indent=4, ensure_ascii=False))
    deep_parse(dt)
    for k, v in outs:
        print(k, v)


if __name__ == "__main__":
    print(__file__)
    ssml_text = '<speak>实验<phoneme alphabet="py" ph="gan4 ma2 a5 ni3">干嘛啊你</phoneme><sub alias="??">？？</sub><phoneme alphabet="py" ph="you4 lai2">又来</phoneme><phoneme alphabet="py" ph="gou1 da5 shei2">勾搭谁</phoneme>。</speak>'
    text = '你好。'
    outs = convert_ssml(ssml_text)
    print(outs)
    outs = convert_ssml(text)
    print(outs)

    from phkit.chinese import text_to_sequence, sequence_to_text
    zp_lst = convert_ssml(ssml_text)
    pin_text = ' '.join([p for z, p in zp_lst])
    seq = text_to_sequence(pin_text, cleaner_names='pinyin')
    print(sequence_to_text(seq))
