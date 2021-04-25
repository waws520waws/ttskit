# author: kuangdd
# date: 2021/4/23
"""
### web_api
语音合成WEB接口。
构建简单的语音合成服务。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

from flask import Flask, request, jsonify, Response

import sdk_api

app = Flask(__name__)


def parse_request(req_data):
    """解析请求数据并以json形式返回"""
    if req_data.method == 'POST':
        data = req_data.json
    elif req_data.method == 'GET':
        data = req_data.args
    else:  # POST
        data = req_data.get_json()
    return data


@app.route('/tts', methods=['POST', 'GET'])
def index():
    data = parse_request(request)
    text = data.get('text', '这是个样例')
    speaker = data.get('speaker', 'biaobei')
    audio = data.get('audio', '0')
    wav = sdk_api.tts_sdk(text=text, speaker=speaker, audio=audio)
    return Response(wav, mimetype='audio/wav')


if __name__ == "__main__":
    logger.info(__file__)
    app.run(host='0.0.0.0', port=2718, debug=False)
