from flask import Flask, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
from io import BytesIO
import base64


def getGraph():
    width = 1200
    height = 400
    graph = Image.new("RGB", (width, height), (256, 0, 256))  # 画像オブジェクトの生成

    buffer = BytesIO()
    graph.save(buffer, format='png')

    base64Img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64Img


app = Flask(__name__)
CORS(app)


@app.route('/hello')
def hello_world():
    imageBase64 = getGraph()
    return jsonify({'imageBase64': imageBase64})

if __name__ == "__main__":
    app.run(debug=True)