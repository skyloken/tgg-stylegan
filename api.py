from flask import Flask, jsonify, request
from PIL import Image, ImageDraw
from io import BytesIO
import base64

import numpy as np

from service import TakedaGoichiGenerator

# tgg = TakedaGoichiGenerator()

def generate_figure():
    """mock"""
    latent = np.random.randn(512)
    image = Image.new("RGB", (256, 256), (256, 0, 256))

    return latent, image


def mix_styles(latent1, latent2):
    """mock"""

    mix_num = 30
    mixed_images = [Image.new(
        "RGB", (256, 256), (0, 0, i * int((256 / (mix_num - 1))))) for i in range(mix_num)]

    return mixed_images


app = Flask(__name__)


def img_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/generate')
def generate():
    gen_num = int(request.args.get('n'))
    style = request.args.get('style')
    label = [1, 0] if style == 'western' else [0, 1]

    generated_images = []
    buffer = BytesIO()
    for _ in range(gen_num):
        latent, image = generate_figure()
        # latent, image = tgg.generate_figure()
        generated_images.append({
            'latent': base64.b64encode(latent.tobytes()).decode('utf-8'),
            'base64': img_to_base64(image)
        })

    return jsonify(generated_images)


@app.route('/mix', methods=['POST'])
def style_mix():
    body = request.get_json()
    wst_latent = np.frombuffer(base64.b64decode(body['wstLatent']))
    jpn_latent = np.frombuffer(base64.b64decode(body['jpnLatent']))

    # Style mixing
    mixed_images = mix_styles(wst_latent, jpn_latent)
    # mixed_images = tgg.mix_styles(wst_latent, jpn_latent)

    return jsonify(list(map(img_to_base64, mixed_images)))


if __name__ == "__main__":
    app.run(debug=True)
