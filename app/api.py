from flask import Flask, jsonify, request
from PIL import Image, ImageDraw
from io import BytesIO
import base64

import numpy as np


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

    generated_images = []
    buffer = BytesIO()
    for _ in range(gen_num):
        latent, image = generate_figure()
        generated_images.append({
            'latent': str(latent.tolist()),
            'base64': img_to_base64(image)
        })

    return jsonify(generated_images)


@app.route('/mix')
def style_mix():
    latent1 = request.args.get('latent1')
    latent2 = request.args.get('latent2')

    # TODO: Style mixing
    mixed_images = mix_styles(latent1, latent2)

    return jsonify(list(map(img_to_base64, mixed_images)))


if __name__ == "__main__":
    app.run(debug=True)
