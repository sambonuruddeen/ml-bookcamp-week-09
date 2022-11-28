import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x = x/255
    return x

def predict(url):
    target_size = (150,150)

    image = download_image(url)
    image = prepare_image(image, target_size=target_size)

    x = np.array(image, dtype='float32')
    X = np.array([x])

    X = preprocess_input(X)

    # Inference
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return preds
# Interpreter

interpreter = tflite.Interpreter(model_path='dino-dragon.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# Download & prepare Image
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'

pred = predict(url)

print(pred)
