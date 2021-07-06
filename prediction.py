from joblib import load
from urllib import request
from PIL import Image
import numpy as np
model = load('./saved_models/model.joblib')
image_path = './input_images/input_image.jpg'

def prediction_function(link):
    request.urlretrieve(link, image_path)
    input_file = Image.open(open(image_path, 'rb')).convert('L')
    input_image = input_file.resize((28,28))
    data = np.array(input_image)
    image_value = data.reshape(784)
    y_pred = model.predict([image_value])
    for i in y_pred:
        str(i)
    return i

# y_pred = prediction('https://i.stack.imgur.com/RdEpj.png')
# print(y_pred)