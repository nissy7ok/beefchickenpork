import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys

import tensorflow as tf
def memory_limit():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")

memory_limit()


# パラメーターの初期化
classes = ["cow", "chicken", "pig"]
num_classes = len(classes)
image_size = 224

# 引数から画像ファイルを参照して読み込む
image = Image.open(sys.argv[1])
image = image.convert("RGB")
image = image.resize((image_size, image_size))
data = np.asarray(image) / 255.0
X = []
X.append(data)
X = np.array(X)

# モデルのロード
model = load_model('./vgg16_transfer.h5')

result = model.predict([X])[0]
predicted = result.argmax() # 最大値のインデックスの値を取る
percentage = int(result[predicted] * 100)

cow_res = str(round(result[0] * 100, 1))
chicken_res = str(round(result[1] * 100, 1))
pig_res = str(round(result[2] * 100, 1))

# print(classes[predicted], percentage, "%")
print("cow: " + cow_res + "%\n" +
    "chicken: " + chicken_res + "%\n" +
    "pig: " + pig_res + "%")