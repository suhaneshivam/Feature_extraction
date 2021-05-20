from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from helper import config
from imutils import paths
import numpy as np
import pickle
import random
import os

print("[INFO] loading network...")
model = VGG16(weights = "imagenet" ,include_top = False)
le = None

for split in (config.TRAIN ,config.TEST ,config.VAL):
    print("[INFO] processing '{} split'...".format(split))
    p = os.path.sep.join([config.BASE_PATH ,split])
    imagePaths = list(paths.list_images(p))
    random.seed(42)
    random.shuffle(imagePaths)

    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    if le is None:
        le = LabelEncoder()
        le.fit(labels)

    csvPath = os.path.sep.join([config.BASE_CSV_PATH ,"{}.csv".format(split)])
    csv = open(csvPath ,"w")

    for (b ,i) in enumerate(range(0 ,len(imagePaths) ,config.BATCH_SIZE)):
        print("[INFO] processing {}/{} batch".format(b+1 ,int(np.ceil(len(imagePaths)/float(config.BATCH_SIZE)))))
        batchPaths = imagePaths[i : i+config.BATCH_SIZE]
        batchLabels = le.transform(labels[i: i + config.BATCH_SIZE])
        batchImages = []

        for imagePath in batchPaths:
            image = load_img(imagePath ,target_size = (224 ,224))
            image = img_to_array(image)
            # we are adding the batch dim per in every image to preprocess because it expext a batch dim.
            image = np.expand_dims(image, axis = 0)

            image = preprocess_input(image)
            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages ,batch_size=config.BATCH_SIZE)
        features = features.reshape(features.shape[0] ,7*7*512)

        for (label ,vec) in zip(batchLabels ,features):
            vec = ",".join([str(v) for v in vec])
            csv.write("{},{}\n".format(label ,vec))
    csv.close()

f = open(config.LE_PATH ,"wb")
f.write(pickle.dumps(le))
f.close()
