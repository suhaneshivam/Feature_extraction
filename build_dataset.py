from imutils import paths
import config
import os
import shutil

for split in (config.TRAIN,config.TEST ,config.VAL):
    print("[INFO] Building {} dataset".format(split))
    split_path = os.path.sep.join([config.ORIG_INPUT_DATASET ,split])
    imagePaths = list(paths.list_images(split_path))

    for path in imagePaths:
        filename = path.split(os.path.sep)[-1]
        label = int(filename.split("_")[0])

        new_split_path = os.path.sep.join([config.BASE_PATH ,split ,config.CLASSES[label]])

        if not os.path.exists(new_split_path):
            os.makedirs(new_split_path)

        p = os.path.sep.join([new_split_path ,filename])
        shutil.copy2(path ,p)
