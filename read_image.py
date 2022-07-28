import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50
from glob import glob as gl
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', metavar='N', type=str, nargs='+',
                    help='video path, must not ends with "/"')

parser.add_argument('--thresh', metavar='N', type=float, nargs='+', default=0.5,
                    help='threshold for classification default = 0.5')


def main(image_dir, thresh):
    resnet_backbone = ResNet50(weights="imagenet", include_top=False, pooling=None,
                               input_tensor=Input(shape=(224, 224, 3)))
    backbone_output = resnet_backbone.output

    DIR_1 = os.path.join(image_dir, 'output_other/')
    if not os.path.exists(DIR_1):
        os.mkdir(DIR_1)

    DIR_2 = os.path.join(image_dir, 'output_wood-making/')
    if not os.path.exists(DIR_2):
        os.mkdir(DIR_2)

    # head
    head = AveragePooling2D(pool_size=(7, 7))(backbone_output)
    head = Flatten(name="flatten")(head)
    head = Dense(512, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(512, activation="relu")(head)
    head = Dropout(0.25)(head)
    head = Dense(256, activation="relu")(head)
    head = Dropout(0.25)(head)
    head = Dense(64, activation="relu")(head)
    head = Dropout(0.25)(head)
    head = Dense(1, activation="sigmoid")(head)

    model = Model(inputs=resnet_backbone.input, outputs=head)
    model.load_weights('ep003-loss0.549-val_loss0.523.h5')

    IMAGE_PATH = gl(image_dir + '/*')

    for image_path in IMAGE_PATH:
        if not image_path.endswith(('.png', '.jpg', '.PNG', '.JPEG', 'jpeg')):
            continue
        image = cv2.imread(image_path)
        temp = image.copy()
        img_name = image_path.replace('\\', '').split('/')[-1]

        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        preds = int(model.predict(image)[0][0])

        if preds > thresh:
            output_name = os.path.join(DIR_2, img_name)
            cv2.imwrite(output_name, temp)
        else:
            output_name = os.path.join(DIR_1, img_name)
            cv2.imwrite(output_name, temp)


if __name__ == '__main__':
    args = parser.parse_args()
    image_dir = vars(args)['image_dir'][0]
    thresh = vars(args)['thresh']
    main(image_dir, thresh)
