from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from glob import glob as gl
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--vid_dir', metavar='N', type=str, nargs='+',
                    help='video path, must not ends with "/"')
parser.add_argument('--class_thresh', metavar='N', type=float, nargs='+', default=0.5,
                    help='threshold for classification default = 0.5')
parser.add_argument('--content_thresh', metavar='N', type=float, nargs='+', default=0.4,
                    help='threshold for content default = 0.4')


def get_frame(cap, sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    success, image = cap.read()

    return success, image


def main(vid_dir, class_thresh, content_thresh):
    resnet_backbone = ResNet50(weights="imagenet", include_top=False, pooling=None,
                               input_tensor=Input(shape=(224, 224, 3)))
    backbone_output = resnet_backbone.output

    DIR_1 = os.path.join(vid_dir, 'output_other/')
    if not os.path.exists(DIR_1):
        os.mkdir(DIR_1)

    DIR_2 = os.path.join(vid_dir, 'output_wood-making/')
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

    RAW_VIDEO_DATA_PATH = gl(vid_dir + '/*')

    for vid_path in RAW_VIDEO_DATA_PATH:
        if not vid_path.endswith(('.mp4', '.MP4', '.avi', '.mov', '.mkv')) or os.path.isdir(vid_path):
            continue
        cap = cv2.VideoCapture(vid_path)
        vid_name = vid_path.replace('\\', '').split('/')[-1]
        Q = []
        if not cap.isOpened():
            print("Error opening video stream or file")

        frame_rate = 2  # predict every 2 second
        sec = 0
        success, image = get_frame(cap, sec)

        while success:
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            preds = int(model.predict(image)[0][0])
            Q.append(preds)

            sec = sec + frame_rate
            sec = round(sec, 2)
            success, image = get_frame(cap, sec)

        Q = np.array(Q)
        Q[Q > class_thresh] = 1
        Q[Q < class_thresh] = 0
        counts = np.bincount(np.array(Q), minlength=2)
        output_label = np.true_divide(counts, len(Q))
        cap.release()

        if output_label[1] < content_thresh:
            output_name = os.path.join(DIR_1, vid_name)
            print('Video {} has {}% content is other and {}% content is wood-making. Move to {}'.format(vid_name,
                                                                                                        output_label[0],
                                                                                                        output_label[1],
                                                                                                        DIR_1))
            Path(vid_path).rename(output_name)

        else:
            output_name = os.path.join(DIR_2, vid_name)
            print('Video {} has {}% content is other and {}% content is wood-making. Move to {}'.format(vid_name,
                                                                                                        output_label[0],
                                                                                                        output_label[1],
                                                                                                        DIR_2))
            Path(vid_path).rename(output_name)


if __name__ == '__main__':
    args = parser.parse_args()
    vid_dir = vars(args)['vid_dir'][0]
    class_thresh = vars(args)['class_thresh']
    content_thresh = vars(args)['content_thresh']

    main(vid_dir, class_thresh, content_thresh)
