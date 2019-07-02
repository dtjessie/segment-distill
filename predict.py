""" This script views predictions of a trained model on a specified directory of images.
    The model is passed in --unet_weights
    If no directory is given, the training data is used
    If --valid True is passed, then the validation data is used
    Else --img_dir can be specified and will use the images located there """


import argparse
from collections import deque
import datetime
import numpy as np
import os
from PIL import Image
from keras.models import load_model
from utils.config import TRAIN_IMAGE_DIR, VALID_IMAGE_DIR
from utils.config import RESIZE_SHAPE, DEQUE_MAXLEN
from utils.data_generator import predict_gen
from utils.jaccard_loss import jaccard_coef, jaccard_coef_hard


def main(unet_weights, valid, img_dir):
    batch_size = 1
    if valid is True:
        img_dir = VALID_IMAGE_DIR
        tag = 'valid'
    elif img_dir is None:
        img_dir = TRAIN_IMAGE_DIR
        tag = 'train'
    else:
        tag = img_dir.strip('.').replace('/', '-')
    print("\n\nUsing data from {}\n\n".format(img_dir))

    num_predictions = len(os.listdir(os.path.join(img_dir, '0/')))
    print("Found {} images".format(num_predictions))

    # unet_weights can be specified as a relative path, so find the part of
    # the name that has h5 in the name
    unet_name = [name for name in unet_weights.split('/') if '.h5' in name][0]
    unet_version = unet_name.split('.')[0]
    save_dir = "{}-{}".format(unet_version, tag)

    model = load_model(unet_weights, custom_objects={'jaccard_coef': jaccard_coef, 'jaccard_coef_hard': jaccard_coef_hard})
    predict_generator = predict_gen(img_dir=img_dir, label_dir=None, batch_size=batch_size)

    try:
        os.mkdir('./predictions/' + save_dir)
    except OSError as e:
        print("Error making directory:\t {}".format(e))
        print("Exiting...")
        return 0
    save_path = os.path.join('./predictions/', save_dir)

    # If there are thousands of frames, don't store them all in memory
    # Use a double ended queue instead
    preds = deque(maxlen=DEQUE_MAXLEN)
    batches = num_predictions // DEQUE_MAXLEN
    remainder = num_predictions % DEQUE_MAXLEN

    i = 0
    for batch in range(batches):
        start_time = datetime.datetime.now()
        for idx in range(DEQUE_MAXLEN):
            x = predict_generator.__next__()
            y_pred = model.predict(x)
            preds.append(y_pred)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print("Predictions on {} frames took {} seconds".format(DEQUE_MAXLEN, elapsed_time.total_seconds()))
        print("Saving predictions ...\n")
        for y_pred in preds:
            im = Image.fromarray(np.reshape(255*y_pred[0], RESIZE_SHAPE)).convert('L')
            im.save(os.path.join(save_path, 'frame{:06d}.png'.format(i)))
            i += 1
            im.close()
    preds.clear()
    for idx in range(remainder):
        x = predict_generator.__next__()
        y_pred = model.predict(x)
        preds.append(y_pred)
    for y_pred in preds:
        im = Image.fromarray(np.reshape(255*y_pred[0], RESIZE_SHAPE)).convert('L')
        im.save(os.path.join(save_path, 'frame{:06d}.png'.format(i)))
        i += 1
        im.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_weights", type=str, required=True)
    parser.add_argument("--valid", type=bool, default=False)
    parser.add_argument("--img_dir", type=str)
    args = parser.parse_args()

    main(args.unet_weights, args.valid, args.img_dir)
