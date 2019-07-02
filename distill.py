"""
    Shrink U-Net model using knowledge distillation techniques.
    Teacher is a large model, student is a small model.
    Teacher has 64 initial channels, student has 4.
    The outputs of the teacher model are the targets for the student model.
    These soft target increase the ability of the student to learn.
    If the student has difficulty learning, one adjustment to make is to change
    the teacher's final output layer to include a temperature parameter. This
    will soften the targets further and can improve the student's ability to learn.

    Note: This file will create a './tmp_distill/' directory that has links
          to the img_dir specified. It's safe to delete this directory and
          should be removed whenever new teacher labels are created
"""

import argparse
from collections import deque
from datetime import datetime
from keras import callbacks
from keras.models import load_model
from keras.backend import clear_session
import matplotlib.pyplot as plt
import numpy as np
import os
from unet import unet
from utils.config import DEQUE_MAXLEN, BASE_DIR
from utils.data_generator import predict_gen, distill_gen
from utils.jaccard_loss import jaccard_coef
from utils.make_plots import plot_history


def log(model_name, init_channels, batch_size, num_epochs, img_gen_args, mask_gen_args):
    args_file = open("./models/{}".format(model_name.split(".")[0]+".args"), "x")
    args_file.write("{}\ninit_chanels: {}\nbatch_size: {}\nepochs: {}\n".format(model_name, init_channels, batch_size, num_epochs))
    for (key, value) in img_gen_args.items():
        args_file.write("{}: {}\n".format(key, value))
    for (key, value) in mask_gen_args.items():
        args_file.write("{}: {}\n".format(key, value))
    args_file.close()


def make_labels(img_dir, teacher_model_file):
    """ Use the teacher model to create labels from img_dir.
        These labels will be used to train the student model below.
        If the directory exists, it will skip making labels.  """
    teacher = load_model(teacher_model_file, custom_objects={"jaccard_coef": jaccard_coef})

    train_img_dir = os.path.join(img_dir, 'image')
    save_path = os.path.join(img_dir, 'label_teacher/0')
    if os.path.isdir(save_path):
        print("\n\nDirectory {} already exists".format(save_path))
        print("\nNot making labels ..\n\n")
        return 0

    os.mkdir(os.path.join(img_dir, 'label_teacher'))
    os.mkdir(os.path.join(img_dir, 'label_teacher/0'))

    predict_generator = predict_gen(img_dir=train_img_dir, label_dir=None)
    num_predictions = len(os.listdir(os.path.join(train_img_dir, '0/')))

    # If there are thousands of frames, we don't want ot put them all in memory
    # so use a double-ended queue, deque
    preds = deque(maxlen=DEQUE_MAXLEN)
    batches = num_predictions // DEQUE_MAXLEN
    remainder = num_predictions % DEQUE_MAXLEN

    print("Generating labels in {}\nUsing model {}\n".format(save_path, teacher_model_file))
    i = 0
    for batch in range(batches):
        for idx in range(DEQUE_MAXLEN):
            x = predict_generator.__next__()
            y_pred = teacher.predict(x)
            preds.append(y_pred)
        for idx, y_pred in enumerate(preds):
            tmp = np.reshape(y_pred[0], (x.shape[1], x.shape[2]))
            plt.imsave(os.path.join(save_path, 'frame{:06d}.png'.format(i)), tmp, cmap='gray')
            i += 1
            plt.close()
    preds.clear()
    for idx in range(remainder):
        x = predict_generator.__next__()
        y_pred = teacher.predict(x)
        preds.append(y_pred)
    for idx, y_pred in enumerate(preds):
        tmp = np.reshape(y_pred[0], (x.shape[1], x.shape[2]))
        plt.imsave(os.path.join(save_path, 'frame{:06d}.png'.format(i)), tmp, cmap='gray')
        i += 1
        plt.close()
    clear_session()


def setup_dirs(img_dir):
    """ Need to place training and validation data into separate directories so
        create a ./tmp_distill/ directory and make links to img_dir """
    try:
        os.mkdir('tmp_distill')
        
        os.mkdir('tmp_distill/train')
        os.mkdir('tmp_distill/train/image/')
        os.mkdir('tmp_distill/train/image/0')
        os.mkdir('tmp_distill/train/label')
        os.mkdir('tmp_distill/train/label/0')

        os.mkdir('tmp_distill/valid')
        os.mkdir('tmp_distill/valid/image/')
        os.mkdir('tmp_distill/valid/image/0')
        os.mkdir('tmp_distill/valid/label')
        os.mkdir('tmp_distill/valid/label/0')
    except OSError as e:
        print("OSError: {}".format(e))
        print("Not making links to {}".format(img_dir))
        return 0
    img_list = os.listdir(os.path.join(img_dir + 'image/0'))
    img_list.sort()
    total_train = int(.9*len(img_list))

    for train_img in img_list[:total_train]:
        os.system("ln -s {}/{} ./tmp_distill/train/image/0/".format(os.path.join(BASE_DIR + img_dir + 'image/0'), train_img))
        os.system("ln -s {}/{} ./tmp_distill/train/label/0/".format(os.path.join(BASE_DIR + img_dir + 'label_teacher/0'), train_img))
    for valid_img in img_list[total_train:]:
        os.system("ln -s {}/{} ./tmp_distill/valid/image/0/".format(os.path.join(BASE_DIR + img_dir + 'image/0'), valid_img))
        os.system("ln -s {}/{} ./tmp_distill/valid/label/0/".format(os.path.join(BASE_DIR + img_dir + 'label_teacher/0'), valid_img))


def main(init_channels, num_epochs, img_dir, teacher):
    """
    Inputs:
        init_channels   gives size of the student model
        num_epcohs      how many epochs to train student
        img_dir         base directory for images used in training
        teacher         model file to generate labels for img_dir

    The output is a student model file located in ./models/
    """

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = 'unet_student_{}_'.format(init_channels) + now + '.h5'
    batch_size = 8

    # Set up labels and build our model which is defined in model.py
    make_labels(img_dir, teacher)
    setup_dirs(img_dir)
    student = unet(init_channels)
    student.summary()

    # Note tht he generators below may include data augmentation!
    # Experiment with this to improve the learning of the student
    train_generator = distill_gen(img_dir='./tmp_distill/train/image',
                                  label_dir='./tmp_distill/train/label',
                                  batch_size=batch_size)
    validation_generator = distill_gen(img_dir='./tmp_distill/valid/image',
                                       label_dir='./tmp_distill/valid/label',
                                       batch_size=batch_size)

    num_train_samples = len(os.listdir('./tmp_distill/train/image/0'))
    num_valid_samples = len(os.listdir('./tmp_distill/valid/label/0'))

    steps_per_epoch = num_train_samples / batch_size
    validation_steps = int(num_valid_samples / batch_size)

    # Basic callbacks
    checkpoint = callbacks.ModelCheckpoint(filepath='./models/' + model_name,
                                           monitor='val_loss',
                                           save_best_only=True)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=100)

    csv_logger = callbacks.CSVLogger('./logs/' + model_name.split('.')[0] + '.csv')

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=4)

    callback_list = [checkpoint, csv_logger, early_stop, reduce_lr]

    # Training begins
    history = student.fit_generator(train_generator,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=num_epochs,
                                    verbose=1,
                                    callbacks=callback_list,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps)

    plot_history(history, model_name[:-2]+'png')

    student.save('./models/' + model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_channels", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--img_dir", type=str, default='./warehouse_data/images/Jonathan/Schneider/sch_2018-04-26-21-59-02/')
    parser.add_argument("--teacher", type=str, default='models/unet_2019-06-21_14:26:22.h5')
    args = parser.parse_args()

    main(args.init_channels, args.num_epochs, args.img_dir, args.teacher)
