"""
    This trains U-Net model on the labelled training data
    The location of the data is defined in ./utils/config.py
"""

import argparse
import os
from datetime import datetime
from keras import callbacks
from unet import unet
from utils.config import TRAIN_IMAGE_DIR, VALID_IMAGE_DIR
from utils.config import LOG_DIR, MODEL_DIR
from utils.data_generator import train_gen, valid_gen
from utils.make_plots import plot_history


def log(model_name, batch_size, num_epochs, img_gen_args, mask_gen_args):
    """ This logs information related to training, including the actual
        unet.py file in case architecture is changed.
        The results are saved in ./models/ """
    os.system("cp unet.py {}/{}".format(MODEL_DIR, model_name.split(".")[0]+".py"))
    args_file = open("{}/{}".format(MODEL_DIR, model_name.split(".")[0]+".args"), "x")
    args_file.write("{}\nbatch_size: {}\nepochs: {}\n".format(model_name, batch_size, num_epochs))
    args_file.write("Image Generator Arguments:\n")
    for (key, value) in img_gen_args.items():
        args_file.write("\t{}: {}\n".format(key, value))
    args_file.write("Label Generator Arguments:\n")
    for (key, value) in mask_gen_args.items():
        args_file.write("\t{}: {}\n".format(key, value))
    args_file.close()


def main(init_channels, batch_size, num_epochs):
    """ Entry point for training U-Net model.
        init_channels:  Determines the size of the U-Net model.
                        The default is 64
        batch_size:     Default is 2. Since the image size is large
                        it's easy to get out-of-memory errors.
        num_epochs:     Default is 100. It's not uncommon for training
                        to move slowly in the beginning and then take
                        off after even 50 epochs.

        Data is loaded from data_generator.py
        U-Net model is defined in unet.py
        csv log and graph are saved in ./logs/
        Trained model is saved in ./models/ with format unet-{init_channels}_{start_time}.h5 """

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = 'unet-{}_'.format(init_channels) + now + '.h5'

    # Setup our model which is defined in model.py
    model = unet(init_channels)
    model.summary()

    # Extra data augmetation can be done by changing the data_generator.py
    # file. The data augmentation arguments used are returned and logged
    train_generator, img_gen_args, mask_gen_args = train_gen(batch_size)
    validation_generator = valid_gen(batch_size)

    num_train_samples = len(os.listdir(os.path.join(TRAIN_IMAGE_DIR, '0')))
    num_valid_samples = len(os.listdir(os.path.join(VALID_IMAGE_DIR, '0')))

    steps_per_epoch = num_train_samples / batch_size
    validation_steps = num_valid_samples / batch_size

    log(model_name, batch_size, num_epochs, img_gen_args, mask_gen_args)

    # Basic callbacks
    checkpoint = callbacks.ModelCheckpoint(filepath=MODEL_DIR + model_name,
                                           monitor='val_loss',
                                           save_best_only=True)

    early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                         patience=50)

    csv_logger = callbacks.CSVLogger(LOG_DIR + model_name.split('.')[0] + '.csv')

    callback_list = [checkpoint, early_stop, csv_logger]

    # Training begins
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  verbose=1,
                                  callbacks=callback_list,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps)

    plot_history(history, model_name[:-2]+'png')

    model.save('./models/' + model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_channels", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    args = parser.parse_args()

    main(args.init_channels, args.batch_size, args.num_epochs)
