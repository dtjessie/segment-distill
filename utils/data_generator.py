"""
    Data generator for serving images to our model.
    Used by train.py, predict.py, and distill.py
    For more or less data augmentation, the arguments
    in each generator can be adjusted.
"""

from keras.preprocessing.image import ImageDataGenerator
from .config import TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR
from .config import VALID_IMAGE_DIR, VALID_LABEL_DIR
from .config import RESIZE_SHAPE, SEED


def train_gen(batch_size):
    seed = SEED
    img_gen_args = dict(rescale=1.0/255,
                        rotation_range=10,
                        #width_shift_range=.1,
                        #height_shift_range=.1,
                        #shear_range=.05,
                        #zoom_range=.3,
                        #brightness_range=[.8, 1.3],
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect'
                        )
    mask_gen_args = dict(rescale=1.0/255,
                         rotation_range=10,
                         #width_shift_range=.1,
                         #height_shift_range=.1,
                         #shear_range=.05,
                         #zoom_range=.3,
                         #brightness_range=[.8,1.2],
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect'
                         )
    image_datagen = ImageDataGenerator(**img_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    image_generator = image_datagen.flow_from_directory(TRAIN_IMAGE_DIR,
                                                        target_size=RESIZE_SHAPE,
                                                        batch_size=batch_size,
                                                        class_mode=None,
                                                        color_mode='grayscale',
                                                        shuffle=True,
                                                        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(TRAIN_LABEL_DIR,
                                                      target_size=RESIZE_SHAPE,
                                                      batch_size=batch_size,
                                                      class_mode=None,
                                                      color_mode='grayscale',
                                                      shuffle=True,
                                                      seed=seed)

    train_generator = zip(image_generator, mask_generator)
    return train_generator, img_gen_args, mask_gen_args


def valid_gen(batch_size):
    seed = SEED
    img_gen_args = dict(rescale=1.0/255,
                        #rotation_range=10,
                        #width_shift_range=.1,
                        #height_shift_range=.1,
                        #shear_range=.05,
                        #zoom_range=.3,
                        #brightness_range=[.8, 1.3],
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect'
                        )
    image_datagen = ImageDataGenerator(**img_gen_args)
    mask_datagen = ImageDataGenerator(**img_gen_args)

    image_generator = image_datagen.flow_from_directory(VALID_IMAGE_DIR,
                                                        target_size=RESIZE_SHAPE,
                                                        batch_size=batch_size,
                                                        class_mode=None,
                                                        color_mode='grayscale',
                                                        shuffle=False,
                                                        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(VALID_LABEL_DIR,
                                                      target_size=RESIZE_SHAPE,
                                                      batch_size=batch_size,
                                                      class_mode=None,
                                                      color_mode='grayscale',
                                                      shuffle=False,
                                                      seed=seed)

    valid_generator = zip(image_generator, mask_generator)
    return valid_generator


def predict_gen(img_dir=TRAIN_IMAGE_DIR, label_dir=TRAIN_LABEL_DIR, batch_size=1):
    seed = SEED
    image_datagen = ImageDataGenerator(rescale=1.0/255)
    mask_datagen = ImageDataGenerator(rescale=1.0/255)

    image_generator = image_datagen.flow_from_directory(img_dir,
                                                        target_size=RESIZE_SHAPE,
                                                        batch_size=batch_size,
                                                        class_mode=None,
                                                        shuffle=False,
                                                        color_mode='grayscale',
                                                        seed=seed)
    if label_dir is not None:
        mask_generator = mask_datagen.flow_from_directory(label_dir,
                                                          target_size=RESIZE_SHAPE,
                                                          batch_size=batch_size,
                                                          class_mode=None,
                                                          color_mode='grayscale',
                                                          shuffle=False,
                                                          seed=seed)

        return zip(image_generator, mask_generator)

    return image_generator


def distill_gen(img_dir=TRAIN_IMAGE_DIR, label_dir=TRAIN_LABEL_DIR, batch_size=1):
    seed = SEED
    img_gen_args = dict(rescale=1.0/255,
                        #rotation_range=10,
                        #width_shift_range=.1,
                        #height_shift_range=.1,
                        #shear_range=.05,
                        #zoom_range=.3,
                        #brightness_range=[.8, 1.3],
                        horizontal_flip=True,
                        #vertical_flip=True,
                        fill_mode='reflect'
                        )
    mask_gen_args = dict(rescale=1.0/255,
                         #rotation_range=10,
                         #width_shift_range=.1,
                         #height_shift_range=.1,
                         #shear_range=.05,
                         #zoom_range=.3,
                         #brightness_range=[.8,1.2],
                         horizontal_flip=True,
                         #vertical_flip=True,
                         fill_mode='reflect'
                         )
    image_datagen = ImageDataGenerator(**img_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    image_generator = image_datagen.flow_from_directory(img_dir,
                                                        target_size=RESIZE_SHAPE,
                                                        batch_size=batch_size,
                                                        class_mode=None,
                                                        shuffle=False,
                                                        color_mode='grayscale',
                                                        seed=seed)
    if label_dir is not None:
        mask_generator = mask_datagen.flow_from_directory(label_dir,
                                                          target_size=RESIZE_SHAPE,
                                                          batch_size=batch_size,
                                                          class_mode=None,
                                                          color_mode='grayscale',
                                                          shuffle=False,
                                                          seed=seed)

        return zip(image_generator, mask_generator)

    return image_generator
