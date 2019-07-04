# Image Segmentation and Knowledge Distillation

This repo provides a way to train a neural network for a binary image segmentation task and then shrink the model using knowledge distillation. After setting up the directory structures as detailed below, you can train the U-Net model defined in unet.py by running train.py. After training a model, which is saved in ./models, you can use this model for inference by running predict.py. Finally, because the default U-Net architecture is so large, there is a distill.py script that can be used to build a smaller U-Net given the large trained U-Net created by train.py.

These scripts expect a certain directory structure and rely on the files in ./utils.

The directory setup needs to be defined in ./utils/config.py in the variables TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VALID_IMG_DIR, VALID_LABEL_DIR. The default directories are in the labelled_data/ directory but the generic structure is 

```
        BASE_DIR/
            |
             ->TRAIN_IMG_DIR/
            |       |
            |        ->0/
            |
             ->TRAIN_LABEL_DIR/
            |       |
            |        ->0/
            |
             ->VALID_IMG_DIR/
            |       |
            |        ->0/
            |
             ->VALID_LABEL_DIR/
                    |
                     ->0/
 ```

so that the 0/ directories contain the actual image and label files. Once the labelled data is in place, we are ready to run train.py

# How to Train

To train a model on already labelled data we run the train.py script. The model and information on training variables will be saved in ./models by default and are created by the log() function inside train.py. Training history and a plot of loss and the Jaccard coefficient will be stored in ./logs and are created by a Keras callback and utils/make_plots.py, respectively. The locations of the ./log and ./models adirectories are defined in utils/config.py. The locations of the ./log and ./models adirectories are defined in utils/config.py..
```
usage: train.py [-h] [--init_channels INIT_CHANNELS]
                [--batch_size BATCH_SIZE]
                [--num_epochs NUM_EPOCHS]
optional arguments:
  -h, --help            show this help message and exit
  --init_channels INIT_CHANNELS
  --batch_size BATCH_SIZE
  --num_epochs NUM_EPOCHS
```
The default input size for the model is defined in ./utils/config.py

The images are sent to the U-Net model by ./utils/data_generator.py and this file also controls the data augmentation. For more or less augmentation, the train_gen() function can be modified by uncommenting the desired type of augmentation. Note that the augmentation arguments for the image and label are defined separately in case we want try adjusting brightness of the images.

# Generating Predictions

After training a model, we can generate predictions by calling predict.py and specifying a trained model. By default, predict.py will generate predictions for the training data, but this can be changed by either passing --valid True or --img_dir /path/to/image/dir/ to the script.

```
usage: predict.py [-h] --unet_weights UNET_WEIGHTS 
                  [--valid VALID]
                  [--img_dir IMG_DIR]
optional arguments:
  -h, --help            show this help message and exit
  --unet_weights UNET_WEIGHTS
  --valid VALID
  --img_dir IMG_DIR
```
Note that predict.py uses a utils/data_generator.py function called predict_gen() to feed the images to the model and so the images should be located in img_dir/0 and not img_dir itself.

The results will be saved in ./predictions/

# Knowledge Distillation

There are a number of ways to shrink a large model and this repo implements knowledge distillation in the distill.py script. The idea is to use a large model to train a small model. To do this, you need one pre-trained model that will generate labels on a given data set. Note that this means you do not need to have labelled data to do this. 

Once the large model has generated predictions, we can use these as the labels to train the small model. These labels are "soft targets" rather than 0/1 binary masks and improve the ability of the small model to learn.

This is all taken care of by the distill.py script. All that's needed is a trained model to act as the teacher and a target image directory, img_dir. This script will create a directory inside img_dir called teacher_label to hold the labels

```
usage: distill.py [-h] --teacher TEACHER
                  --base_img_dir IMG_DIR
                  [--init_channels INIT_CHANNELS]
                  [--num_epochs NUM_EPOCHS]
                  [--train_prop TRAIN_PROP]

optional arguments:
  -h, --help            show this help message and exit
  --teacher TEACHER
  --base_img_dir IMG_DIR
  --init_channels INIT_CHANNELS
  --num_epochs NUM_EPOCHS
  --train_prop TRAIN_PROP
```
Important: The --base_img_dir has a slightly different format in this case. Since we will be making labels we need to have the setup as

```
base_img_dir/
    |
    -> image/
         |
         ->0/
```

Then the script will save the labels in base_img_dir/label_teacher/0.

The directories ./tmp_distill/ contain only links to the files in base_img_dir and ./tmp_distill/ can be safely removed.

Finally, the training loss can take a large number of epochs before it begins to drop. If the U-Net 4 model has difficulty learning, try setting --init_channels 8, using a smaller teacher model (teacher with --init_channels = 8), adjusting augmentation strategies, or finally, incorporating a temperature parameter in the final output layer of the teacher model.
