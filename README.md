# Image Segmentation and Knowledge Distillation

This repo provides a way to train a neural network for a binary image segmentation task and then shrink the model using knowledge distillation.

It's set up now to train the U-Net model defined in unet.py. 

This repo consists of three main files to run: train.py, predict.py, distill.py.

These scripts expect a certain directory structure and rely on the files in ./utils.

The directory setup needs to be defined in ./utils/config.py in the variables TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VALID_IMG_DIR, VALID_LABEL_DIR.

The files expect a directory setup as
```
        BASE_DIR
            |
             ->TRAIN_IMG_DIR
            |       |
            |        ->0
            |
             ->TRAIN_LABEL_DIR
            |       |
            |        ->0
            |
             ->VALID_IMG_DIR
            |       |
            |        ->0
            |
             ->VALID_LABEL_DIR
                    |
                     ->0
```
so that the 0/ directories contain the actual image and label files.

# How to Train

To train a model on already labelled data you need the train.py script.

The model will be saved in ./models by default.

Training history and a plot of loss and the Jaccard coefficient will be stored in ./logs.

Currently, these directories are hard-coded in train.py but should be moved to utils/config.py

The default input size for the model is defined in ./utils/config.py

# Generating Predictions

Once you have a trained model, you can generate predictions by calling predict.py and specifying
any trained model.

The results will be saved in ./predictions/

# Knowledge Distillation

There are a number of ways to shrink a large model and this repo uses the distll.py script to do knowledge distillation.

The idea is to use a large model to train a small model.

To do this, you need one pre-trained model that will generate labels on a given data set. Note that this means you do not need to have labelled data to do this.

Once the large model has generated predictions, we can use these as the labels to train the small model.

These labels are "soft targets" rather than 0/1 binary masks and improve the ability of the small model to learn.

Note that training can be unstable here and it may not see improvement for 200 epochs of training before the loss begins to drop.

This is all taken care of by the distill.py script. All that's needed is a trained model to act as the teacher and a target image directory.

Note that the image directory should contain the actual images and not contain the 0/ directory as in the training above. The reason for this is that the images in this case may not be labelled.
