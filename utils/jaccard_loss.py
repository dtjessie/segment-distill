""" Jaccard is another name for Intersection over Union, IoU.
    The standard formula is
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The above forumula is from
    https://en.wikipedia.org/wiki/Jaccard_index

    Since the Jaccard is always between 0 (worst case)
    and 1 (best case), we can define a Jaccard loss function
    as the distance from 1.

    For a smoothed version and other ideas, please see
    https://gist.github.com/wassname """


import keras.backend as K


def jaccard_coef(y_true, y_pred):
    """ Intersection over union
        Gives a value between 0 and 1 """
    intersection = K.sum(y_true * y_pred, axis=-1)
    summation = K.sum(y_true + y_pred, axis=-1)
    jaccard = intersection / (summation - intersection)
    return jaccard


def jaccard_coef_hard(y_true, y_pred):
    """ Round the values of y_pred to be 0 or 1 to
        compute and then compute IoU
        y_true is also rounded so it will change the
        labels if using soft targets in knowledge distilation """
    y_true_round = K.round(y_true)
    y_pred_round = K.round(y_pred)
    return jaccard_coef(y_true_round, y_pred_round)


def jaccard_distance_loss(y_true, y_pred):
    """ Jaccard coefficient is between 0 and 1, so for loss use
        distance from 1 """
    return 1 - jaccard_coef(y_true, y_pred)
