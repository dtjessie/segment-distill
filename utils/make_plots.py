"""
    Make some plots of the keras.history and save in logs/ so we can understand our loss better
"""

import argparse
import os
import matplotlib.pyplot as plt
from utils.config import BASE_DIR


def plot_csv(csv_file):
    """ Plot a keras csv file from logs/ """
    return 0


def plot_history(history, save_name):
    """ Plot the history from keras training, save in logs/ """
    plt.ylim((0.0, 1.0))
    plt.plot(history.history['loss'], 'r--')
    plt.plot(history.history['val_loss'], 'r-')
    plt.plot(history.history['jaccard_coef'], 'b--')
    plt.plot(history.history['val_jaccard_coef'], 'b-')
    plt.title("Loss and Jaccard Coefficient")
    plt.legend(["Train Loss", "Val Loss", "Train Jaccard", "Val Jaccard"], loc="center right")
    plt.savefig(os.path.join(BASE_DIR, "logs/", save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()

    plot_csv(args.csv_file)
