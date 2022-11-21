"""
    python ./analysis/grapher_util.py

    vvv IMPORTANT vvv
    RUN it in the ROOT directory, the relative directory creation
    is not implemented cleanly!

    Given the log data tensors from logger, creates the (training 
    and testing) loss and accuracy graphs.

    Takes in a configuration YAML file. The YAML needs to include:
    - graph_dir_path: the path to the directory that we want the graphs to be stored in
        (consider informative naming, such as "relnet-5way" or "matchnet-20way")
    - logdatas: the list of logdata dictionaries, where every dictionary should include the keys
        - label: label to be shown on the plot
        - path: path to the logdata tensor

"""

import torch
import numpy as np
from sys import argv, exit
import os
from matplotlib import pyplot as plt
import yaml


def read_logdata(path):
    """
        Takes the relative path of a logdata tensor.
        Returns (all epoch-indexed)
        - train_loss_arr
        - train_acc_arr
        - test_loss_arr
        - test_acc_arr
    """
    data = torch.load(path)
    batch_per_epoch = data.shape[1]

    # finds the total numbers of epoch to be graphed
    # (assumes that the final epoch to be included in the graph was greater than 0)
    total_e = 0
    is_nonzero = True

    while is_nonzero and total_e != data.shape[0]-1:
        # any batch in that epoch should have non-zero training loss
        # since we save only once every epoch
        total_e += 1
        is_nonzero = torch.is_nonzero(data[total_e, 0, 0])

    train_loss_arr = np.ndarray(total_e)
    train_acc_arr = np.ndarray(total_e)
    test_loss_arr = np.ndarray(total_e)
    test_acc_arr = np.ndarray(total_e)

    for e in range(total_e):

        train_loss_arr[e] = data[e, :, 0].mean()
        train_acc_arr[e] = data[e, :, 1].mean()
        test_loss_arr[e] = data[e, :, 2].mean()
        test_acc_arr[e] = data[e, :, 3].mean()

    return train_loss_arr, train_acc_arr, test_loss_arr, test_acc_arr


def find_num_epochs(some_metric_list):
    """
        Given a list of train/test loss/acc array,
        returns the smallest total number of epochs of these different model trainings.
    """
    smallest_e = np.inf

    for metric_arr in some_metric_list:
        num_e = len(metric_arr)
        if num_e < smallest_e:
            smallest_e = num_e

    return smallest_e


###########################
#           Main          #
###########################
if __name__ == "__main__":

    error_msg = "Usage: python ./analysis/grapher_util.py"
    if len(argv) != 1:
        exit(error_msg)

    path_to_config = "./analysis/graph_config.yaml"  # default value for easier usage

    with open(path_to_config, 'r') as file:
        config_dict = yaml.safe_load(file)

    if not os.path.exists(config_dict["graph_dir_path"]):
        os.makedirs(config_dict["graph_dir_path"])

    list_of_train_losses = []
    list_of_train_accs = []
    list_of_test_losses = []
    list_of_test_accs = []

    for logdata in config_dict["logdatas"]:

        train_loss_arr, train_acc_arr, test_loss_arr, test_acc_arr = read_logdata(
            logdata["path"])
        list_of_train_losses.append(train_loss_arr)
        list_of_train_accs.append(train_acc_arr)
        list_of_test_losses.append(test_loss_arr)
        list_of_test_accs.append(test_acc_arr)

    total_e = find_num_epochs(list_of_train_losses)
    epoch_arr = np.arange(start=0, stop=total_e)

    ### train loss plot ###
    series = 0  # index of series plotting
    for logdata in config_dict["logdatas"]:
        train_loss_arr = list_of_train_losses[series]
        plt.plot(epoch_arr, train_loss_arr[:total_e], label=logdata["label"])
        series += 1
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config_dict["graph_dir_path"] + "/train_loss.png")
    plt.close()

    ### train acc plot ###
    series = 0  # index of series plotting
    for logdata in config_dict["logdatas"]:
        train_acc_arr = list_of_train_accs[series]
        plt.plot(epoch_arr, train_acc_arr[:total_e], label=logdata["label"])
        series += 1
    plt.title("Training Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(config_dict["graph_dir_path"] + "/train_acc.png")
    plt.close()

    ### test loss plot ###
    series = 0  # index of series plotting
    for logdata in config_dict["logdatas"]:
        test_loss_arr = list_of_test_losses[series]
        plt.plot(epoch_arr, test_loss_arr[:total_e], label=logdata["label"])
        series += 1
    plt.title("Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config_dict["graph_dir_path"] + "/test_loss.png")
    plt.close()

    ### test acc plot ###
    series = 0  # index of series plotting
    for logdata in config_dict["logdatas"]:
        test_acc_arr = list_of_test_accs[series]
        plt.plot(epoch_arr, test_acc_arr[:total_e], label=logdata["label"])
        series += 1
    plt.title("Testing Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(config_dict["graph_dir_path"] + "/test_acc.png")
    plt.close()

# plt.plot(epoch_arr, train_loss_arr, color="red", label="Training Loss")
# plt.plot(epoch_arr, test_loss_arr, color="blue", label="Testing Loss")
# plt.title("Loss over Training Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig("analysis/graphs/loss.png")
# plt.close()

# plt.plot(epoch_arr, train_acc_arr, color="red", label="Training Accuracy")
# plt.plot(epoch_arr, test_acc_arr, color="blue", label="Testing Accuracy")
# plt.title("Classification Accuracy over Training Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig("analysis/graphs/acc.png")
# plt.close()
