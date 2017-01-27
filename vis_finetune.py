
"""
Visualise Caffe training progress from log file
from https://github.com/yassersouri/omgh/blob/master/src/scripts/vis_finetune.py
and
https://groups.google.com/forum/#!searchin/caffe-users/vis_finetune.py%7Csort:relevance/caffe-users/FJ4_mYTNK70/hAl2GhsnbbMJ

"""
import numpy as np
import re
import click
from matplotlib import pylab as plt
import matplotlib.pyplot


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy %')
    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, t_loss_iterations, t_losses = parse_log(
            log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations,
                     accuracies, accuracies_iteration_checkpoints_ind, t_loss_iterations, t_losses, color_ind=i)
    plt.show()


def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    training_loss_pattern = r"Iteration (?P<iter_num>\d+) training set loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    t_losses = []
    t_loss_iterations = []

    for r in re.findall(training_loss_pattern, log):
        t_loss_iterations.append(int(r[0]))
        t_losses.append(float(r[1]))

    t_loss_iterations = np.array(t_loss_iterations)
    t_losses = np.array(t_losses)
    print 'Number of train loss iter {}, Number of train losses {}'.format(
        np.shape(t_loss_iterations), np.shape(t_losses))

    # accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*
    # accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracy_pattern = r"Iteration (?P<iter_num>\d+) trip accuracy (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracy_iterations = []
    accuracies_iteration_checkpoints_ind = []

    for r in re.findall(accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(
                len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)

    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, t_loss_iterations, t_losses


def disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations,
                 accuracies, accuracies_iteration_checkpoints_ind, t_loss_iterations,
                 t_losses, color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(loss_iterations, losses, color=plt.rcParams[
             'axes.color_cycle'][(color_ind * 2 + 0) % modula])
    ax1.plot(t_loss_iterations, t_losses, color=plt.rcParams[
             'axes.color_cycle'][(color_ind * 2 + 2) % modula])

    ax2.plot(accuracy_iterations, accuracies, plt.rcParams[
             'axes.color_cycle'][(color_ind * 2 + 1) % modula])
    ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[
             accuracies_iteration_checkpoints_ind], 'o', color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])


if __name__ == '__main__':
    main()
