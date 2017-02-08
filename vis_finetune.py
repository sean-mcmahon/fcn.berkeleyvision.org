
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
import os


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('trip accuracy %')
    print type(files), 'shape ', np.shape(files)
    print files
    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, accuracies, \
            accuracies_iteration_checkpoints_ind, t_loss_iterations, \
            t_losses = parse_log(log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations,
                     accuracies, accuracies_iteration_checkpoints_ind,
                     t_loss_iterations, t_losses, color_ind=i)
    if len(files) == 1:
        log_name = os.path.splitext(os.path.basename(files[0]))[0]
        save_dir = os.path.dirname(files[0])
        fig.suptitle('Logfile ' + log_name, fontsize=14, fontweight='bold')
    elif len(files) > 1:
        log_name = 'Mulitple Log files'
        # save to scripts directory
        save_dir = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__)))
        fig.suptitle(log_name, fontsize=7, fontweight='bold')
    print files[0]
    fig.savefig(save_dir + log_name + '.pdf')
    print 'PDF file ({}.pdf) saved to {}'.format(log_name, save_dir)
    try:
        plt.show()
    except:
        print '----\nvis_finetune.py (main) plotting has failed\n----'


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

    training_loss_pattern = r"Iteration (?P<iter_num>\d+) val set loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
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
    accuracy_pattern = r"Iteration (?P<iter_num>\d+) val trip accuracy (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
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
    val_l_h, = ax1.plot(loss_iterations, losses, color=plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 0) % modula], linestyle='-',
        label='val loss')
    train_l_h, = ax1.plot(t_loss_iterations, t_losses, color=plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 2) % modula], linestyle='--',
        label='training loss')

    val_a_h, = ax2.plot(accuracy_iterations, accuracies, plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 1) % modula],
        label='val accuracy')
    ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[
             accuracies_iteration_checkpoints_ind], 'o',
             color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])
    if color_ind == 0:
        fig.legend((val_l_h, train_l_h, val_a_h), ('Train loss',
                                                   'Val Loss',
                                                   'Val Trip Acc'),
                   loc='upper right')
    # else:
    #     fig.legend((train_l_h), ('Train Loss'), loc='upper right')

if __name__ == '__main__':
    main()
