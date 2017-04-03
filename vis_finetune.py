
"""
Visualise Caffe training progress from log file
from https://github.com/yassersouri/omgh/blob/master/src/scripts/vis_finetune.py
and
https://groups.google.com/forum/#!searchin/caffe-users/vis_finetune.py%7Csort:relevance/caffe-users/FJ4_mYTNK70/hAl2GhsnbbMJ

"""
import numpy as np
import re
import click
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
# import matplotlib.pyplot
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
        loss_iterations, losses, val_acc_iterations, val_accuracies, \
            val_acc_iteration_checkpoints_ind, val_loss_iterations, val_losses,  \
            train_acc_iterations, train_acc, train_acc_iteration_checkpoints_ind = parse_log(
                log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses, val_acc_iterations,
                     val_accuracies, val_acc_iteration_checkpoints_ind,
                     val_loss_iterations, val_losses,
                     train_acc_iterations, train_acc,
                     train_acc_iteration_checkpoints_ind, color_ind=i)
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
    fig.savefig(os.path.join(save_dir, log_name + '.pdf'))
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

    val_loss_pattern = r"Iteration (?P<iter_num>\d+) val set loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    val_losses = []
    val_loss_iterations = []

    for r in re.findall(val_loss_pattern, log):
        val_loss_iterations.append(int(r[0]))
        val_losses.append(float(r[1]))

    val_loss_iterations = np.array(val_loss_iterations)
    val_losses = np.array(val_losses)
    print 'Number of train loss iter {}, Number of train losses {}'.format(
        np.shape(val_loss_iterations), np.shape(val_losses))

    # accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*
    # accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    val_acc_pattern = r"Iteration (?P<iter_num>\d+) val trip accuracy (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    val_acc = []
    val_acc_iterations = []
    val_acc_iteration_checkpoints_ind = []

    for r in re.findall(val_acc_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            val_acc_iteration_checkpoints_ind.append(
                len(val_acc_iterations))

        val_acc_iterations.append(iteration)
        val_acc.append(accuracy)

    val_acc_iterations = np.array(val_acc_iterations)
    val_acc = np.array(val_acc)

    train_acc_pattern = r"Iteration (?P<iter_num>\d+) train trip accuracy (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    train_acc = []
    train_acc_iterations = []
    train_acc_iteration_checkpoints_ind = []

    for r in re.findall(train_acc_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            train_acc_iteration_checkpoints_ind.append(
                len(train_acc_iterations))

        train_acc_iterations.append(iteration)
        train_acc.append(accuracy)

    train_acc_iterations = np.array(train_acc_iterations)
    train_acc = np.array(train_acc)

    return loss_iterations, losses, val_acc_iterations, val_acc, val_acc_iteration_checkpoints_ind,  \
        val_loss_iterations, val_losses,  \
        train_acc_iterations, train_acc, train_acc_iteration_checkpoints_ind


def disp_results(fig, ax1, ax2, loss_iterations, losses, val_acc_iterations,
                 val_accuracies, val_acc_iteration_checkpoints_ind, val_loss_iterations,
                 t_losses,
                 train_acc_iterations, train_acc, train_acc_iteration_checkpoints_ind,
                 color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    train_l_h, = ax1.plot(loss_iterations, losses, color=plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 0) % modula], linestyle='-',
        label='val loss')
    val_l_h, = ax1.plot(val_loss_iterations, t_losses, color=plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 2) % modula], linestyle='--',
        label='training loss')

    val_a_h, = ax2.plot(val_acc_iterations, val_accuracies, plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 1) % modula], linestyle='-.',
        linewidth=2, label='val accuracy')
    ax2.plot(val_acc_iterations[val_acc_iteration_checkpoints_ind], val_accuracies[
             val_acc_iteration_checkpoints_ind], 'o',
             color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])

    train_a_h, = ax2.plot(train_acc_iterations, train_acc, plt.rcParams[
        'axes.color_cycle'][(color_ind * 2 + 1) % modula],linestyle=':',
        label='val accuracy')
    ax2.plot(train_acc_iterations[train_acc_iteration_checkpoints_ind], train_acc[
             train_acc_iteration_checkpoints_ind], 'o',
             color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 3) % modula])
    if color_ind == 0:
        fig.legend((train_l_h, val_l_h,
                    val_a_h, train_a_h), ('Train loss',
                                          'Val Loss',
                                          'Val Trip Acc',
                                          'Train Trip Acc'),
                   loc='upper right')
    # else:
    #     fig.legend((train_l_h), ('Train Loss'), loc='upper right')

if __name__ == '__main__':
    main()
