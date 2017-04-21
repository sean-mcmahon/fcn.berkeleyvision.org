"""
Found an error with the log files. The c++ printouts start inbetween the python
printouts. So this script goes and fixes them!
The chars "I0" should always start on a new line. If not it's been placed
inbetween some python prinouts.
By Sean McMahon
"""

import numpy as np
import fnmatch
import os
import click
import re

if __name__ == '__main__':
    # walk through directory and find .log files.
    dir_ = '/home/sean/hpc-home/Fully-Conv-Network/Resources/' + \
        'FCN_paramsearch/rgb_workers/rgb_1_23/'

    folders = os.walk(dir_)
    logfile_names = []
    for root, dirs, files in folders:
        for f in fnmatch.filter(files, '*.log'):
            logfile_names.append(os.path.join(root, f))
            print os.path.join(root, f)
    print '-' * 20
    for logfile_name in logfile_names:
        # regex matching!
        with open(logfile_name, 'r') as f:
            logfile = f.read()
        io_pattern = '[^\n]I0'
        # have read the entire logfile into memory as a string
        # find 'I0's not at the start of a line
        idx_match = re.finditer(io_pattern, logfile)
        for match in idx_match:
            print match

        # find all 'I0's
        # find those without '\n' before it?

        # copy and remove text before matched 'I0' and then add '\n' at start
        # put copied text (remove \n) at start of next line after 'I0's
