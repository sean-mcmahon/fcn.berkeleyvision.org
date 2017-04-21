"""
Found an error with the log files. The c++ printouts start inbetween the python
printouts. So this script goes and fixes them!
The chars "I0" should always start on a new line. If not it's been placed
inbetween some python prinouts.
By Sean McMahon
"""

import numpy as np
import glob
import os.path
import os.walk
import click

if __name__ == '__main__':
    # walk through directory and find .log files.
