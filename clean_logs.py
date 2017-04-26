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
    # dir_ = '/home/sean/Dropbox/Uni/Code/FCN_models'

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
        # print '+'*30, '\n', logfile, '\n', '+'*30
        io_pattern = '[^\n]I0'
        # have read the entire logfile into memory as a string
        # find 'I0's not at the start of a line
        idx_match = re.finditer(io_pattern, logfile)
        for count, match in enumerate(idx_match):
            # print logfile[match.start(0)]
            first_nl = logfile.rfind('\n',
                                     match.start(0) - 200, match.start(0) + 1)
            print '---> First {} before "{}" match'.format(r'\n', r'[^\n]I0')
            print repr(logfile[first_nl: match.end(0) + 5])
            line_beg = logfile[first_nl: match.start(0) + 1]
            print '{}'.format(repr(line_beg))

            # Find next line without I0
            # End of the c++ print (first after line after match not starting
            # with I0)
            newlines = logfile.find('\n', match.end(0))
            while True:
                if 'I0' not in logfile[newlines:newlines + 3]:
                    print '---> First non I0 line found after', r'[^\n]I0', 'match'
                    print repr(logfile[newlines:newlines + 15])
                    break
                else:
                    pass
                newlines = logfile.find('\n', newlines + 1)

            # add line_beg text to next non I0 line
            log_bef = logfile[:newlines]
            log_after = logfile[newlines + 1:]
            new_log = log_bef + line_beg + log_after
            # remove text at line_beg. Make sure there are no other matches
            logfile = new_log.replace(line_beg, "\n", 1)
            # line begin should start with a "\n"
            # print '+' * 30
            # print new_log2
        break
    print '{} matches found (out of 17)'.format(count)
    with open('/home/sean/Dropbox/Uni/Code/FCN_models/new_logfile.log', 'w') as f:
        f.write(logfile)
