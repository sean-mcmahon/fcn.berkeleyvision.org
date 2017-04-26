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
import time


def log_search(walk_iterator, pattern):
    logfile_names = []

    print 'Searchin for logs'
    search_time = time.time()
    for root, dirs, files in walk_iterator:
        for f in fnmatch.filter(files, pattern):
            logfile_names.append(os.path.join(root, f))
            # print os.path.join(root, f)
    if len(logfile_names) > 10:
        print 'Search found {} logs, and took {} seconds'.format(
            len(logfile_names), time.time() - search_time)
    else:
        print 'Search found \n- {}\nAnd took {} seconds'.format(
            "\n- ".join(logfile_names), time.time() - search_time)
    print '-' * 20
    return logfile_names


def find_next_py_line(logfile, beg, debug=False):
    # Find next line without I0
    # End of the c++ print (first after line after match not starting
    # with I0)
    next_py_line = logfile.find('\n', beg)
    while True:
        if 'I0' not in logfile[next_py_line:next_py_line + 3]:
            if debug:
                print '---> First non I0 line found after', r'[^\n]I0', 'match'
                print repr(logfile[next_py_line:next_py_line + 15])
            break
        else:
            pass
        next_py_line = logfile.find('\n', next_py_line + 1)
    return next_py_line


def replace_txt(logfile_in, beg_id, end_id, pattern, new_txt="\n"):
    # add a margin arounf the ids
    begin = beg_id - 5
    end = end_id + 6
    removal_area = logfile_in[begin: end]
    bef_rem = logfile_in[:begin]
    after_rem = logfile_in[end:]
    removed_area = removal_area.replace(pattern, new_txt, 1)

    return bef_rem + removed_area + after_rem


def add_txt(logfile_in, insertion_p, new_txt):
    log_bef = logfile_in[:insertion_p]
    log_after = logfile_in[insertion_p + 1:]
    return log_bef + new_txt + log_after


def main(log_dir):
    print 'walkin...'
    walk_start = time.time()
    folders = os.walk(log_dir)
    print 'Walk took {} seconds'.format(time.time() - walk_start)

    logfile_names = log_search(folders, '*.log')

    for logfile_name in logfile_names:
        # regex matching!
        with open(logfile_name, 'r') as f:
            logfile = f.read()
        # print '+'*30, '\n', logfile, '\n', '+'*30
        io_pattern = '[^\n]I0'
        # have read the entire logfile into memory as a string
        # find 'I0's not at the start of a line
        idx_match = re.finditer(io_pattern, logfile)
        count = 0
        for count, match in enumerate(idx_match):
            # print logfile[match.start(0)]
            first_nl = logfile.rfind('\n',
                                     match.start(0) - 200, match.start(0) + 1)
            # print '---> First {} before "{}" match'.format(r'\n', r'[^\n]I0')
            # print repr(logfile[first_nl: match.end(0) + 5])
            line_beg = logfile[first_nl: match.start(0) + 1]

            newlines = find_next_py_line(logfile, match.end(0))

            # add line_beg text to next non I0 line
            new_log = add_txt(logfile, newlines, line_beg)
            # remove text at line_beg. Make sure there are no other matches
            logfile = replace_txt(new_log, first_nl, match.start(0), line_beg)

        print '{} matches found in {}'.format(count, os.path.basename(logfile_name))
        log_dr, base_name = os.path.split(logfile_name)
        with open(logfile_name + 'FIXED', 'w') as f:
            f.write(logfile)
        print 'saved ', logfile_name + 'FIXED'


@click.command()
@click.argument('log_dir', nargs=1, type=click.Path(exists=True))
def main_click(log_dir):
    # walk through directory and find .log files.
    # dir_ = '/home/sean/hpc-home/Fully-Conv-Network/Resources/' + \
    #     'FCN_paramsearch/rgb_workers/rgb_1_23/'
    # dir_ = '/home/sean/Dropbox/Uni/Code/FCN_models'
    # dir_ = '/home/sean/Documents/logfix_test/'
    start_t = time.time()
    main(log_dir)
    duration = time.time() - start_t
    print ' Log fixing took {} seconds'.format(duration)

if __name__ == '__main__':
    main_click()
