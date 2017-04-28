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
import psutil


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
    begin = beg_id - len(pattern)
    end = end_id + len(pattern)
    removal_area = logfile_in[begin:end]
    # bef_rem = logfile_in[:begin]
    # after_rem = logfile_in[end:]
    # removed_area = removal_area.replace(pattern, new_txt, 1)

    # instead of replace could find the indexes arounfd the text to removed
    # and create a new string from two sub strings
    # just add len(pattern) to the indecies.

    # return bef_rem + removed_area + after_rem
    return logfile_in[:begin] + removal_area.replace(pattern,
                                                     new_txt, 1) + logfile_in[end:]


def add_txt(logfile_in, insertion_p, new_txt):
    log_bef = logfile_in[:insertion_p]
    log_after = logfile_in[insertion_p + 1:]
    return log_bef + new_txt + log_after


def save_log_backup(logfile, logfile_name, debug=False):
    log_dr, base_name = os.path.split(logfile_name)
    name, ext = os.path.splitext(base_name)
    save_name = os.path.join(log_dr, name + '_BACKUP.txt')
    with open(save_name, 'w') as f:
        f.write(logfile)
    if debug:
        print 'saved backup to: ', save_name


def clean_string(logfile_str):
    # print '+'*30, '\n', logfile_str, '\n', '+'*30
    io_pattern = '[^\n]I0'
    # have read the entire logfile into memory as a string
    # find 'I0's not at the start of a line
    idx_match = re.finditer(io_pattern, logfile_str)
    m_count = 0
    for m_count, match in enumerate(idx_match):
        # print logfile_str[match.start(0)]
        first_nl = logfile_str.rfind('\n',
                                     match.start(0) - 200, match.start(0) + 1)
        # print '---> First {} before "{}" match'.format(r'\n', r'[^\n]I0')
        # print repr(logfile_str[first_nl: match.end(0) + 5])
        line_beg = logfile_str[first_nl: match.start(0) + 1]

        newlines = find_next_py_line(logfile_str, match.end(0))

        # add line_beg text to next non I0 line
        newlog = add_txt(logfile_str, newlines, line_beg)
        if psutil.swap_memory().percent >= 5.0:
            raise(Exception('Using too much memory fixed {} matches \n{} \n{}'.format(
                m_count, psutil.swap_memory(), psutil.virtual_memory())))
        # remove text at line_beg. Make sure there are no other matches
        logfile_str = replace_txt(newlog, first_nl, match.start(0), line_beg)
    return logfile_str, m_count


def main(log_dir):
    print 'walkin...'
    walk_start = time.time()
    folders = os.walk(log_dir)
    print 'Walk took {} seconds'.format(time.time() - walk_start)

    logfile_names = log_search(folders, '*.log')
    # print logfile_names
    # raise(Exception('quiting early'))
    # logfile_names = logfile_names[:5]
    for log_count, logfile_name in enumerate(logfile_names):
        # regex matching!
        print 'backing-up logfile {}/{}'.format(log_count + 1,
                                                len(logfile_names))
        with open(logfile_name, 'r') as f:
            logfile = f.read()
        save_log_backup(logfile, logfile_name, debug=False)
        print 'len of {} is {}'.format(os.path.basename(logfile_name),
                                       len(logfile))
    logfile = ''
    # raise(Exception("Quitting early"))

    for log_count, logfile_name in enumerate(logfile_names):
        # regex matching!
        print 'loading logfile {}/{}'.format(log_count + 1, len(logfile_names))
        with open(logfile_name, 'r') as f:
            logfile = f.read()

        if len(logfile) > 700000:
            logfile = logfile[:len(logfile) / 3]
            logfile, match_count = clean_string(logfile)
            # sec_h, mc2 = clean_string(sec_h)
            print '{} matches found in {}'.format(match_count + 1,
                                                  os.path.basename(logfile_name))
        else:
            logfile, match_count = clean_string(logfile)
            print '{} matches found in {}'.format(match_count + 1,
                                                  os.path.basename(logfile_name))
        save_name = logfile_name
        # with open(save_name, 'w') as f:
        #     f.write(logfile)
        print 'saved to: ', save_name
        logfile = None


@click.command()
@click.argument('log_dir', nargs=-1, type=click.Path(exists=True))
def main_click(log_dir):
    if not log_dir:
        log_dir = '/home/sean/Documents/logfix_test/hha2_11'

    # walk through directory and find .log files.
    # dir_ = '/home/sean/hpc-home/Fully-Conv-Network/Resources/' + \
    #     'FCN_paramsearch/rgb_workers/rgb_1_23/'
    # dir_ = '/home/sean/Dropbox/Uni/Code/FCN_models'
    # dir_ = '/home/sean/Documents/logfix_test/'

    start_t = time.time()
    print 'input = \n', log_dir
    print '=' * 50
    if isinstance(log_dir, tuple):
        for dir_ in log_dir:
            main(dir_)
    else:
        main(log_dir)
    duration = time.time() - start_t
    print ' Log fixing took {} seconds'.format(duration)

if __name__ == '__main__':
    main_click()
