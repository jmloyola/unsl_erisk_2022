"""
Utilities used in other scripts.
Copyright (C) 2022 Juan Martín Loyola

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import datetime
import json
import numpy as np
import os
import sys
import time

from src.config import END_OF_POST_TOKEN


def print_message(msg, file=sys.stdout):
    """
    Print message log to text stream file including the current time.

    Parameters
    ----------
    msg : str
        Message to print.
    file : text stream file, default=sys.stdout
        Text stream file.
    """
    print(datetime.datetime.now(), '|', f'{msg}', file=file)


def print_elapsed_time(prefix=''):
    """
    Print the elapsed time between each call to this function.

    If this function is call for the first time, store the current time.
    The following calls to this function report (print) the number of minutes
    and seconds elapsed.

    Parameters
    ----------
    prefix : str, default=''
        Prefix to include in the elapsed time report.
    
    Returns
    -------
    elapsed_mins : int
        Elapsed minutes since this function was last called. If this is the
        first time this function was called, return `None`.
    elapsed_secs : int
        Elapsed seconds since this function was last called. If this is the
        first time this function was called, return `None`.
    """
    e_time = time.time()
    if not hasattr(print_elapsed_time, 's_time'):
        print_elapsed_time.s_time = e_time
        return None, None
    else:
        elapsed_time = e_time - print_elapsed_time.s_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        print_message(f'{prefix} elapsed time: {elapsed_mins}m {elapsed_secs}s')
        print_elapsed_time.s_time = e_time
        return elapsed_mins, elapsed_secs


def get_documents_from_corpus(path, corpus_info_path=None):
    """
    Get the documents and labels from the corpus.

    Parameters
    ----------
    path : str
        Corpus path.
    corpus_info_path : str, default=None
        Path used to store the corpus information. If the path is provided and
        the file does not exist, save the path, the number of posts per user and
        the median number of posts per user of the corpus.
        If this parameter is not provided, the corpus information is not saved.

    Returns
    -------
    documents : list of str
        Users' posts.
    labels : list of int
        Users' label.
    """
    documents = []
    labels = []
    num_post_users = []
    with open(path, 'r') as f:
        for line in f:
            label, document = line.split(maxsplit=1)
            label = 1 if label == 'positive' else 0
            documents.append(document)
            labels.append(label)
            num_posts_current_user = len(document.split(END_OF_POST_TOKEN))
            num_post_users.append(num_posts_current_user)
    print_message(f'num_post_users for {path}: {num_post_users}')

    corpus_info = {
        'path': path,
        'num_post_users': num_post_users,
        'median_num_post_users': np.median(num_post_users),
    }
    if corpus_info_path is not None and not os.path.exists(corpus_info_path):
        with open(corpus_info_path, "w") as f:
            json.dump(fp=f, obj=corpus_info, indent='\t')
    return documents, labels
