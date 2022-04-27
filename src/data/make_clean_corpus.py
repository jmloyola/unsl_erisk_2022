"""
Pre-process eRisk corpus.
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


import argparse
import os
import pandas as pd
import re

from src.config import END_OF_POST_TOKEN, PATH_INTERIM_CORPUS, MAX_SEQ_LEN_BERT
from src.utils.utilities import print_message


# Regular expressions.
UNICODE_REGEX = re.compile(r' #(?P<unicode>\d+);')
HTML_REGEX = re.compile(r'[ &](?P<html>amp|lt|gt);')
URL_FORMAT_PATTERN = re.compile(r'\[[^]]+?\]\(.+?\)')
WEB_URL_REGEX = re.compile(
    r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""")
SUB_REDDIT_REGEX = re.compile(r'/r/(?P<subreddit>[a-z0-9_]+?\b)')
NUMBER_REGEX = re.compile(r'\b[0-9]+?\b')
NOT_WORD_REGEX = re.compile(r"[^a-z0-9 ']")


def get_cleaned_post(post):
    """Clean post."""
    # Transform post to lowercase.
    clean_post = post.lower()
    # Replace unicode values with their symbol.
    clean_post = UNICODE_REGEX.sub(repl=replace_unicode, string=clean_post)
    # Replace the HTML codes with their symbol.
    clean_post = HTML_REGEX.sub(repl=replace_html_characters, string=clean_post)
    # Replace links to pages in reddit format with the token `weblink`.
    clean_post = URL_FORMAT_PATTERN.sub(repl='weblink', string=clean_post)
    # Replace direct links to pages with the token `weblink`.
    clean_post = WEB_URL_REGEX.sub(repl='weblink', string=clean_post)
    # Replace link to subreddit with the subreddit name.
    clean_post = SUB_REDDIT_REGEX.sub(repl='\g<subreddit>', string=clean_post)
    # Remove all characters except for letters, numbers and white spaces.
    clean_post = NOT_WORD_REGEX.sub(repl='', string=clean_post)
    # Replace sequence of numbers with the token `number`. This doesn't hold if
    # the sequence is not composed of numbers only.
    clean_post = NUMBER_REGEX.sub(repl='number', string=clean_post)
    # Remove repeated white spaces, new lines and tabs.
    clean_post = " ".join(clean_post.split())
    # If the document ends up empty, add the word "empty" to represent it.
    clean_post = clean_post + 'empty' if clean_post == '' else clean_post
    return clean_post


def generate_clean_corpus(corpus_name, corpus_kind, replace_old=True):
    """Pre-process the corpus.

    The pre-processing steps followed were:
        - convert text to lower case;
        - replace the decimal code for Unicode characters with its corresponding
          character;
        - replace HTML codes with their symbols;
        - replace reddit links to the web with the token weblink;
        - replace direct links to the web with the token weblink;
        - replace internal links to subreddits with the name of the subreddits;
        - delete any character that is not a number or letter;
        - replace numbers with the token number. Note that if the number is
          inside a word it is not replaced;
        - delete new lines, tab, and multiple consecutive white spaces.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous pre-process exists, if that
        is the case print an error message, otherwise, build the pre-processed
        corpus.
    """
    interim_corpus_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name)

    for stage in ['train', 'test']:
        input_corpus_path = os.path.join(interim_corpus_path, f'{corpus_name}-{stage}-raw.txt')
        output_file_name = f'{corpus_name}-{stage}-clean.txt'
        output_file_path = os.path.join(interim_corpus_path, output_file_name)
        print_message(f'Creating the corpus {output_file_name}.')

        continue_processing_this_corpus = True

        if os.path.isfile(output_file_path):
            if replace_old:
                print_message(f'Cleaning the corpus {output_file_name} previously created.')
                os.remove(output_file_path)
            else:
                print_message(f'The corpus {output_file_name} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
                continue_processing_this_corpus = False

        if continue_processing_this_corpus:
            if os.path.exists(input_corpus_path):
                with open(input_corpus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        label, document = line.split(maxsplit=1)
                        clean_document = ''
                        posts = document.split(END_OF_POST_TOKEN)
                        num_posts = len(posts)
                        for i, post in enumerate(posts):
                            clean_post = get_cleaned_post(post)
                            clean_post = clean_post + END_OF_POST_TOKEN if i < num_posts-1 else clean_post
                            clean_document = clean_document + clean_post
                        with open(output_file_path, 'a', encoding='utf-8') as f2:
                            f2.write(label + '\t' + clean_document + '\n')
            else:
                print_message(f'The corpus {input_corpus_path} does not exists.')


def trim_string(x):
    """Trim post to at most `MAX_SEQ_LEN_BERT` number of tokens."""
    x = x.split(maxsplit=MAX_SEQ_LEN_BERT)
    x = ' '.join(x[:MAX_SEQ_LEN_BERT])
    return x


def replace_unicode(match):
    """Replace unicode code value with symbol."""
    unicode_value = int(match.group('unicode'))
    return chr(unicode_value)


def replace_html_characters(match):
    """Replace HTML code value with symbol."""
    html_character = match.group('html')
    if html_character == 'amp':
        return '&'
    elif html_character == 'lt':
        return '<'
    elif html_character == 'gt':
        return '>'


def generate_csv_truncated_corpus(corpus_name, corpus_kind, replace_old=True):
    """Generate truncated corpus using the comma separated values format.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous pre-process exists, if that
        is the case print an error message, otherwise, generate the csv
        truncated corpus.
    """
    partial_input_output_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name)
    input_file_path_train = os.path.join(partial_input_output_path, f'{corpus_name}-train-raw.txt')
    input_file_path_test = os.path.join(partial_input_output_path, f'{corpus_name}-test-raw.txt')

    output_train_file_name = f'{corpus_name}_truncated_train.csv'
    output_train_file_path = os.path.join(partial_input_output_path, output_train_file_name)
    output_test_file_name = f'{corpus_name}_truncated_test.csv'
    output_test_file_path = os.path.join(partial_input_output_path, output_test_file_name)

    print_message(f'Creating the datasets {output_train_file_name} and {output_test_file_name}.')

    if os.path.isfile(output_train_file_path):
        if replace_old:
            print_message(f'Cleaning the datasets {output_train_file_name} and {output_test_file_name} previously '
                          'created.')
            os.remove(output_train_file_path)
            os.remove(output_test_file_path)
        else:
            print_message(f'The datasets {output_train_file_name} and {output_test_file_name} already exist. Delete '
                          'them beforehand or call this function with the parameter `replace_old=True`.')
            return

    labels_train = []
    documents_train = []
    if os.path.exists(input_file_path_train):
        with open(input_file_path_train, 'r') as f:
            for line in f:
                label, document = line.split(maxsplit=1)
                labels_train.append(1 if label == 'positive' else 0)

                clean_document = ''
                posts = document.split(END_OF_POST_TOKEN)
                num_posts = len(posts)
                for i, post in enumerate(posts):
                    clean_post = get_cleaned_post(post)
                    clean_post = clean_post + ' ' if i < num_posts - 1 else clean_post
                    clean_document = clean_document + clean_post

                documents_train.append(clean_document)
    else:
        print_message(f'The corpus {input_file_path_train} does not exists.')
    df_train = pd.DataFrame({'label': labels_train, 'posts': documents_train})

    labels_test = []
    documents_test = []
    if os.path.exists(input_file_path_test):
        with open(input_file_path_test, 'r') as f:
            for line in f:
                label, document = line.split(maxsplit=1)
                labels_test.append(1 if label == 'positive' else 0)

                clean_document = ''
                posts = document.split(END_OF_POST_TOKEN)
                num_posts = len(posts)
                for i, post in enumerate(posts):
                    clean_post = get_cleaned_post(post)
                    clean_post = clean_post + ' ' if i < num_posts - 1 else clean_post
                    clean_document = clean_document + clean_post

                documents_test.append(clean_document)
    else:
        print_message(f'The corpus {input_file_path_test} does not exists.')
    df_test = pd.DataFrame({'label': labels_test, 'posts': documents_test})

    # Trim the posts to the maximum length accepted by BERT.
    df_train.posts = df_train.posts.apply(trim_string)
    df_test.posts = df_test.posts.apply(trim_string)

    df_train.to_csv(output_train_file_path, index=False)
    df_test.to_csv(output_test_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to build a clean corpus.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['depression', 'gambling'])
    parser.add_argument("kind", help="eRisk task corpus kind", choices=['xml', 'reddit'])
    args = parser.parse_args()

    generate_clean_corpus(corpus_name=args.corpus, corpus_kind=args.kind, replace_old=False)
    if args.kind == 'reddit':
        generate_csv_truncated_corpus(corpus_name=args.corpus, corpus_kind=args.kind, replace_old=False)

    print_message('#' * 50)
    print_message('END OF SCRIPT')
