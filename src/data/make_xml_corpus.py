"""
Build corpus from the xml files provided by the eRisk organizers.
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
from collections import Counter
import glob
import lxml.etree
import os
import re

from src.config import END_OF_POST_TOKEN, PATH_RAW_CORPUS, PATH_INTERIM_CORPUS
from src.data.make_clean_corpus import HTML_REGEX, NOT_WORD_REGEX, NUMBER_REGEX, SUB_REDDIT_REGEX, UNICODE_REGEX, URL_FORMAT_PATTERN, WEB_URL_REGEX
from src.utils.utilities import print_message


def get_ids(path_to_golden_truth):
    """
    Get the positive and negative users ids from the golden truth file.

    Parameters
    ----------
    path_to_golden_truth : str
        Path to the golden truth file.

    Returns
    -------
    positive_list : list of str
        List with nickname of every positive user.
    negative_list : list of str
        List with nickname of every negative user.
    """
    positive_list = []
    negative_list = []
    total_num_subjects = 0
    with open(path_to_golden_truth, 'r') as f:
        for line in f:
            subject, label = line.split()
            total_num_subjects = total_num_subjects + 1
            if label == '1':
                positive_list.append(subject)
            else:
                negative_list.append(subject)
    return positive_list, negative_list


def get_user_documents(xml_file_path, reverse_documents=False):
    """
    Get the user's documents.

    Given an xml file with the user's documents, generate a string with the
    title and body of every post concatenated.

    The `reverse_documents` option was added since the depression corpus
    documents were stored differently than the other corpus.
    The posts for every user in the depression corpus are stored in the reverse
    order.

    Parameters
    ----------
    xml_file_path : str
        Path to the xml file with the user's writings.
    reverse_documents : bool, default False
        Flag to indicate if the documents should be reversed. Related to a bug
        in the depression corpus.

    Returns
    -------
    documents : str
        String with the user's concatenated list of documents.
    """
    tree = lxml.etree.parse(xml_file_path)

    # Recursively search for an entry with `tag == 'TITLE'`.
    xml_titles = tree.findall('//TITLE')
    titles = []
    for tit in xml_titles:
        titles.append(tit.text)

    xml_text = tree.findall('//TEXT')
    text = []
    for t in xml_text:
        text.append(t.text)

    # XXX: Since the depression corpus has every user's post in the reverse
    #      order, that is, the most recent posts appear at the begging of the
    #      sequence while the oldest one at the end, we had to reverse the order
    #      to mantain the correct sequence.
    if reverse_documents:
        titles.reverse()
        text.reverse()

    # Each document is built using the title and the body of the publication,
    # the text, and a token to separate each document (END_OF_POST_TOKEN).
    # Repeated white spaces, tabs and new lines characters are removed.
    # A publication can have both the title and body, only the title or only the
    # body.
    documents = []
    for i in range(len(text)):
        current_title = titles[i] + ' ' if titles[i] is not None else ' '
        current_text = text[i] if text[i] is not None else ' '
        current_document = current_title + current_text
        # Remove multiple white spaces, tabs and new line characters.
        current_document = ' '.join(current_document.split())
        current_document = current_document + END_OF_POST_TOKEN
        documents.append(current_document)
    documents = ''.join(documents)
    # Remove the last END_OF_POST_TOKEN.
    documents = documents[:-len(END_OF_POST_TOKEN)]

    return documents


def generate_raw_gambling_corpus(replace_old=True):
    """
    Generate the raw corpus for gambling.

    The corpus is composed of the title and body of text of every user's posts
    without pre-processing.

    Since the provided compress files had different structure, separated
    functions were generated for each corpus.

    The gambling corpus structure is:
        t1_training
        └── TRAINING_DATA
            └── 2021_cases
                ├── data
                |   └── *.xml
                └── risk_golden_truth.txt

    Parameters
    ----------
    replace_old : bool, default True
        Flag to indicate if the previously generated corpus should be replaced
        or not.

    Returns
    -------
    output_file_paths : list of str
        List of paths to the generated corpus.
    """
    raw_corpus_path = os.path.join(PATH_RAW_CORPUS, 'xml', 'gambling')
    interim_corpus_path = os.path.join(PATH_INTERIM_CORPUS, 'xml', 'gambling')

    corpus_dir = os.path.join(raw_corpus_path, 't1_training', 'TRAINING_DATA', '2021_cases')
    output_file_name = 'gambling-test-raw.txt'
    output_file_path = os.path.join(interim_corpus_path, output_file_name)
    os.makedirs(interim_corpus_path, exist_ok=True)

    golden_truth_path = os.path.join(corpus_dir, 'risk_golden_truth.txt')

    print_message(f'Generating the corpus {output_file_name} ...')

    continue_processing_this_corpus = True

    if os.path.isfile(output_file_path):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name} previously generated.')
            os.remove(output_file_path)
        else:
            print_message(f'The corpus {output_file_name} already exists. Delete it before running the script or call '
                          'this function with the option `replace_old=True`.')
            continue_processing_this_corpus = False

    if continue_processing_this_corpus:
        positive_list, negative_list = get_ids(golden_truth_path)
        for label in ['positive', 'negative']:
            ids = positive_list if label == 'positive' else negative_list
            for sub in ids:
                xml_file_name = sub + '.xml'
                xml_file_path = os.path.join(corpus_dir, 'data', xml_file_name)
                documents = get_user_documents(xml_file_path)
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(label + '\t' + documents + '\n')
    return [output_file_path]


def generate_raw_depression_corpus(replace_old=True):
    """
    Generate the raw corpus for depression.

    The corpus is composed of the title and body of text of every user's posts
    without pre-processing.

    Since the provided compress files had different structure, separated
    functions were generated for each corpus.

    The depression corpus structure is:
        training_t2
        └── TRAINING_DATA
            ├── 2017_cases
            │   ├── neg
            │   └── pos
            └── 2018_cases
                ├── neg
                └── pos

    Parameters
    ----------
    replace_old : bool, default True
        Flag to indicate if the previously generated corpus should be replaced
        or not.

    Returns
    -------
    output_file_paths : list of str
        List of paths to the generated datasets.
    """
    raw_corpus_path = os.path.join(PATH_RAW_CORPUS, 'xml', 'depression')
    interim_corpus_path = os.path.join(PATH_INTERIM_CORPUS, 'xml', 'depression')

    corpus_dir = os.path.join(raw_corpus_path, 'training_t2', 'TRAINING_DATA')
    output_file_name_train = 'depression-train-raw.txt'
    output_file_path_train = os.path.join(interim_corpus_path, output_file_name_train)
    os.makedirs(interim_corpus_path, exist_ok=True)

    print_message(f'Generating the corpus {output_file_name_train} using the data from 2017 ...')

    continue_processing_this_corpus = True

    if os.path.isfile(output_file_path_train):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name_train} previously generated.')
            os.remove(output_file_path_train)
        else:
            print_message(f'The corpus {output_file_name_train} already exists. Delete it before running the script or '
                          'call this function with the option `replace_old=True`.')
            continue_processing_this_corpus = False

    if continue_processing_this_corpus:
        for label in ['positive', 'negative']:
            xml_paths = glob.glob(f'{corpus_dir}/2017_cases/{label[:3]}/*.xml')

            for xml_file_path in xml_paths:
                documents = get_user_documents(xml_file_path, reverse_documents=True)
                with open(output_file_path_train, 'a', encoding='utf-8') as f:
                    f.write(label + '\t' + documents + '\n')

    output_file_name_test = 'depression-test-raw.txt'
    output_file_path_test = os.path.join(interim_corpus_path, output_file_name_test)

    output_paths_list = [output_file_path_train, output_file_path_test]

    print_message(f'Generating the corpus {output_file_name_test} using the data from 2018 ...')

    if os.path.isfile(output_file_path_test):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name_test} previously generated.')
            os.remove(output_file_path_test)
        else:
            print_message(f'The corpus {output_file_name_test} already exists. Delete it before running the script or '
                          'call this function with the option `replace_old=True`.')
            return output_paths_list

    for label in ['positive', 'negative']:
        xml_paths = glob.glob(f'{corpus_dir}/2018_cases/{label[:3]}/*.xml')

        for xml_file_path in xml_paths:
            documents = get_user_documents(xml_file_path, reverse_documents=True)
            with open(output_file_path_test, 'a', encoding='utf-8') as f:
                f.write(label + '\t' + documents + '\n')
    return output_paths_list


def check_regex(path):
    """
    Check presence of certain regular expressions in the generate corpus.

    The regular expressions consider are: URLs, web URLs, subreddits links,
    number tokens and not words tokens.

    For each, report the 20 most common.

    Parameters
    ----------
    path : str
        Path to the generate corpus to check.
    """
    urls = []
    web_urls = []
    sub_reddits = []
    numbers = []
    not_words = []

    print_message(f'Checking the regular expression of {path} ...')

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            _, document = line.split(maxsplit=1)
            document = document.lower()
            urls.extend(re.findall(URL_FORMAT_PATTERN, document))
            web_urls.extend(re.findall(WEB_URL_REGEX, document))
            sub_reddits.extend(re.findall(SUB_REDDIT_REGEX, document))
            numbers.extend(re.findall(NUMBER_REGEX, document))
            not_words.extend(re.findall(NOT_WORD_REGEX, document))

    urls_counter = Counter(urls)
    web_urls_counter = Counter(web_urls)
    sub_reddits_counter = Counter(sub_reddits)
    numbers_counter = Counter(numbers)
    not_words_counter = Counter(not_words)

    print_message('********** urls **********')
    for urls, repetitions in urls_counter.most_common(20):
        print_message(f'{urls} -> {repetitions}')
    print_message('********** web_urls **********')
    for web_urls, repetitions in web_urls_counter.most_common(20):
        print_message(f'{web_urls} -> {repetitions}')
    print_message('********** sub_reddits **********')
    for sub_reddits, repetitions in sub_reddits_counter.most_common(20):
        print_message(f'{sub_reddits} -> {repetitions}')
    print_message('********** numbers **********')
    for numbers, repetitions in numbers_counter.most_common(20):
        print_message(f'{numbers} -> {repetitions}')
    print_message('********** not_words **********')
    for not_words, repetitions in not_words_counter.most_common(20):
        print_message(f'{not_words} -> {repetitions}')


def check_unicode_characters_used(path):
    """
    Check for special character not correctly obtained in the generate corpus.

    There were some characters that were not correctly stored in the xml files.
    Or, at least, their presence was not documented.
    Some unicode characters and HTML codes were not correctly stored, having
    a numeric code instead of the character they represent.

    For this two type, report the 20 most common.

    Parameters
    ----------
    path : str
        Path to the generate corpus to check.
    """
    unicode_characters = []
    html_characters = []
    print_message(f'Checking the unicode characters of {path} ...')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label, document = line.split(maxsplit=1)
            document = document.lower()
            unicode_characters.extend(re.findall(UNICODE_REGEX, document))
            html_characters.extend(re.findall(HTML_REGEX, document))

    unicode_characters_counter = Counter(unicode_characters)
    html_characters_counter = Counter(html_characters)

    print_message('********** unicode characters **********')
    for unicode, repetitions in unicode_characters_counter.most_common():
        print_message(f'{unicode} -> {chr(int(unicode))} -> {repetitions}')

    print_message('********** HTML codes **********')
    for html, repetitions in html_characters_counter.most_common():
        print_message(f'{html} -> {repetitions}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build corpus from the xml files provided by the eRisk organizers.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['depression', 'gambling'])
    args = parser.parse_args()

    if args.corpus == 'depression':
        output_paths_list = generate_raw_depression_corpus(replace_old=False)
    else:
        output_paths_list = generate_raw_gambling_corpus(replace_old=False)
    for path in output_paths_list:
        check_unicode_characters_used(path)
        check_regex(path)

    print_message('#' * 50)
    print_message('END OF SCRIPT')
