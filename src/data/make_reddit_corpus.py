"""
Build corpus from Reddit to be used in eRisk.
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
import glob
import json
import os
import pandas as pd
import re
import requests
import shutil
from sklearn.model_selection import train_test_split
import sys
from threading import Thread, Lock
import time

from src.config import END_OF_POST_TOKEN, PATH_RAW_CORPUS, PATH_INTERIM_CORPUS
from src.utils.utilities import print_message


MIN_NUM_WRITINGS = 30
MIN_AVG_WORDS = 15

NUMBER_LIMIT = 100
USER_AGENT = 'chrome_win7_64:erisk_bot:v1 (by u/UNSL_team)'
TIMEOUT_LIMIT = 5

USER_NAME_REGEX = re.compile(r'\bu/[a-zA-Z\-0-9_]+')
BOT_REGEX = re.compile(r"(\bi(\)? \^{0,2}\(?a|')m\)? \^{0,2}\(?a\)? \^{0,2}\(?(ro)?bot\)?\b)|(this is a bot\b)",
                       flags=re.IGNORECASE)

EXCLUDED_USERS = ['[deleted]', 'AutoModerator', 'LoansBot', 'GoodBot_BadBot', 'B0tRank']
EXCLUDED_POSTS = ['[deleted]', '[removed]', 'removed by moderator']

EXCLUDED_SUBREDDITS = ['copypasta']


def request_get(url, headers, params):
    """Send a GET request and retries up to five times in case it fails.

    Parameters
    ----------
    url : str
        URL to send GET request.
    headers : dict
        HTTP headers information to send with the request. We only used
        'User-agent'.
    params : dict
        Parameters send with the request string. We used 'limit', 'after',
        'include_over_18'.

    Returns
    -------
    r : requests.Response
        Request response.
    status : int
        Status code of the request.

    Notes
    -----
    In case the request fails, the function waits for 5 seconds and retry
    the same request again. After 5 attempts, return the last failed
    result.
    If the request takes longer than the value defined in the global
    variable `TIMEOUT_LIMIT`, the request returns the status code 408.

    Examples
    --------
    >>> h = {'User-agent': 'erisk_2022'}
    >>> param = {'limit': 100}
    >>> u = 'reddit.com/top.json'
    >>> request_get(url=u, headers=h, params=param)
    (<Response [200]>, 200)
    """
    num_tries = 0
    status = 503
    r = None
    while status != 200 and num_tries < 5:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_LIMIT)
            status = r.status_code
        except requests.exceptions.Timeout:
            print_message(f'The GET request took longer than {TIMEOUT_LIMIT} seconds.')
            status = 408
        except requests.exceptions.ConnectionError:
            print_message('Maximum number of retries for the URL reached.')
            status = 429
        num_tries = num_tries + 1
        if status != 200:
            print_message('The GET request failed, trying again...')
            time.sleep(5)
    return r, status


def process_comments(replies, ids, dicts, link, current_post_time, subreddit_name):
    """Recursively process comments from post or comment.

    Store the author, time, and content of the comments of a post.

    Parameters
    ----------
    replies : dict
        Dictionary with the replies to the post or comment to process.
    ids : set
        Set of ids of the user of interest.
    dicts : dict
        Dictionary with the information of every user's post.
    link : str
        URL of the current post being processed. In case a comment it is
         being processed it makes references to the post the comment
         belongs to.
    current_post_time : int
        Time of the current post being processed. In case a comment it
        is being processed it makes references to the post the comment
        belongs to.
    subreddit_name : str
        Name of the subreddit of interest.
    """
    for element in replies['data']['children']:
        if element['kind'] == 't1':
            # It is a comment.
            author = element['data']['author']
            comment_time = element['data']['created_utc']
            comment_content = element['data']['body']
            current_subreddit = element['data']['subreddit']

            if author not in EXCLUDED_USERS and comment_content not in EXCLUDED_POSTS and \
                    BOT_REGEX.search(comment_content) is None and current_subreddit not in EXCLUDED_SUBREDDITS:
                # Remove any reference to other users.
                comment_content = USER_NAME_REGEX.sub(repl='u/erisk_anon_user', string=comment_content)

                if subreddit_name == current_subreddit:
                    ids.add(author)

                if author not in dicts:
                    dicts[author] = {}
                if link not in dicts[author]:
                    dicts[author][link] = {
                        'content': '',
                        'time': current_post_time,
                        'comments': [],
                    }
                dicts[author][link]['comments'].append({
                    'content': comment_content,
                    'time': comment_time,
                })

            if element['data']['replies'] != '':
                process_comments(replies=element['data']['replies'], ids=ids, dicts=dicts, link=link,
                                 current_post_time=current_post_time, subreddit_name=subreddit_name)


def process_post(link, ids, posts_already_processed, subreddit_name, lock, output_directory):
    """Process post from Reddit.

    Store the author, time, content and comments of a post.

    Parameters
    ----------
    link : str
        URL of the post to process.
    ids : set of str
        Set of ids of the user of interest.
    posts_already_processed : set of str
        Set of posts already processed.
    subreddit_name : str
        Name of the subreddit of interest.
    lock : threading.Lock
        A lock object to ensure correct writing of the shared resources.
    output_directory : str
        Directory path to save the auxiliary dictionary with all the
        information from the post.
    """
    len_suffix = len('.json')
    current_post_url = link[:-len_suffix]
    if current_post_url in posts_already_processed:
        return

    r, request_status_code = request_get(link, headers={'User-agent': USER_AGENT}, params={'limit': NUMBER_LIMIT})

    if request_status_code != 200:
        return

    json_r = r.json()

    # Get the post information.
    json_post = json_r[0]
    assert len(json_post['data']['children']) == 1

    post_information = json_post['data']['children'][0]

    author = post_information['data']['author']
    current_subreddit = post_information['data']['subreddit']
    post_time = post_information['data']['created_utc']
    post_title = post_information['data']['title']
    post_body = post_information['data']['selftext']
    content = post_title + ' ' + post_body
    post_identifier = post_information['data']['name']

    aux_dicts = {}
    aux_ids = set()

    if author not in EXCLUDED_USERS and post_title not in EXCLUDED_POSTS and post_body not in EXCLUDED_POSTS and \
            BOT_REGEX.search(content) is None and current_subreddit not in EXCLUDED_SUBREDDITS:
        # Remove any reference to other users.
        content = USER_NAME_REGEX.sub(repl='u/erisk_anon_user', string=content)

        if subreddit_name == current_subreddit:
            aux_ids.add(author)

        aux_dicts[author] = {}
        aux_dicts[author][link] = {
            'content': content,
            'time': post_time,
            'comments': [],
        }

    # Get comments information.
    comments_json = json_r[1]
    for element in comments_json['data']['children']:
        if element['kind'] == 't1':
            # It is a comment.
            author = element['data']['author']
            current_subreddit = element['data']['subreddit']
            comment_time = element['data']['created_utc']
            comment_content = element['data']['body']

            if author not in EXCLUDED_USERS and comment_content not in EXCLUDED_POSTS and\
                    BOT_REGEX.search(comment_content) is None and current_subreddit not in EXCLUDED_SUBREDDITS:
                # Remove any reference to other users.
                comment_content = USER_NAME_REGEX.sub(repl='u/erisk_anon_user', string=comment_content)

                if subreddit_name == current_subreddit:
                    aux_ids.add(author)
                if author not in aux_dicts:
                    aux_dicts[author] = {}

                if link not in aux_dicts[author]:
                    aux_dicts[author][link] = {
                        'content': '',
                        'time': post_time,
                        'comments': [],
                    }
                aux_dicts[author][link]['comments'].append({
                    'content': comment_content,
                    'time': comment_time,
                })

            if element['data']['replies'] != '':
                process_comments(replies=element['data']['replies'], ids=aux_ids, dicts=aux_dicts, link=link,
                                 current_post_time=post_time, subreddit_name=subreddit_name)
        else:
            continue

    # Write the post information in a file.
    aux_dicts_path = os.path.join(output_directory, f'{post_identifier}.json')
    with open(aux_dicts_path, "w") as fp:
        json.dump(fp=fp, obj=aux_dicts, indent='\t')

    # Write the data to the shared data structures.
    with lock:
        ids.union(aux_ids)
        posts_already_processed.add(current_post_url)


def sort_comments(post_comments):
    """Sort comments based on the time they were posted.

    Parameters
    ----------
    post_comments : list of dict
        List of dictionaries with comments of a user in a post.
        Each dictionary has two keys: time and content.

    Returns
    -------
    comments_sorted : list of str
        List with the user's sorted comments in a post.
    number_comments : int
        Number of comments of the user in a particular post.
    """
    number_comments = len(post_comments)
    sorted_comments_with_times = sorted(post_comments, key=lambda comment: comment['time'])
    comments_sorted = [c['content'] for c in sorted_comments_with_times]

    return comments_sorted, number_comments


def get_posts(url, main_subreddit, id_users_subreddit, posts_list, output_directory):
    """Get at most 1000 posts from an URL.

    Parameters
    ----------
    url : str
        URL to request the posts.
    main_subreddit : str
        Name of the subreddit consider for the positive class. Parameter passed
        to `process_post`.
    id_users_subreddit : set of str
        Set with the ids of the users that posted or commented. Parameter passed
        to `process_post`.
    posts_list : set of str
        Set of posts already processed. Parameter passed to `process_post`.
    output_directory : str
        Directory path to save the auxiliary dictionary with all the
        information from the post. Parameter passed to `process_post`.
    """
    id_last_post = None
    num_children = NUMBER_LIMIT
    threads = []
    write_lock = Lock()

    print_message('-·' * 40)
    print_message(f'Getting posts from: "{url}".')

    while num_children == NUMBER_LIMIT:
        get_response, status_code = request_get(url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post,
                                                'include_over_18': 'on'})

        if status_code != 200:
            num_children = 0
            continue

        json_response = get_response.json()
        num_children = len(json_response["data"]["children"])
        print_message(f'Getting the next batch of posts from "{url}".')

        for p in json_response["data"]["children"]:
            id_users_subreddit.add(p['data']['author'])
            post_permalink = 'https://www.reddit.com' + p['data']['permalink'] + '.json'

            thread = Thread(target=process_post,
                            kwargs={'link': post_permalink, 'ids': id_users_subreddit,
                                    'posts_already_processed': posts_list, 'subreddit_name': main_subreddit,
                                    'lock': write_lock, 'output_directory': output_directory})
            threads.append(thread)
            thread.start()

        if num_children > 0:
            id_last_post = json_response["data"]["children"][-1]["data"]['name']

    for t in threads:
        t.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build corpus from Reddit to be used in eRisk.")
    parser.add_argument("--corpus", help="eRisk task corpus name", choices=['depression', 'gambling'])
    parser.add_argument("--mode", help="Script mode", choices=['append', 'overwrite', 'keep'])
    args = parser.parse_args()

    main_subreddit = 'problemgambling' if args.corpus == 'gambling' else 'depression'

    partial_raw_output_path = os.path.join(PATH_RAW_CORPUS, 'reddit', args.corpus)
    post_list_path = os.path.join(partial_raw_output_path, f'reddit-{args.corpus}-post-list.json')
    id_users_path = os.path.join(partial_raw_output_path, f'reddit-{args.corpus}-id-users.json')

    os.makedirs(partial_raw_output_path, exist_ok=True)

    id_users_subreddit = set()
    posts_list = set()

    if os.path.isfile(post_list_path):
        if args.mode == 'overwrite':
            print_message('Overwrite the previously generated Reddit corpus.')
            os.remove(post_list_path)
            os.remove(id_users_path)
            # Remove all the saved posts.
            shutil.rmtree(os.path.join(partial_raw_output_path, 'main_subreddit_new'))
            shutil.rmtree(os.path.join(partial_raw_output_path, 'main_subreddit_top'))
            shutil.rmtree(os.path.join(partial_raw_output_path, 'main_subreddit_users'))
            shutil.rmtree(os.path.join(partial_raw_output_path, 'random_users'))
        elif args.mode == 'keep':
            print_message('The Reddit corpus already exists. Exiting script.')
            sys.exit()
        elif args.mode == 'append':
            print_message('The Reddit corpus already exists. Loading previously generated data to append new data.')
            with open(post_list_path, "r") as f:
                posts_list = json.load(fp=f)
                posts_list = set(posts_list)
            with open(id_users_path, "r") as f:
                id_users_subreddit = json.load(fp=f)
                id_users_subreddit = set(id_users_subreddit)

    # Get the last posts published in the subreddit of interest.
    get_url = f'https://www.reddit.com/r/{main_subreddit}/new.json'
    dict_output_directory = os.path.join(partial_raw_output_path, 'main_subreddit_new')
    os.makedirs(dict_output_directory, exist_ok=True)
    get_posts(url=get_url, main_subreddit=main_subreddit, id_users_subreddit=id_users_subreddit, posts_list=posts_list,
              output_directory=dict_output_directory)

    # Get the top posts of the subreddit of interest.
    get_url = f'https://www.reddit.com/r/{main_subreddit}/top/.json?sort=top&t=all'
    dict_output_directory = os.path.join(partial_raw_output_path, 'main_subreddit_top')
    os.makedirs(dict_output_directory, exist_ok=True)
    get_posts(url=get_url, main_subreddit=main_subreddit, id_users_subreddit=id_users_subreddit, posts_list=posts_list,
              output_directory=dict_output_directory)

    # For every user, collect their last posts published.
    dict_output_directory = os.path.join(partial_raw_output_path, 'main_subreddit_users')
    os.makedirs(dict_output_directory, exist_ok=True)
    for user_name in id_users_subreddit.copy():
        get_url = f'https://www.reddit.com/search.json?q=author:{user_name}'
        get_posts(url=get_url, main_subreddit=main_subreddit, id_users_subreddit=id_users_subreddit, posts_list=posts_list,
                  output_directory=dict_output_directory)

    # Get posts from random users for the corpus negative class.
    # We don't use the function `get_posts` because we are only interested on
    # 100 posts instead of 1000.
    print_message('-·' * 40)
    print_message('Get posts from random users for the corpus negative class.')
    general_subreddits = [
        'sports',
        'jokes',
        'gaming',
        'politics',
        'news',
        'LifeProTips',
    ]
    general_users = set()
    for sub in general_subreddits:
        # Get the last 100 posts from each subreddit.
        get_url = f'https://www.reddit.com/r/{sub}/new.json'
        id_last_post = None
        print_message(f'Getting random users from the subreddit "{sub}".')
        get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post})
        if status_code != 200:
            continue
        json_response = get_response.json()
        for p in json_response["data"]["children"]:
            current_author = p['data']['author']
            if current_author not in EXCLUDED_USERS and current_author not in id_users_subreddit:
                general_users.add(current_author)

    # For every random user, collect their last 100 posts published.
    # In this case, 100 posts are retrieved instead of 1000 because, otherwise,
    # the class imbalance would increase even more.
    dict_output_directory = os.path.join(partial_raw_output_path, 'random_users')
    os.makedirs(dict_output_directory, exist_ok=True)
    for user_name in general_users:
        get_url = f'https://www.reddit.com/search.json?q=author:{user_name}'
        print_message('-·' * 40)
        print_message(f'Getting posts from: "{get_url}".')
        id_last_post = None
        threads = []
        write_lock = Lock()
        get_response, status_code = request_get(get_url, headers={'User-agent': USER_AGENT},
                                                params={'limit': NUMBER_LIMIT, 'after': id_last_post,
                                                        'include_over_18': 'on'})
        if status_code != 200:
            continue
        json_response = get_response.json()
        for p in json_response["data"]["children"]:
            post_permalink = 'https://www.reddit.com' + p['data']['permalink'] + '.json'
            thread = Thread(target=process_post,
                            kwargs={'link': post_permalink, 'ids': id_users_subreddit,
                                    'posts_already_processed': posts_list, 'subreddit_name': main_subreddit,
                                    'lock': write_lock, 'output_directory': dict_output_directory})
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()

    print_message(f'Saving the list of processed posts in "{post_list_path}".')
    with open(post_list_path, "w") as f:
        json.dump(fp=f, obj=list(posts_list), indent='\t')

    print_message(f'Saving the list of positive users in "{id_users_path}".')
    with open(id_users_path, "w") as f:
        json.dump(fp=f, obj=list(id_users_subreddit), indent='\t')

    # Generate the processed corpus in txt.
    # Steps:
    #     - Concatenate each user posts.
    #     - Sort the posts chronologically.
    #     - Filter users with low amount of posts or banned.
    print_message('-·' * 40)
    print_message('Processing the corpus.')

    user_name_list = []
    post_url_list = []
    post_body_list = []
    post_subreddit_list = []
    post_time_list = []
    num_writings_list = []
    num_words_list = []

    len_prefix = len('https://www.reddit.com/r/')

    for file_path in glob.iglob(f'{partial_raw_output_path}/*/*.json'):
        with open(file_path, "r") as f:
            user_dicts = json.load(fp=f)
        for user_name, writings in user_dicts.items():
            for post_url, post_content in writings.items():
                idx_suffix = post_url[len_prefix:].find('/') + len_prefix

                sorted_comments, num_comments = sort_comments(post_content['comments'])

                title = post_content['content']
                join_sorted_comments = END_OF_POST_TOKEN.join(sorted_comments)
                if join_sorted_comments != '' and title != '':
                    post_body = title + END_OF_POST_TOKEN + join_sorted_comments
                else:
                    post_body = title + join_sorted_comments

                # Remove multiple white spaces, tabs and new line characters.
                post_body = ' '.join(post_body.split())
                post_body = post_body + END_OF_POST_TOKEN

                post_subreddit = post_url[len_prefix:idx_suffix]
                post_time = int(post_content['time'])
                num_writings = num_comments if post_content['content'] == '' else num_comments + 1
                num_words = len(post_body.replace(END_OF_POST_TOKEN, ' ').split())

                user_name_list.append(user_name)
                post_url_list.append(post_url)
                post_body_list.append(post_body)
                post_subreddit_list.append(post_subreddit)
                post_time_list.append(post_time)
                num_writings_list.append(num_writings)
                num_words_list.append(num_words)

    df_user_postings = pd.DataFrame({
        "user_name": user_name_list,
        "post_url": post_url_list,
        "post_body": post_body_list,
        "post_subreddit": post_subreddit_list,
        "post_time": post_time_list,
        "num_writings": num_writings_list,
        "num_words": num_words_list,
    })

    df_user_postings = df_user_postings.astype({
        "user_name": 'object',
        "post_url": 'object',
        "post_body": 'object',
        "post_subreddit": 'object',
        "post_time": 'int32',
        "num_writings": 'int32',
        "num_words": 'int32',
    })

    ag_sum = df_user_postings.groupby('user_name').sum()
    selected_users = ag_sum[ag_sum.num_writings > MIN_NUM_WRITINGS].index.to_list()
    filtered_df_user_postings = df_user_postings[df_user_postings.user_name.isin(selected_users)]
    print_message(f'Number of users with more than {MIN_NUM_WRITINGS} posts: {len(selected_users)}.')

    ag_sum = filtered_df_user_postings.groupby('user_name').sum()
    avg_num_words = ag_sum.num_words / ag_sum.num_writings
    avg_num_words = avg_num_words.astype(int)
    selected_users = avg_num_words[avg_num_words > MIN_AVG_WORDS].index.to_list()
    filtered_df_user_postings = filtered_df_user_postings[filtered_df_user_postings.user_name.isin(selected_users)]
    print_message(f'Number of users that besides have more than {MIN_AVG_WORDS} words in average per post: '
                  f'{len(selected_users)}')

    user_list = filtered_df_user_postings.user_name.unique()

    partial_interim_output_path = os.path.join(PATH_INTERIM_CORPUS, 'reddit', args.corpus)
    corpus_file_path = os.path.join(partial_interim_output_path, f'reddit-{args.corpus}-raw.txt')

    os.makedirs(os.path.dirname(corpus_file_path), exist_ok=True)

    labels = []
    documents = []
    for user in user_list:
        user_df = filtered_df_user_postings[filtered_df_user_postings.user_name == user]
        label = 'positive' if (user_df.post_subreddit == main_subreddit).any() else 'negative'
        labels.append(label)
        document = user_df.sort_values(by='post_time').post_body.sum()[:-len(END_OF_POST_TOKEN)]
        documents.append(document)

        with open(corpus_file_path, 'a', encoding='utf-8') as f:
            f.write(label + '\t' + document + '\n')

    # Split the corpus in train and test.
    print_message('-·' * 40)
    print_message('Save the generated Reddit corpus.')
    documents_train, documents_test, labels_train, labels_test = train_test_split(documents, labels, test_size=0.5,
                                                                                  stratify=labels, random_state=30)

    train_file_path = os.path.join(partial_interim_output_path, f'{args.corpus}-train-raw.txt')
    for i, document in enumerate(documents_train):
        with open(train_file_path, 'a', encoding='utf-8') as f:
            f.write(labels_train[i] + '\t' + document + '\n')

    test_file_path = os.path.join(partial_interim_output_path, f'{args.corpus}-test-raw.txt')
    for i, document in enumerate(documents_test):
        with open(test_file_path, 'a', encoding='utf-8') as f:
            f.write(labels_test[i] + '\t' + document + '\n')

    print_message('#' * 50)
    print_message('END OF SCRIPT')
