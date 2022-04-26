import os


# Project common paths.
PROJECT_BASE_PATH = os.path.dirname(os.path.abspath(__file__ + '/..'))
PATH_RAW_CORPUS = os.path.join(PROJECT_BASE_PATH, 'data/raw')
PATH_INTERIM_CORPUS = os.path.join(PROJECT_BASE_PATH, 'data/interim')

# Token used to identify the end of each post.
END_OF_POST_TOKEN = '$END_OF_POST$'

MAX_SEQ_LEN_BERT = 512