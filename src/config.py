import os


# Project common paths.
PROJECT_BASE_PATH = os.path.dirname(os.path.abspath(__file__ + '/..'))
PATH_RAW_CORPUS = os.path.join(PROJECT_BASE_PATH, 'data/raw')
PATH_INTERIM_CORPUS = os.path.join(PROJECT_BASE_PATH, 'data/interim')
PATH_PROCESSED_CORPUS = os.path.join(PROJECT_BASE_PATH, 'data/processed')

# Token used to identify the end of each post.
END_OF_POST_TOKEN = '$END_OF_POST$'

# Maximum number of token used for BERT.
MAX_SEQ_LEN_BERT = 512
# Maximum number of posts to consider for BERT.
NUM_POSTS_FOR_BERT_REP = 10

# Pickle protocol version used.
PICKLE_PROTOCOL = 4
