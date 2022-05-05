import os


# Project common paths.
PROJECT_BASE_PATH = os.path.dirname(os.path.abspath(__file__ + "/.."))
PATH_RAW_CORPUS = os.path.join(PROJECT_BASE_PATH, "data/raw")
PATH_INTERIM_CORPUS = os.path.join(PROJECT_BASE_PATH, "data/interim")
PATH_PROCESSED_CORPUS = os.path.join(PROJECT_BASE_PATH, "data/processed")
PATH_MODELS = os.path.join(PROJECT_BASE_PATH, "models")

# Token used to identify the end of each post.
END_OF_POST_TOKEN = "$END_OF_POST$"

# Maximum number of token used for BERT.
MAX_SEQ_LEN_BERT = 512
# Maximum number of posts to consider for BERT.
NUM_POSTS_FOR_BERT_REP = 10

# Pickle protocol version used.
PICKLE_PROTOCOL = 4

# Number of digits to consider for the elapsed times floating points.
FP_PRECISION_ELAPSED_TIMES = 3

# EARLIEST parameters.
MAX_SEQ_LENGTH = 300
BATCH_SIZE = 140
