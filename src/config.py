import os

# Project common paths.
PROJECT_BASE_PATH = os.path.dirname(os.path.abspath(__file__ + "/.."))
PATH_RAW_CORPUS = os.path.join(PROJECT_BASE_PATH, "data/raw")
PATH_INTERIM_CORPUS = os.path.join(PROJECT_BASE_PATH, "data/interim")
PATH_PROCESSED_CORPUS = os.path.join(PROJECT_BASE_PATH, "data/processed")
PATH_MODELS = os.path.join(PROJECT_BASE_PATH, "models")
PATH_BEST_MODELS = os.path.join(PROJECT_BASE_PATH, "best_models")
PATH_REPORTS = os.path.join(PROJECT_BASE_PATH, "reports")
PATH_COMPETITION = os.path.join(PROJECT_BASE_PATH, "competition")
PATH_DEPLOY = os.path.join(PATH_COMPETITION, "models")

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

# TODO: You need to update this.
# File name of representations used by the best EarlyModels with doc2vec.
BEST_DOC2VEC_REP = {
    "depression": "04_representation_doc2vec.pkl",
    "gambling": "03_representation_doc2vec.pkl",
}
