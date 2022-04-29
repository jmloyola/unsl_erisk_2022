"""
Get the features for each model.
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
import gensim
import glob
import json
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import torch
from torch.utils.data import TensorDataset
from torchtext.data import Field, Example, Dataset

from src.config import PATH_INTERIM_CORPUS, PATH_PROCESSED_CORPUS, PICKLE_PROTOCOL, MAX_SEQ_LEN_BERT, NUM_POSTS_FOR_BERT_REP, END_OF_POST_TOKEN
from src.utils.utilities import print_message, have_same_parameters


def get_bow_representation(documents, count_vect, tfidf_transformer):
    """Get the Bag of Words (BoW) representation for the users' documents.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `config.END_OF_POST_TOKEN`.
    count_vect : sklearn.feature_extraction.text.CountVectorizer
        The trained scikit-learn CountVectorizer to use.
    tfidf_transformer : sklearn.feature_extraction.text.TfidfTransformer
        The trained scikit-learn TfidfTransformer to use.

    Returns
    -------
    x_tfidf : sparse matrix
        The tf or tf-idf representation of the users' posts.
    """
    concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ') for user_posts in documents]
    x_counts = count_vect.transform(concat_posts)
    x_tfidf = tfidf_transformer.transform(x_counts)
    return x_tfidf


def get_lda_representation(documents, lda_model, id2word, bigram_model):
    """Get the Latent Dirichlet Allocation (LDA) representation for the users' documents.

    Each users is represented based on the topics distribution of its documents.
    The topics are learnt using LDA.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `config.END_OF_POST_TOKEN`.
    lda_model : gensim.models.LdaModel
        The trained gensim LDA model to use.
    id2word : dict of (int, str)
        Mapping from word IDs to words.
    bigram_model : gensim.models.phrases.Phraser
        The trained gensim Phraser model to use.

    Returns
    -------
    x_lda : numpy.ndarray of shape (n_users, n_topics)
        The LDA topics distribution of the users' posts.
    """
    concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ').split() for user_posts in documents]
    bigrams = [bigram_model[post] for post in concat_posts]
    transformed_corpus = [id2word.doc2bow(text) for text in bigrams]

    x_lda = []
    for transformed_doc in transformed_corpus:
        top_topics = lda_model.get_document_topics(transformed_doc, minimum_probability=0.0)
        topic_vec = [topic_tuple[1] for topic_tuple in top_topics]
        x_lda.append(topic_vec)
    return np.array(x_lda, dtype=np.float32)


def get_lsa_representation(documents, lsa_model, id2word, bigram_model):
    """Get the Latent Semantic Analysis (LSA) representation for the users' documents.

    Each users is represented based on the factors distribution of its documents.
    The factors (latent dimensions) are learnt using LSA.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `config.END_OF_POST_TOKEN`.
    lsa_model : gensim.models.LsiModel
        The trained gensim LSA model to use.
    id2word : dict of (int, str)
        Mapping from word IDs to words.
    bigram_model : gensim.models.phrases.Phraser
        The trained gensim Phraser model to use.

    Returns
    -------
    x_lsa : numpy.ndarray of shape (n_users, n_topics)
        The LSA factors distribution of the users' posts.
    """
    concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ').split() for user_posts in documents]
    bigrams = [bigram_model[post] for post in concat_posts]
    transformed_corpus = [id2word.doc2bow(text) for text in bigrams]

    x_lsa = np.zeros((len(documents), lsa_model.num_topics), dtype=np.float32)

    vectorized_corpus_train = lsa_model[transformed_corpus]

    for j, vector in enumerate(vectorized_corpus_train):
        for dim_number, dim_value in vector:
            x_lsa[j, dim_number] = dim_value
    return x_lsa


def get_doc2vec_representation(documents, doc2vec_model, sequential=False, max_sequence_length=None,
                               is_competition=False):
    """Get the doc2vec representation for the users' documents.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `config.END_OF_POST_TOKEN`.
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The trained doc2vec model used to infer the vector representation
        of each post.
    sequential : bool, default=False
        A flag to indicate if the input should be represented as a sequence
        of posts or as one big post for each user.
        If `sequential=True` a document embedding will be inferred for each post
        of a user.
    max_sequence_length : int, default=None
        The maximum sequence length, i.e., the maximum number of posts
        allowed for each user. User to limit the size of the
        representation in memory.
        Used only when `sequential=True` and `is_competition=False`, that
        is, during training for the competition.
    is_competition : bool, default=False
        A flag to indicate if the current representation is to be used
        during the competition or not.

    Returns
    -------
    x_doc2vec : ndarray
        The doc2vec representation of the users' posts.
    """
    if sequential:
        if is_competition:
            max_num_post = max([len(posts.split(END_OF_POST_TOKEN)) for posts in documents])
            users_posts_truncated = [user_posts.split(END_OF_POST_TOKEN) for user_posts in documents]
            x_doc2vec = np.zeros((len(documents), max_num_post, doc2vec_model.vector_size), dtype=np.float32)
            for j, posts in enumerate(users_posts_truncated):
                for k, current_post in enumerate(posts):
                    # In case the user does not have so many posts, we add an empty post.
                    if current_post == '':
                        continue
                    x_doc2vec[j, k, :] = doc2vec_model.infer_vector(current_post.split())
            return x_doc2vec
        else:
            assert max_sequence_length is not None
            max_num_post = max([len(posts.split(END_OF_POST_TOKEN)) for posts in documents])
            seq_lim = max_sequence_length if max_sequence_length < max_num_post else max_num_post
            users_posts_truncated = [user_posts.split(END_OF_POST_TOKEN)[:seq_lim] for user_posts in documents]
            x_doc2vec = np.zeros((len(documents), seq_lim, doc2vec_model.vector_size), dtype=np.float32)
            for j, posts in enumerate(users_posts_truncated):
                for k, current_post in enumerate(posts):
                    x_doc2vec[j, k, :] = doc2vec_model.infer_vector(current_post.split())
            return x_doc2vec
    else:
        concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ').split() for user_posts in documents]
        x_doc2vec = np.zeros((len(documents), doc2vec_model.vector_size), dtype=np.float32)
        for j, post in enumerate(concat_posts):
            x_doc2vec[j, :] = doc2vec_model.infer_vector(post)
        return x_doc2vec


def get_padded_sequential_representation(documents, vocab_to_int, seq_len):
    """Get the padded sequential representation for the users' documents.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `config.END_OF_POST_TOKEN`.
    vocab_to_int : dict of (str, int)
        Mapping from words to word IDs.
    seq_len : int
        Sequence length.

    Returns
    -------
    padded_sequential_documents : torch.utils.data.TensorDataset
        The padded sequential representation of the users' posts.
    """
    concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ').split() for user_posts in documents]
    encoded_documents = [[vocab_to_int.get(word, 1) for word in post] for post in concat_posts]
    padded_documents = pad_text(encoded_documents=encoded_documents, sequence_length=seq_len)

    padded_sequential_documents = TensorDataset(padded_documents)
    return padded_sequential_documents


def get_bert_representation(documents, tokenizer):
    """Get the BERT representation for the users' documents.

    Since BERT has a limit on the length it can process, we restrict the number
    of posts from which we get the representation.
    In this case, we arbitrarily keep the latest `config.NUM_POSTS_FOR_BERT_REP`
    posts.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `config.END_OF_POST_TOKEN`.
    tokenizer : transformers.tokenization_utils.PreTrainedTokenizer
        BERT tokenizer to use.

    Returns
    -------
    test_data : torch.utils.data.TensorDataset
        BERT representation of the users' posts.
    """
    concat_posts = [' '.join(user_posts.split(END_OF_POST_TOKEN)[-NUM_POSTS_FOR_BERT_REP:]) for user_posts in documents]

    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    posts_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                        batch_first=True, fix_length=MAX_SEQ_LEN_BERT, pad_token=pad_index, unk_token=unk_index)
    fields = [('posts', posts_field)]

    example_list = [Example.fromlist(data=[d], fields=fields) for d in concat_posts]
    test_data = Dataset(examples=example_list, fields=fields)

    return test_data


def generate_bow_corpus(corpus_name, corpus_kind, replace_old=True, cv_params=None, transformer_params=None):
    """Generate the corpus' Bag of Words (BoW) representation.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous BoW corpus exists, if that
        is the case print an error message, otherwise, build the BoW
        corpus.
    cv_params : dict
        Dictionary with the parameters for `sklearn.feature_extraction.text.CountVectorizer`.
    transformer_params : dict
        Dictionary with the parameters for `sklearn.feature_extraction.text.TfidfTransformer`.
    """
    print_message(f"Generating the corpus {corpus_kind}/{corpus_name} using the Bag of Words representation.")

    if cv_params['analyzer'] == 'word' and cv_params['ngram_range'] in [(3, 3), (4, 4)]:
        print_message('To avoid generating very large corpus that when training fill the memory, we do not generate '
                      f'this corpus (analyzer={cv_params["analyzer"]} and ngram_range={cv_params["ngram_range"]}).')
        return

    cv_params = {} if cv_params is None else cv_params.copy()
    transformer_params = {} if transformer_params is None else transformer_params.copy()

    count_vect = CountVectorizer(**cv_params)
    tfidf_transformer = TfidfTransformer(**transformer_params)

    current_parameters_dict = {
        'CountVectorizer_params': count_vect.get_params(),
        'TfidfTransformer_params': tfidf_transformer.get_params(),
    }

    if 'dtype' in current_parameters_dict.get('CountVectorizer_params'):
        new_value = str(current_parameters_dict['CountVectorizer_params']['dtype'])
        current_parameters_dict['CountVectorizer_params']['dtype'] = new_value

    partial_output_path = os.path.join(PATH_PROCESSED_CORPUS, corpus_kind, corpus_name, 'bow')
    possible_files = glob.glob(f'{partial_output_path}/{corpus_name}_bow_*.json')
    max_id = 0
    current_id = 0
    already_exists = False
    for file in possible_files:
        current_id = int(file[-7:-5])
        if current_id > max_id:
            max_id = current_id
        already_exists = have_same_parameters(current_parameters_dict, file)
        if already_exists:
            if replace_old:
                print_message(f'Cleaning the corpus {file[:-5]} previously created.')
                os.remove(file)
                pickle_file_train = file[:-5] + '_train.pkl'
                pickle_file_test = file[:-5] + '_test.pkl'
                vocabulary_file = file[:-5] + '_vocabulary.pkl'
                os.remove(pickle_file_train)
                os.remove(pickle_file_test)
                os.remove(vocabulary_file)
            else:
                print_message(f'The corpus {file[:-5]} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
                return
            break
    id_number = current_id if already_exists else max_id + 1

    partial_input_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name)
    input_file_path_train = os.path.join(partial_input_path, f'{corpus_name}-train-clean.txt')
    input_file_path_test = os.path.join(partial_input_path, f'{corpus_name}-test-clean.txt')

    output_pkl_train_name = f'{corpus_name}_bow_{id_number:02d}_train.pkl'
    output_pkl_test_name = f'{corpus_name}_bow_{id_number:02d}_test.pkl'
    output_json_name = f'{corpus_name}_bow_{id_number:02d}.json'
    output_vocabulary_name = f'{corpus_name}_bow_{id_number:02d}_vocabulary.pkl'
    output_features_models_name = f'{corpus_name}_bow_{id_number:02d}_features_models.pkl'
    output_pkl_train_path = os.path.join(partial_output_path, output_pkl_train_name)
    output_pkl_test_path = os.path.join(partial_output_path, output_pkl_test_name)
    output_json_path = os.path.join(partial_output_path, output_json_name)
    output_vocabulary_path = os.path.join(partial_output_path, output_vocabulary_name)
    output_feature_models_path = os.path.join(partial_output_path, output_features_models_name)

    # Make the corpus directory if it does not exists.
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    print_message(f'Saving the corpus configuration at "{output_json_path}".')
    with open(output_json_path, "w") as f:
        json.dump(fp=f, obj=current_parameters_dict, indent='\t')

    labels_train = []
    documents_train = []
    with open(input_file_path_train, 'r') as f:
        for line in f:
            label, document = line.split(maxsplit=1)
            label = 1 if label == 'positive' else 0
            labels_train.append(label)
            posts = ' '.join(document.split(END_OF_POST_TOKEN))
            documents_train.append(posts)

    x_train_counts = count_vect.fit_transform(documents_train)

    print_message(f'BoW training matrix shape: {x_train_counts.shape}')

    print_message('Saving the vocabulary used by BoW.')
    with open(output_vocabulary_path, 'wb') as fp:
        pickle.dump((count_vect.vocabulary_, count_vect.stop_words_), fp, protocol=PICKLE_PROTOCOL)

    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    y_train = np.array(labels_train, dtype=np.float32)

    print_message('Saving the BoW models (CountVectorizer and TfidfTransformer).')
    with open(output_feature_models_path, 'wb') as fp:
        pickle.dump((count_vect, tfidf_transformer), fp, protocol=PICKLE_PROTOCOL)

    labels_test = []
    documents_test = []
    with open(input_file_path_test, 'r') as f:
        for line in f:
            label, document = line.split(maxsplit=1)
            label = 1 if label == 'positive' else 0
            labels_test.append(label)
            posts = ' '.join(document.split(END_OF_POST_TOKEN))
            documents_test.append(posts)

    x_test_counts = count_vect.transform(documents_test)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    y_test = np.array(labels_test, dtype=np.float32)

    print_message(f'Saving the generated datasets at "{output_pkl_train_path}" and "{output_pkl_test_path}".')
    with open(output_pkl_train_path, 'wb') as fp:
        pickle.dump((x_train_tfidf, y_train), fp, protocol=PICKLE_PROTOCOL)

    with open(output_pkl_test_path, 'wb') as fp:
        pickle.dump((x_test_tfidf, y_test), fp, protocol=PICKLE_PROTOCOL)


def get_words_list_from_corpus(input_file_path):
    """Get the words list from the corpus.

    Parameters
    ----------
    input_file_path : str
        Path to the corpus.

    Returns
    -------
    all_posts : list of list of str
        List containing the users' list of words of each posts.
    all_labels : list of int
        List containing the users' labels.
    all_words : list of str
        List containing all the corpus' words.
    """
    all_posts = []
    all_labels = []
    all_words = []
    with open(input_file_path, 'r') as f:
        for line in f:
            label, document = line.split(maxsplit=1)
            user_posts = ' '.join(document.split(END_OF_POST_TOKEN))
            words = user_posts.split()
            all_posts.append(words)
            all_words.extend(words)
            label = 1 if label == 'positive' else 0
            all_labels.append(label)
    return all_posts, all_labels, all_words


def get_bigrams_model(words, min_count=15):
    """Get the bigram model.

    Parameters
    ----------
    words : list of list of str
        List containing the users' list of words of each posts.
    min_count : int, default=15
        Minimum number of times a bigram has to appear to be consider.

    Returns
    -------
    bigram_mod : gensim.models.phrases.Phraser
        Bigram model.
    """
    bigram = gensim.models.Phrases(words, min_count=min_count)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_corpus_id2word(posts, bigram_model=None, id2word=None):
    """Get the mapping from word IDs to words from the corpus.

    When `bigram_model` and `id2word` are not provided (`None` by default) the
    bigram model and the mapping from word IDs to words is calculated based on
    the corpus. Then, they are used to transform the corpus.
    Otherwise, the trained bigram model and mapping are used to transform the
    current corpus.

    Parameters
    ----------
    posts : list of list of str
        List containing the users' list of words of each posts.
    bigram_model : gensim.models.phrases.Phraser, default=None
        Trained gensim Phraser model. If `bigram_model is None` the model is
        trained, otherwise the passed model is used.
    id2word : dict of (int, str), default=None
        Mapping from word IDs to words. If `id2word is None` the model is
        trained, otherwise the passed model is used.

    Returns
    -------
    transformed_corpus : list of list of str
        Transformed corpus.
    id2word : dict of (int, str)
        The trained mapping from word IDs to words.
    bigram_model : gensim.models.phrases.Phraser
        The trained gensim Phraser model.
    """
    if bigram_model is None:
        bigram_model = get_bigrams_model(posts)
    else:
        bigram_model = bigram_model
    bigrams = [bigram_model[post] for post in posts]

    if id2word is None:
        id2word = gensim.corpora.Dictionary(bigrams)
        id2word.filter_extremes(no_below=10, no_above=0.6)
        id2word.compactify()
    else:
        id2word = id2word
    transformed_corpus = [id2word.doc2bow(text) for text in bigrams]
    return transformed_corpus, id2word, bigram_model


def get_lda_model(corpus_name, corpus_kind, number_topics, number_passes):
    """Get the Latent Dirichlet Allocation (LDA) model.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    number_topics : int
        Number of requested latent topics to be extracted from the training
        corpus.
    number_passes : int
        Number of passes through the corpus during training.

    Returns
    -------
    lda_model : gensim.models.LdaModel
        The trained gensim LDA model.
    transformed_corpus : list of list of str
        Transformed corpus.
    labels : list of int
        List containing the users' labels.
    id2word : dict of (int, str)
        The trained mapping from word IDs to words.
    bigram_model : gensim.models.phrases.Phraser
        The trained gensim Phraser model.
    """
    print_message(f"Training the LDA model for {corpus_kind}/{corpus_name} with {number_topics} topics.")
    input_file_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name, f'{corpus_name}-train-clean.txt')

    posts, labels, _ = get_words_list_from_corpus(input_file_path)
    transformed_corpus, id2word, bigram_model = get_corpus_id2word(posts)

    lda_model = gensim.models.LdaModel(
        corpus=transformed_corpus,
        num_topics=number_topics,
        id2word=id2word,
        chunksize=100,
        passes=number_passes,
        eval_every=1,
        random_state=30,
        per_word_topics=True)

    print_message(f"Latent topics extracted:")
    for topic_id, topics_words in lda_model.print_topics(num_topics=-1, num_words=20):
        print_message(f'Topic id: {topic_id} -> {topics_words}')

    return lda_model, transformed_corpus, labels, id2word, bigram_model


def generate_lda_corpus(corpus_name, corpus_kind, replace_old=True, number_topics=20, number_passes=20):
    """Generate the corpus' Latent Dirichlet Allocation (LDA) representation.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous LDA corpus exists, if that
        is the case print an error message, otherwise, build the LDA corpus.
    number_topics : int, default=20
        Number of requested latent topics to be extracted from the training
        corpus.
    number_passes : int, default=20
        Number of passes through the corpus during training.
    """
    print_message(f"Generating the corpus {corpus_kind}/{corpus_name} using the LDA representation with "
                  f"{number_topics} topics.")
    partial_output_path = os.path.join(PATH_PROCESSED_CORPUS, corpus_kind, corpus_name, 'lda')
    output_file_name_train = f'{corpus_name}_lda_corpus_{number_topics:02d}topics_train.pkl'
    output_file_path_train = os.path.join(partial_output_path, output_file_name_train)
    output_lda_model_name = f'{corpus_name}_lda_model_{number_topics:02d}topics.pkl'
    output_lda_model_path = os.path.join(partial_output_path, output_lda_model_name)
    output_id2word_bigram_model_name = f'{corpus_name}_id2word_bigram_model_{number_topics:02d}topics.pkl'
    output_id2word_bigram_model_path = os.path.join(partial_output_path, output_id2word_bigram_model_name)

    # Make the corpus directory if it does not exists.
    os.makedirs(os.path.dirname(output_file_path_train), exist_ok=True)

    continue_processing_this_corpus = True
    lda_model_generated = False
    lda_model, id2word, bigram_model = None, None, None

    if os.path.isfile(output_file_path_train):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name_train} previously created.')
            os.remove(output_file_path_train)
            os.remove(output_id2word_bigram_model_path)
            os.remove(output_lda_model_path)
        else:
            print_message(f'The corpus {output_file_name_train} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
            continue_processing_this_corpus = False

    if continue_processing_this_corpus:
        lda_model, corpus_train, labels_train, id2word, bigram_model = \
            get_lda_model(corpus_name=corpus_name, corpus_kind=corpus_kind,
                          number_topics=number_topics, number_passes=number_passes)

        lda_model_generated = True

        print_message('Saving the LDA model, the word IDs to model dictionary and the bigram model.')

        lda_model.save(output_lda_model_path, pickle_protocol=PICKLE_PROTOCOL)
        with open(output_id2word_bigram_model_path, 'wb') as fp:
            pickle.dump((id2word, bigram_model), fp, protocol=PICKLE_PROTOCOL)

        train_vecs = []
        for i in range(len(corpus_train)):
            top_topics = lda_model.get_document_topics(corpus_train[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(number_topics)]
            train_vecs.append(topic_vec)

        x = np.array(train_vecs, dtype=np.float32)
        y = np.array(labels_train, dtype=np.float32)

        print_message(f'Saving the generated dataset at "{output_file_path_train}".')
        with open(output_file_path_train, 'wb') as fp:
            pickle.dump((x, y), fp, protocol=PICKLE_PROTOCOL)

    # Generate the test corpus.
    output_file_name_test = f'{corpus_name}_lda_corpus_{number_topics:02d}topics_test.pkl'
    output_file_path_test = os.path.join(partial_output_path, output_file_name_test)

    continue_processing_this_corpus = True

    if os.path.isfile(output_file_path_test):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name_test} previously created.')
            os.remove(output_file_path_test)
        else:
            print_message(f'The corpus {output_file_name_test} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
            continue_processing_this_corpus = False

    if continue_processing_this_corpus:
        if not lda_model_generated:
            lda_model, corpus_train, labels_train, id2word, bigram_model = \
                get_lda_model(corpus_name=corpus_name, corpus_kind=corpus_kind,
                              number_topics=number_topics, number_passes=number_passes)

        input_file_path_test = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name,
                                            f'{corpus_name}-test-clean.txt')

        posts_test, labels_test, _ = get_words_list_from_corpus(input_file_path_test)
        corpus_test, _, _ = get_corpus_id2word(posts_test, bigram_model=bigram_model, id2word=id2word)

        test_vecs = []
        for i in range(len(corpus_test)):
            top_topics = lda_model.get_document_topics(corpus_test[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(number_topics)]
            test_vecs.append(topic_vec)

        x = np.array(test_vecs, dtype=np.float32)
        y = np.array(labels_test, dtype=np.float32)

        print_message(f'Saving the generated dataset at "{output_file_path_test}".')
        with open(output_file_path_test, 'wb') as fp:
            pickle.dump((x, y), fp, protocol=PICKLE_PROTOCOL)


def get_lsa_model(corpus_name, corpus_kind, num_factors):
    """Get the Latent Semantic Analysis (LSA) model.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    num_factors : int
        Number of requested factors (latent dimensions).

    Returns
    -------
    lsa_model : gensim.models.LsiModel
        The trained gensim LSA model.
    transformed_corpus : list of list of str
        Transformed corpus.
    labels : list of int
        List containing the users' labels.
    id2word : dict of (int, str)
        The trained mapping from word IDs to words.
    bigram_model : gensim.models.phrases.Phraser
        The trained gensim Phraser model.
    """
    print_message(f"Training the LSA model for {corpus_kind}/{corpus_name} with {num_factors} factors.")
    input_file_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name, f'{corpus_name}-train-clean.txt')

    posts, labels, _ = get_words_list_from_corpus(input_file_path)
    transformed_corpus, id2word, bigram_model = get_corpus_id2word(posts)

    lsa_model = gensim.models.LsiModel(corpus=transformed_corpus, id2word=id2word, num_topics=num_factors)

    print_message(f"Latent topics extracted:")
    for topic_id, topics_words in lsa_model.print_topics(num_topics=-1, num_words=20):
        print_message(f'Topic id: {topic_id} -> {topics_words}')

    return lsa_model, transformed_corpus, labels, id2word, bigram_model


def generate_lsa_corpus(corpus_name, corpus_kind, replace_old=True, num_factors=200):
    """Generate the corpus' Latent Semantic Analysis (LSA) representation.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous LSA corpus exists, if that
        is the case print an error message, otherwise, build the LSA corpus.
    num_factors : int, default=200
        Number of requested factors (latent dimensions).
    """
    print_message(f"Generating the corpus {corpus_kind}/{corpus_name} using the LSA representation with "
                  f"{num_factors} factors.")
    partial_output_path = os.path.join(PATH_PROCESSED_CORPUS, corpus_kind, corpus_name, 'lsa')
    output_file_name_train = f'{corpus_name}_lsa_corpus_{num_factors:03d}factors_train.pkl'
    output_file_path_train = os.path.join(partial_output_path, output_file_name_train)
    output_lsa_model_name = f'{corpus_name}_lsa_model_{num_factors:03d}factors.pkl'
    output_lsa_model_path = os.path.join(partial_output_path, output_lsa_model_name)
    output_id2word_bigram_model_name = f'{corpus_name}_id2word_bigram_model_{num_factors:03d}factors.pkl'
    output_id2word_bigram_model_path = os.path.join(partial_output_path, output_id2word_bigram_model_name)

    # Make the corpus directory if it does not exists.
    os.makedirs(os.path.dirname(output_file_path_train), exist_ok=True)

    continue_processing_this_corpus = True
    lsa_model_generated = False
    lsa_model, id2word, bigram_model = None, None, None

    if os.path.isfile(output_file_path_train):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name_train} previously created.')
            os.remove(output_file_path_train)
            os.remove(output_lsa_model_path)
            os.remove(output_id2word_bigram_model_path)
        else:
            print_message(f'The corpus {output_file_name_train} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
            continue_processing_this_corpus = False

    if continue_processing_this_corpus:
        lsa_model, corpus_train, labels_train, id2word, bigram_model = \
            get_lsa_model(corpus_name=corpus_name, corpus_kind=corpus_kind, num_factors=num_factors)

        lsa_model_generated = True

        print_message('Saving the LSA model, the word IDs to model dictionary and the bigram model.')

        lsa_model.save(output_lsa_model_path, pickle_protocol=PICKLE_PROTOCOL)
        with open(output_id2word_bigram_model_path, 'wb') as fp:
            pickle.dump((id2word, bigram_model), fp, protocol=PICKLE_PROTOCOL)

        x = np.zeros((len(corpus_train), num_factors), dtype=np.float32)
        y = np.array(labels_train, dtype=np.float32)

        vectorized_corpus_train = lsa_model[corpus_train]

        for i, vector in enumerate(vectorized_corpus_train):
            for dim_number, dim_value in vector:
                x[i, dim_number] = dim_value

        print_message(f'Saving the generated dataset at "{output_file_path_train}".')
        with open(output_file_path_train, 'wb') as fp:
            pickle.dump((x, y), fp, protocol=PICKLE_PROTOCOL)

    # Generate the test corpus.
    output_file_name_test = f'{corpus_name}_lsa_corpus_{num_factors:03d}factors_test.pkl'
    output_file_path_test = os.path.join(partial_output_path, output_file_name_test)

    continue_processing_this_corpus = True

    if os.path.isfile(output_file_path_test):
        if replace_old:
            print_message(f'Cleaning the corpus {output_file_name_test} previously created.')
            os.remove(output_file_path_test)
        else:
            print_message(f'The corpus {output_file_name_test} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
            continue_processing_this_corpus = False

    if continue_processing_this_corpus:
        if not lsa_model_generated:
            lsa_model, corpus_train, labels_train, id2word, bigram_model = \
                get_lsa_model(corpus_name=corpus_name, corpus_kind=corpus_kind, num_factors=num_factors)

        input_file_path_test = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name,
                                            f'{corpus_name}-test-clean.txt')

        posts_test, labels_test, _ = get_words_list_from_corpus(input_file_path_test)
        corpus_test, _, _ = get_corpus_id2word(posts_test, bigram_model=bigram_model, id2word=id2word)

        x = np.zeros((len(corpus_test), num_factors), dtype=np.float32)
        y = np.array(labels_test, dtype=np.float32)

        vectorized_corpus_test = lsa_model[corpus_test]

        for i, vector in enumerate(vectorized_corpus_test):
            for dim_number, dim_value in vector:
                x[i, dim_number] = dim_value

        print_message(f'Saving the generated dataset at "{output_file_path_test}".')
        with open(output_file_path_test, 'wb') as fp:
            pickle.dump((x, y), fp, protocol=PICKLE_PROTOCOL)


def generate_doc2vec_corpus(corpus_name, corpus_kind, replace_old=True, training_algorithm=1, v_size=300, w=5):
    """Generate the corpus' doc2vec representation.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous doc2vec corpus exists, if
        that is the case print an error message, otherwise, build the doc2vec
        corpus.
    training_algorithm : {1, 0}, default=1
        Defines the training algorithm. If `training_algorithm=1`,
        distributed memory (PV-DM) is used. Otherwise, distributed bag of words
        (PV-DBOW) is employed.
    v_size : int, default=300
        Dimensionality of the feature vectors.
    w : int, default=5
        The maximum distance between the current and predicted word within a
        sentence.
    """
    training_algorithm_name = 'distributed memory' if training_algorithm else 'distributed bag of words'
    print_message(f"Generating the corpus {corpus_kind}/{corpus_name} using the doc2vec representation with "
                  f"the {training_algorithm_name} algorithm, vector size of {v_size} and window of {w}.")
    params = {
        'dm': training_algorithm,
        'vector_size': v_size,
        'window': w,
        'seed': 30,
        'workers': 4,
        'epochs': 50,
        'min_count': 5,
    }

    partial_output_path = os.path.join(PATH_PROCESSED_CORPUS, corpus_kind, corpus_name, 'doc2vec')
    possible_files = glob.glob(f'{partial_output_path}/{corpus_name}_doc2vec_*.json')
    max_id = 0
    current_id = 0
    already_exists = False
    for file in possible_files:
        current_id = int(file[-7:-5])
        if current_id > max_id:
            max_id = current_id
        already_exists = have_same_parameters(params, file)
        if already_exists:
            if replace_old:
                print_message(f'Cleaning the corpus {file[:-5]} previously created.')
                os.remove(file)
                pickle_file_train = file[:-5] + '_train.pkl'
                pickle_file_test = file[:-5] + '_test.pkl'
                pickle_file_model = file[:-5] + '.model'
                os.remove(pickle_file_train)
                os.remove(pickle_file_test)
                os.remove(pickle_file_model)
            else:
                print_message(f'The corpus {file[:-5]} already exists. Delete it beforehand or '
                              'call this function with the parameter `replace_old=True`.')
                return
            break
    id_number = current_id if already_exists else max_id + 1

    partial_input_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name)
    input_file_path_train = os.path.join(partial_input_path, f'{corpus_name}-train-clean.txt')
    input_file_path_test = os.path.join(partial_input_path, f'{corpus_name}-test-clean.txt')

    output_pkl_train_name = f'{corpus_name}_doc2vec_{id_number:02d}_train.pkl'
    output_pkl_test_name = f'{corpus_name}_doc2vec_{id_number:02d}_test.pkl'
    output_json_name = f'{corpus_name}_doc2vec_{id_number:02d}.json'
    output_pkl_train_path = os.path.join(partial_output_path, output_pkl_train_name)
    output_pkl_test_path = os.path.join(partial_output_path, output_pkl_test_name)
    output_json_path = os.path.join(partial_output_path, output_json_name)

    # Make the corpus directory if it does not exists.
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    print_message(f'Saving the corpus configuration at "{output_json_path}".')
    with open(output_json_path, "w") as f:
        json.dump(fp=f, obj=params, indent='\t')

    posts_train, labels_train, _ = get_words_list_from_corpus(input_file_path_train)

    documents = []
    for i, post in enumerate(posts_train):
        documents.append(gensim.models.doc2vec.TaggedDocument(post, [i]))

    doc2vec_model = gensim.models.doc2vec.Doc2Vec(documents=documents, **params)

    print_message(f'Elapsed time to train the model: {doc2vec_model.total_train_time}.')

    print_message('Saving the doc2vec model.')
    model_file_name = f'{corpus_name}_doc2vec_{id_number:02d}.model'
    model_file_path = os.path.join(partial_output_path, model_file_name)
    doc2vec_model.save(model_file_path)

    # Get the training documents representation.
    x_train = np.zeros((len(posts_train), params['vector_size']), dtype=np.float32)
    y_train = np.array(labels_train, dtype=np.float32)

    for i, post in enumerate(posts_train):
        x_train[i, :] = doc2vec_model.infer_vector(post)

    # Get the testing documents representation.
    posts_test, labels_test, _ = get_words_list_from_corpus(input_file_path_test)
    x_test = np.zeros((len(posts_test), params['vector_size']), dtype=np.float32)
    y_test = np.array(labels_test, dtype=np.float32)

    for i, post in enumerate(posts_test):
        x_test[i, :] = doc2vec_model.infer_vector(post)

    print_message(f'Saving the generated datasets at "{output_pkl_train_path}" and "{output_pkl_test_path}".')
    with open(output_pkl_train_path, 'wb') as fp:
        pickle.dump((x_train, y_train), fp, protocol=PICKLE_PROTOCOL)

    with open(output_pkl_test_path, 'wb') as fp:
        pickle.dump((x_test, y_test), fp, protocol=PICKLE_PROTOCOL)


def pad_text(encoded_documents, sequence_length):
    """Pad the encoded documents.

    If the `len(encoded_document) >= sequence_length` trim the encoded document.
    Otherwise, fill the beginning of the encoded document with zeros (pad
    token).

    Parameters
    ----------
    encoded_documents : list of list of int
        List containing the users' list of encoded words of each posts.
    sequence_length : int
        The maximum sequence length.

    Returns
    -------
    padded_documents : torch.tensor
        Padded encoded documents.
    """
    documents = []
    for document in encoded_documents:
        if len(document) >= sequence_length:
            documents.append(document[:sequence_length])
        else:
            documents.append([0] * (sequence_length - len(document)) + document)
    return torch.tensor(documents, dtype=torch.long)


def generate_padded_sequential_corpus(corpus_name, corpus_kind, replace_old=True, seq_len=200):
    """Generate the corpus' padded sequential representation.

    Each document is represented by the word IDS it contains.
    There are two word IDS with special meaning:
        - Zero (0) represents the padding token.
        - One (1) represents the unknown token, those words that were not found
            in the vocabulary.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    replace_old : bool, default=True
        If `replace_old=True` replace last generated corpus if it exists.
        If `replace_old=False` check if a previous padded sequential corpus
        exists, if that is the case print an error message, otherwise, build the
        padded sequential corpus.
    seq_len : int, default=200
        The maximum sequence length.
    """
    print_message(f"Generating the corpus {corpus_kind}/{corpus_name} using the padded sequential representation with "
                  f"maximum sequence length of {seq_len}.")
    partial_input_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name)
    input_file_path_train = os.path.join(partial_input_path, f'{corpus_name}-train-clean.txt')
    input_file_path_test = os.path.join(partial_input_path, f'{corpus_name}-test-clean.txt')

    partial_output_path = os.path.join(PATH_PROCESSED_CORPUS, corpus_kind, corpus_name, 'padded_sequential')
    output_train_file_name = f'{corpus_name}_padded_sequential_{seq_len:06d}_train.pt'
    output_train_file_path = os.path.join(partial_output_path, output_train_file_name)
    output_test_file_name = f'{corpus_name}_padded_sequential_{seq_len:06d}_test.pt'
    output_test_file_path = os.path.join(partial_output_path, output_test_file_name)

    # Make the corpus directory if it does not exists.
    os.makedirs(os.path.dirname(output_train_file_path), exist_ok=True)

    if os.path.isfile(output_train_file_path):
        if replace_old:
            print_message(f'Cleaning the datasets {output_train_file_name} and {output_test_file_name} previously '
                          'created.')
            os.remove(output_train_file_path)
            os.remove(output_test_file_path)
        else:
            print_message(f'The datasets{output_train_file_name} and {output_test_file_name} already exist. Delete them'
                          ' beforehand or call this function with the parameter `replace_old=True`.')
            return

    all_documents_train, labels_train, all_words_train = get_words_list_from_corpus(input_file_path_train)
    all_documents_test, labels_test, all_words_test = get_words_list_from_corpus(input_file_path_test)

    word_counts = Counter(all_words_train)
    word_list = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: idx + 2 for idx, word in enumerate(word_list)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}

    encoded_documents_train = [[vocab_to_int.get(word, 1) for word in document] for document in all_documents_train]
    encoded_documents_test = [[vocab_to_int.get(word, 1) for word in document] for document in all_documents_test]

    padded_documents_train = pad_text(encoded_documents_train, sequence_length=seq_len)
    padded_documents_test = pad_text(encoded_documents_test, sequence_length=seq_len)

    labels_train = torch.tensor(labels_train, dtype=torch.long)
    labels_test = torch.tensor(labels_test, dtype=torch.long)

    train_data = TensorDataset(labels_train, padded_documents_train)
    test_data = TensorDataset(labels_test, padded_documents_test)

    output_vocabulary_name = f'{corpus_name}_padded_sequential_{seq_len:06d}_vocabulary.pkl'
    output_vocabulary_path = os.path.join(partial_output_path, output_vocabulary_name)
    print_message('Saving the vocabulary (words list, and the words to word IDS and words IDS to words dictionaries).')
    with open(output_vocabulary_path, 'wb') as fp:
        pickle.dump((word_list, vocab_to_int, int_to_vocab), fp, protocol=PICKLE_PROTOCOL)

    print_message(f'Saving the generated datasets at "{output_train_file_path}" and "{output_test_file_path}".')
    torch.save(obj=train_data, f=output_train_file_path, pickle_protocol=PICKLE_PROTOCOL)
    torch.save(obj=test_data, f=output_test_file_path, pickle_protocol=PICKLE_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to build corpora using different document representations.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['depression', 'gambling'])
    parser.add_argument("kind", help="eRisk task corpus kind", choices=['xml', 'reddit'])
    parser.add_argument("replace_old", help="replace old corpus?", type=bool, default=False)
    args = parser.parse_args()

    print_message("-" * 60)
    dm_list = [0, 1]
    vector_size_list = [300]
    window_list = [5, 10, 15]
    for dm in dm_list:
        for vector_size in vector_size_list:
            for window in window_list:
                generate_doc2vec_corpus(corpus_name=args.corpus, corpus_kind=args.kind, replace_old=args.replace_old,
                                        training_algorithm=dm, v_size=vector_size, w=window)

    print_message("-" * 60)
    number_topic_list = [10, 15, 20, 25]
    for number_topic in number_topic_list:
        generate_lda_corpus(corpus_name=args.corpus, corpus_kind=args.kind, replace_old=args.replace_old,
                            number_topics=number_topic, number_passes=20)

    print_message("-" * 60)
    number_factors_list = [10, 50, 100, 200, 400]
    for number_factors in number_factors_list:
        generate_lsa_corpus(corpus_name=args.corpus, corpus_kind=args.kind, replace_old=args.replace_old,
                            num_factors=number_factors)

    print_message("-" * 60)
    seq_length_list = [10_000, 20_000]
    for seq_length in seq_length_list:
        generate_padded_sequential_corpus(corpus_name=args.corpus, corpus_kind=args.kind, replace_old=args.replace_old,
                                          seq_len=seq_length)

    print_message("-" * 60)
    analyzer_param_list = ['word', 'char_wb']
    ngram_range_param_list = [(1, 1), (2, 2), (3, 3), (4, 4)]
    max_df_param_list = [1.0, 0.95]
    min_df_param_list = [0.002, 0.1]
    norm_param_list = ['l2']
    use_idf_param_list = [True]

    for analyzer in analyzer_param_list:
        for ngram_range in ngram_range_param_list:
            for max_df in max_df_param_list:
                for min_df in min_df_param_list:
                    for norm in norm_param_list:
                        for use_idf in use_idf_param_list:
                            countvectorizer_params = {
                                'analyzer': analyzer,
                                'ngram_range': ngram_range,
                                'max_df': max_df,
                                'min_df': min_df,
                            }
                            tfidftransformer_params = {
                                'norm': norm,
                                'use_idf': use_idf,
                            }
                            generate_bow_corpus(corpus_name=args.corpus, corpus_kind=args.kind,
                                                replace_old=args.replace_old,
                                                cv_params=countvectorizer_params,
                                                transformer_params=tfidftransformer_params)

    print_message('#' * 50)
    print_message('END OF SCRIPT')
