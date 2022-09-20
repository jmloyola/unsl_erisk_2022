"""
Models definition.
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


import copy
import glob
import json
import os
import pickle
import random
import shutil
import time
from abc import ABC, abstractmethod

import gensim
import numpy as np
import pyss3
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data import Iterator
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from src.config import (
    BATCH_SIZE,
    END_OF_POST_TOKEN,
    FP_PRECISION_ELAPSED_TIMES,
    MAX_SEQ_LENGTH,
    NUM_POSTS_FOR_BERT_REP,
    PICKLE_PROTOCOL,
)
from src.features.build_features import (
    get_bert_representation,
    get_bow_representation,
    get_doc2vec_representation,
    get_lda_representation,
    get_lsa_representation,
    get_padded_sequential_representation,
)
from src.utils.utilities import print_message


class EmbeddingLSTM(nn.Module):
    """Simple LSTM implementation.

    Parameters
    ----------
    n_vocab : int
        The number of unique words in the vocabulary.
    n_embed : int
        The number of features in the embedding.
    n_hidden : int
        The number of features in the hidden state.
    n_output : int
        The number of outputs.
    n_layers : int
        The number of LSTM layers.
    drop_p : float, default=0
        If non-zero, introduces a Dropout layer on the outputs of each LSTM
        layer except the last layer, with dropout probability equal to `drop_p`.
    """

    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(
            n_embed, n_hidden, n_layers, batch_first=True, dropout=drop_p
        )
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, input_words):
        batch_size = input_words.size(0)
        seq_length = input_words.size(1)
        # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(
            input_words
        )  # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)  # (batch_size, seq_length, n_hidden)
        lstm_out = lstm_out.contiguous().view(
            -1, self.n_hidden
        )  # (batch_size*seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)  # (batch_size*seq_length, n_output)

        fc_out = fc_out.view(batch_size, seq_length, -1)
        fc_out = fc_out[:, -1, :]  # get last batch of labels

        return fc_out, h


class BERT(nn.Module):
    """BERT model.

    Parameters
    ----------
    bert_architecture : {'bert-base-uncased', 'roberta-base'}
        BERT architecture to use.
    num_labels : int
        The number of labels.
    freeze_encoder : bool, default=False
        Flag to indicate if the encoder parameters should be freezed or not.
    resize : int, default=None
        The number of new tokens in the embedding matrix. Increasing the size
        will add newly initialized vectors at the end. Reducing the size will
        remove vectors from the end. If `resize is None` the token embeddings
        are not resized.
    """

    def __init__(
        self, bert_architecture, num_labels, freeze_encoder=False, resize=None
    ):
        super(BERT, self).__init__()
        if bert_architecture == "bert-base-uncased":
            self.encoder = BertForSequenceClassification.from_pretrained(
                bert_architecture, num_labels=num_labels
            )
        elif bert_architecture == "roberta-base":
            self.encoder = RobertaForSequenceClassification.from_pretrained(
                bert_architecture, num_labels=num_labels
            )
        else:
            raise Exception(
                f"The architecture {bert_architecture} is not supported yet."
            )

        if freeze_encoder:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        if resize:
            self.encoder.resize_token_embeddings(resize)

    def forward(self, text):
        logits = self.encoder(text).logits
        return logits


class StopCriterion(ABC):
    """Abstract base class for the decision policy."""

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def decide(self, probabilities, delay, *args, **kwargs):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def clear_state(self):
        pass


class SimpleStopCriterion(StopCriterion):
    """Simple decision policy to determine when to send an alarm.

    Parameters
    ----------
    threshold : float
        The probability threshold to consider a user as risky.
    min_delay : int, default=None
        The minimum delay, that is, the minimum number of posts necessary
        to start considering if a user is at-risk or not.
    max_delay : int, default=None
        The maximum delay, that is, the maximum number of posts that will be
        consider to determine if a user is at-risk or not.
        If the limit is reached, the model will not stop, thus it will not
        classify as positive.
    """

    def __init__(self, threshold, min_delay=None, max_delay=None):
        self.threshold = threshold
        self.min_delay = min_delay
        self.max_delay = max_delay

    def get_parameters(self):
        return {
            "threshold": self.threshold,
            "min_delay": self.min_delay,
            "max_delay": self.max_delay,
        }

    def __repr__(self):
        return f"Threshold = {self.threshold} - Min delay = {self.min_delay} - Max delay = {self.max_delay}"

    def __eq__(self, other):
        are_equal = self.__class__ == other.__class__
        if are_equal:
            are_equal = are_equal and self.threshold == other.threshold
            are_equal = are_equal and self.min_delay == other.min_delay
            are_equal = are_equal and self.max_delay == other.max_delay
        return are_equal

    def clear_state(self):
        pass

    def decide(self, probabilities, delay):
        """Decide to issue an alarm or not for each user.

        Parameters
        ----------
        probabilities : numpy.ndarray
            The probability of belonging to the positive class as
            estimated by the model.
        delay : int
            The current number of post being processed.

        Returns
        -------
        numpy.ndarray
            The decision to issue an alarm or not for every user.
        """
        should_stop_threshold = probabilities >= self.threshold
        should_stop_min_delay = np.ones_like(should_stop_threshold)
        if self.min_delay is not None and delay <= self.min_delay:
            should_stop_min_delay = np.zeros_like(should_stop_min_delay)
        should_stop_max_delay = np.zeros_like(should_stop_threshold)
        if self.max_delay is not None and delay > self.max_delay:
            should_stop_max_delay = np.ones_like(should_stop_max_delay)
        return should_stop_min_delay & (should_stop_threshold | should_stop_max_delay)


class LearnedDecisionTreeStopCriterion(StopCriterion):
    """Learned decision tree decision policy to determine when to send an alarm.

    The features used to train this model were:
    feature_names = [
        "current_probability",
        "avg_last_10_probabilities",
        "avg_last_5_probabilities",
        "median_last_10_probabilities",
        "current_delay",
        "num_words_information_gain_percentile_0_01",
        "num_words_chi2_percentile_0_015",
        "current_cpi_decision",
        "avg_last_10_cpi_decision",
    ]

    Parameters
    ----------
    model_path : str
        Path to the trained DecisionTreeClassifier.
    information_gain_list : list of str
        List of the words from the training corpus with the biggest information
        gain value.
    chi2_list : list of str
        List of the words from the training corpus with the biggest chi2 value.
    last_probabilities : list of list of int, default=None
        List containing a list of the last probabilities for each user.
    last_decisions : list of list of int, default=None
        List containing a list of the last decisions for each user.
    history_length : int, default=10
        The number of steps in the past to store in the history lists.
    """

    def __init__(
        self,
        model_path,
        information_gain_list,
        chi2_list,
        last_probabilities=None,
        last_decisions=None,
        history_length=10,
    ):
        self.model_path = model_path
        with open(self.model_path, "rb") as fp:
            self.clf = pickle.load(fp)
        self.information_gain_list = information_gain_list
        self.chi2_list = chi2_list
        self.last_probabilities = last_probabilities
        self.last_decisions = last_decisions
        self.history_length = history_length

    def get_parameters(self):
        return {
            "model_path": self.model_path,
            "information_gain_list": self.information_gain_list,
            "chi2_list": self.chi2_list,
            "last_probabilities": self.last_probabilities,
            "last_decisions": self.last_decisions,
            "history_length": self.history_length,
        }

    def __repr__(self):
        return (
            f"DecisionTreeClassifier: {self.clf}\ninformation_gain_list: {self.information_gain_list}\n"
            f"chi2_list: {self.chi2_list}"
        )

    def __eq__(self, other):
        are_equal = self.__class__ == other.__class__
        if are_equal:
            are_equal = are_equal and self.model_path == other.model_path
            are_equal = (
                are_equal and self.information_gain_list == other.information_gain_list
            )
            are_equal = are_equal and self.chi2_list == other.chi2_list
        return are_equal

    def clear_state(self):
        self.last_probabilities = None
        self.last_decisions = None

    def get_representation(
        self, probabilities, decisions, raw_posts, idx_non_stopped_doc, delay
    ):
        """Get the representation used by the DecisionTreeClassifier to decide.

        The list of features are:
        feature_names = [
            "current_probability",
            "avg_last_10_probabilities",
            "avg_last_5_probabilities",
            "median_last_10_probabilities",
            "current_delay",
            "num_words_information_gain_percentile_0_01",
            "num_words_chi2_percentile_0_015",
            "current_cpi_decision",
            "avg_last_10_cpi_decision",
        ]

        Parameters
        ----------
        probabilities : numpy.ndarray
            The probability of belonging to the positive class as estimated by
            the model.
        decisions : numpy.ndarray
            The predicted class as estimated by the model.
        raw_posts : list of str
            List of users' posts. These are only the posts of the users that
            have more posts to process. Note that each users has all her posts
            concatenated using the string `config.END_OF_POST_TOKEN`.
        idx_non_stopped_doc : numpy.ndarray
            Indices of the users that are still being processed.
        delay : int
            The current number of post being processed.

        Returns
        -------
        numpy.ndarray
            Representation used by the DecisionTreeClassifier.
        """
        current_features = []
        if self.last_probabilities is None:
            # If this is the first time the DMC is called, initialize the history lists.
            self.last_probabilities = [[] for _ in idx_non_stopped_doc]
            self.last_decisions = [[] for _ in idx_non_stopped_doc]
        for i, idx in enumerate(idx_non_stopped_doc):
            if len(self.last_probabilities[idx]) < self.history_length:
                self.last_probabilities[idx].append(probabilities[i].item())
                self.last_decisions[idx].append(decisions[i].item())
            else:
                self.last_probabilities[idx] = self.last_probabilities[idx][1:] + [
                    probabilities[i].item()
                ]
                self.last_decisions[idx] = self.last_decisions[idx][1:] + [
                    decisions[i].item()
                ]
            raw_current_post = " ".join(raw_posts[i].split(END_OF_POST_TOKEN))
            user_features = [
                probabilities[i].item(),
                np.average(self.last_probabilities[idx]).item(),
                np.average(self.last_probabilities[idx][-5:]).item(),
                np.median(self.last_probabilities[idx]).item(),
                delay,
                sum(
                    [
                        1 if w in self.information_gain_list else 0
                        for w in raw_current_post.split()
                    ]
                ),
                sum(
                    [1 if w in self.chi2_list else 0 for w in raw_current_post.split()]
                ),
                decisions[i].item(),
                np.average(self.last_decisions[idx]).item(),
            ]
            current_features.append(user_features)
        return np.asarray(current_features)

    def decide(self, probabilities, decisions, raw_posts, idx_non_stopped_doc, delay):
        """Decide to issue an alarm or not for each user.

        Parameters
        ----------
        probabilities : numpy.ndarray
            The probability of belonging to the positive class as
            estimated by the model.
        decisions : numpy.ndarray
            The predicted class as estimated by the model.
        raw_posts : list of str
            List of users' posts. These are only the posts of the users that
            have more posts to process.
        idx_non_stopped_doc : numpy.ndarray
            Indices of the users that are still being processed.
        delay : int
            The current number of post being processed.

        Returns
        -------
        should_stop : numpy.ndarray
            The decision to issue an alarm or not for every user.
        """
        features = self.get_representation(
            probabilities=probabilities,
            decisions=decisions,
            raw_posts=raw_posts,
            idx_non_stopped_doc=idx_non_stopped_doc,
            delay=delay,
        )
        should_stop = self.clf.predict(features)
        return should_stop


class HistoricStopCriterion(StopCriterion):
    """Historic decision policy to determine when to send an alarm.

    This is similar to the class `SimpleStopCriterion` where we also consider
    the historic probabilities of every user up to a certain point. If the
    previous 10 probabilities and the last probability of the positive class
    surpass the threshold a decision is made.

    Parameters
    ----------
    threshold : float
        The probability threshold to consider a user as risky.
    min_delay : int, default=None
        The minimum delay, that is, the minimum number of posts necessary
        to start considering if a user is at-risk or not.
    last_probabilities : list of list of int, default=None
        List containing a list of the last probabilities for each user.
    history_length : int, default=10
        The number of steps in the past to store in the history lists.
    """

    def __init__(
        self, threshold, min_delay=None, last_probabilities=None, history_length=10
    ):
        self.threshold = threshold
        self.min_delay = min_delay
        self.last_probabilities = last_probabilities
        self.history_length = history_length

    def get_parameters(self):
        return {
            "threshold": self.threshold,
            "min_delay": self.min_delay,
            "last_probabilities": self.last_probabilities,
            "history_length": self.history_length,
        }

    def __repr__(self):
        return f"Threshold = {self.threshold} - Min delay = {self.min_delay} - History length = {self.history_length}"

    def __eq__(self, other):
        are_equal = self.__class__ == other.__class__
        if are_equal:
            are_equal = are_equal and self.threshold == other.threshold
            are_equal = are_equal and self.min_delay == other.min_delay
            are_equal = are_equal and self.history_length == other.history_length
        return are_equal

    def clear_state(self):
        self.last_probabilities = None

    def decide(self, probabilities, idx_non_stopped_doc, delay):
        """Decide to issue an alarm or not for each user.

        Parameters
        ----------
        probabilities : numpy.ndarray
            The probability of belonging to the positive class as
            estimated by the model.
        idx_non_stopped_doc : numpy.ndarray
            Indices of the users that are still being processed.
        delay : int
            The current number of post being processed.

        Returns
        -------
        should_stop : numpy.ndarray
            The decision to issue an alarm or not for every user.
        """
        should_stop_threshold = probabilities >= self.threshold
        should_stop_min_delay = np.ones_like(should_stop_threshold)
        if self.min_delay is not None and delay <= self.min_delay:
            should_stop_min_delay = np.zeros_like(should_stop_min_delay)

        if self.last_probabilities is None:
            # If this is the first time the DMC is called, initialize the history lists.
            self.last_probabilities = [[] for _ in idx_non_stopped_doc]

        for i, idx in enumerate(idx_non_stopped_doc):
            # Before updating the probability history, count the number of time steps
            # the threshold was surpassed.
            num_steps_over_threshold = sum(
                [
                    1 if prob >= self.threshold else 0
                    for prob in self.last_probabilities[idx]
                ]
            )
            should_stop_threshold[i] = should_stop_threshold[i] and (
                num_steps_over_threshold == NUM_POSTS_FOR_BERT_REP
            )

            if len(self.last_probabilities[idx]) < self.history_length:
                self.last_probabilities[idx].append(probabilities[i].item())
            else:
                self.last_probabilities[idx] = self.last_probabilities[idx][1:] + [
                    probabilities[i].item()
                ]

        return should_stop_min_delay & should_stop_threshold


def nn_predict_probability(classifier, loader, device):
    """Predict the label and probability of the input.

    Parameters
    ----------
    classifier : {EmbeddingLSTM, BERT, EARLIEST}
        Model used to predict new input.
    loader : torch.utils.data.DataLoader
        Data loader. Combines a dataset and a sampler, and provides an iterable
        over the given dataset.
    device : torch.device
        Selected device where the model and input will reside.
    """
    all_predictions = []
    all_probabilities = []
    all_halting_points = []
    classifier.eval()
    with torch.no_grad():
        for batch in loader:
            if type(classifier) == EmbeddingLSTM:
                inputs = batch[0]
                inputs = inputs.to(device)
                logits, h = classifier(inputs)
                logits = logits.squeeze()
                # round predictions to the closest integer.
                probabilities = torch.sigmoid(logits)
                predictions = torch.round(probabilities)
                all_predictions.append(predictions.view(-1))
                all_probabilities.append(probabilities.view(-1))
            elif type(classifier) == BERT:
                inputs = batch.posts
                inputs = inputs.to(device)
                logits = classifier(inputs)
                predictions = torch.argmax(logits, dim=1)
                probabilities = nn.functional.softmax(logits, dim=-1)
                # Only get the probability of the positive class.
                probabilities = probabilities[:, 1]
                all_predictions.append(predictions.view(-1))
                all_probabilities.append(probabilities.view(-1))
            elif type(classifier) == EARLIEST:
                inputs = batch[0]
                inputs = inputs.to(device)
                # Put the current time in the first position.
                inputs = torch.transpose(inputs, 0, 1)

                logits, halting_points, halting_points_mean = classifier(
                    inputs, test=True
                )

                if classifier.n_classes > 1:
                    predictions = torch.argmax(logits, dim=1)
                    probabilities = nn.functional.softmax(logits, dim=-1)
                    # Only get the probability of the positive class.
                    probabilities = probabilities[:, 1]
                else:
                    probabilities = torch.sigmoid(logits)
                    predictions = torch.round(probabilities).int()

                all_predictions.append(predictions.view(-1))
                all_probabilities.append(probabilities.view(-1))
                all_halting_points.append(halting_points.view(-1))
            else:
                raise Exception(
                    f'Model "{type(classifier)}" not implemented in nn_predict_probability().'
                )
    if type(classifier) == EARLIEST:
        return (
            torch.cat(all_predictions),
            torch.cat(all_probabilities),
            torch.cat(all_halting_points),
        )
    else:
        return torch.cat(all_predictions), torch.cat(all_probabilities)


class CompetitionModel(ABC):
    """Abstract base class for the competition models."""

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load(path, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, documents_test, delay):
        pass

    @abstractmethod
    def clear_model_state(self):
        pass


class EarlyModel(CompetitionModel):
    """Simple model for early classification.

    This model is composed of two parts:
        - a classifier able to categorize partial documents;
        - a rule capable of determining if an alarm should be raised.

    Parameters
    ----------
    path_to_model_information : str
        Path for the file with the information of the model to load.
    stop_criterion : StopCriterion
        The decision policy to determine when to send an alarm.
    is_competition : bool, default=True
        Boolean flag to indicate if the model is used in the eRisk competition.
        If `is_competition is True` we keep processing the input even when the
        user was classified as positive to continue to report the score for each
        user.
    """

    def __init__(self, path_to_model_information, stop_criterion, is_competition=True):
        if not path_to_model_information.endswith(".json"):
            raise Exception(
                'To initialize the model, the `path_to_model_information` should end with ".json".'
            )

        prefix_model = "01_model_"
        base_path = os.path.dirname(path_to_model_information)
        file_name = os.path.basename(path_to_model_information)
        self.model_id = file_name[:2]
        self.classifier_type = file_name[len(prefix_model) : -len(".json")]
        self.path_to_model_information = path_to_model_information
        with open(path_to_model_information, "r") as fp:
            self.model_information = json.load(fp=fp)

        if self.classifier_type == "BERT" or self.classifier_type == "EmbeddingLSTM":
            self.path_to_model_parameters = (
                path_to_model_information[: -len("json")] + "pt"
            )
            self.classifier = None
        else:
            self.path_to_model_parameters = (
                path_to_model_information[: -len("json")] + "pkl"
            )
            with open(self.path_to_model_parameters, "rb") as fp:
                self.classifier = pickle.load(fp)

        representation_file_name = file_name[:2] + "_representation_*.json"
        self.path_to_representation_information = glob.glob(
            f"{base_path}/{representation_file_name}"
        )[0]
        with open(self.path_to_representation_information, "r") as fp:
            self.representation_information = json.load(fp=fp)
        prefix_representation = "01_representation_"
        self.representation_name = os.path.basename(
            self.path_to_representation_information
        )[len(prefix_representation) : -len(".json")]
        self.representation = None
        self.path_to_representation_parameters = (
            self.path_to_representation_information[: -len("json")] + "pkl"
        )
        self.id2word, self.bigram_model = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stop_criterion = stop_criterion
        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None
        self.is_competition = is_competition
        self.num_post_processed = 0

    def __repr__(self):
        model_information = "-" * 50 + "\n"
        model_information += "EarlyModel model\n"
        model_information += f"Type of representation: {self.representation_name}\n"
        model_information += (
            f"Representation parameters: {self.representation_information}\n"
        )
        model_information += f"Type of classifier: {self.classifier_type}\n"
        model_information += f"Classifier parameters: {self.model_information}\n"
        model_information += f"Stop criterion: {self.stop_criterion}\n"
        model_information += "-" * 50 + "\n"
        return model_information

    def __eq__(self, other):
        are_equal = self.path_to_model_information == other.path_to_model_information
        are_equal = are_equal and self.representation_name == other.representation_name
        are_equal = (
            are_equal
            and self.representation_information == other.representation_information
        )
        are_equal = are_equal and self.classifier_type == other.classifier_type
        are_equal = are_equal and self.model_information == other.model_information
        are_equal = are_equal and self.stop_criterion == other.stop_criterion
        are_equal = are_equal and self.is_competition == other.is_competition
        return are_equal

    def get_representation(self, documents):
        """Get representation of the documents.

        Parameters
        ----------
        documents : list of str
            Raw documents to get the representation of.

        Returns
        -------
        documents_representation : numpy.ndarray
            Representation of the documents.
        """
        if self.representation_name == "bow":
            if self.representation is None:
                with open(self.path_to_representation_parameters, "rb") as fp:
                    self.representation = pickle.load(fp)
            count_vect, tfidf_transformer = self.representation

            return get_bow_representation(
                documents=documents,
                count_vect=count_vect,
                tfidf_transformer=tfidf_transformer,
            )

        elif self.representation_name == "lda" or self.representation_name == "lsa":
            if self.representation is None:
                if self.representation_name == "lda":
                    self.representation = gensim.models.LdaModel.load(
                        self.path_to_representation_parameters
                    )
                else:
                    self.representation = gensim.models.LsiModel.load(
                        self.path_to_representation_parameters
                    )
                id2word_bigram_path = self.path_to_representation_information.replace(
                    ".json", "_id2word_bigram_model.pkl"
                )
                with open(id2word_bigram_path, "rb") as fp:
                    self.id2word, self.bigram_model = pickle.load(fp)

            if self.representation_name == "lda":
                return get_lda_representation(
                    documents=documents,
                    lda_model=self.representation,
                    id2word=self.id2word,
                    bigram_model=self.bigram_model,
                )
            else:
                return get_lsa_representation(
                    documents=documents,
                    lsa_model=self.representation,
                    id2word=self.id2word,
                    bigram_model=self.bigram_model,
                )

        elif self.representation_name == "doc2vec":
            if self.representation is None:
                self.representation = gensim.models.doc2vec.Doc2Vec.load(
                    self.path_to_representation_parameters
                )
            return get_doc2vec_representation(
                documents=documents, doc2vec_model=self.representation, sequential=False
            )

        elif self.representation_name == "bert_tokenizer":
            if self.representation is None:
                bert_architecture = self.model_information["model_params"].get(
                    "bert_architecture", "bert-base-uncased"
                )
                if bert_architecture == "bert-base-uncased":
                    self.representation = BertTokenizer.from_pretrained(
                        bert_architecture
                    )
                elif bert_architecture == "roberta-base":
                    self.representation = RobertaTokenizer.from_pretrained(
                        bert_architecture
                    )
                else:
                    raise Exception(
                        f'The architecture "{bert_architecture}" is not supported yet.'
                    )
            return get_bert_representation(
                documents=documents, tokenizer=self.representation
            )

        elif self.representation_name == "padded_sequential":
            if self.representation is None:
                with open(self.path_to_representation_parameters, "rb") as fp:
                    # This file contains the tuple: `(word_list, vocab_to_int, int_to_vocab)`.
                    # Here, only the `vocab_to_int` is used.
                    _, self.representation, _ = pickle.load(fp)
            return get_padded_sequential_representation(
                documents=documents,
                vocab_to_int=self.representation,
                seq_len=self.representation_information["seq_length"],
            )

        else:
            raise Exception(
                f'Representation "{self.representation_name}" not implemented yet.'
            )

    def predict(self, documents_test, delay):
        """Predict the class for the current users' posts.

        Parameters
        ----------
        documents_test : list of str
            List of users' posts.
        delay : int
            Current delay, i.e., post number being processed.

        Returns
        -------
        len_active_users : int
            The number of documents with more posts to process.
        feature_elapsed_time : float
            Time elapsed building the features for the input.
        prediction_elapsed_time : float
            Time elapsed while predicting the input.
        """
        feature_elapsed_time = 0
        prediction_elapsed_time = 0
        start_time = time.time()
        # The first time this function is called, initialize the attributes of
        # the class.
        if self.predictions is None:
            self.predictions = np.array([-1] * len(documents_test))
            self.probabilities = -np.ones_like(self.predictions, dtype=float)
            self.delays = -np.ones_like(self.predictions)
            self.already_finished = np.zeros_like(self.predictions)

        # For the users with no more posts and for which a decision has not been
        # made, issue the last label and store the delay.
        cant_posts_docs = [len(doc.split(END_OF_POST_TOKEN)) for doc in documents_test]
        for j, num_posts in enumerate(cant_posts_docs):
            if num_posts < delay:
                self.already_finished[j] = 1
                if self.delays[j] == -1:
                    self.delays[j] = delay - 1

        if self.is_competition:
            # Keep reporting the scores of already flag users for the laboratory eRisk.
            idx_non_stopped_doc = [
                j
                for j, has_finished in enumerate(self.already_finished)
                if not has_finished
            ]
        else:
            idx_non_stopped_doc = [j for j, d in enumerate(self.delays) if d == -1]

        if len(idx_non_stopped_doc) > 0:
            documents_not_finished = [documents_test[j] for j in idx_non_stopped_doc]

            print_message(
                f"Current post number: {delay} - Number of documents not "
                f"finished: {len(documents_not_finished)}"
            )

            x_test = self.get_representation(documents_not_finished)
            feature_elapsed_time = time.time() - start_time
            feature_elapsed_time = round(
                feature_elapsed_time, FP_PRECISION_ELAPSED_TIMES
            )

            start_time_pred = time.time()
            if (
                self.classifier_type != "BERT"
                and self.classifier_type != "EmbeddingLSTM"
            ):
                y_predicted = self.classifier.predict(x_test)
                # Only get the probability of the positive class.
                probabilities = self.classifier.predict_proba(x_test)[:, 1]
            else:
                if self.classifier_type == "EmbeddingLSTM":
                    if self.classifier is None:
                        n_vocab = len(self.representation) + 2
                        n_embed = 200
                        n_hidden = 256
                        n_output = 1  # 1 ("positive") or 0 ("negative")
                        n_layers = self.representation_information.get("n_layers", 1)
                        self.classifier = EmbeddingLSTM(
                            n_vocab, n_embed, n_hidden, n_output, n_layers
                        )
                        # `map_location=self.device` maps the loaded tensor to the correct device.
                        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
                        self.classifier.load_state_dict(
                            torch.load(
                                self.path_to_model_parameters, map_location=self.device
                            )
                        )
                        self.classifier.to(self.device)

                    test_data = DataLoader(x_test, batch_size=5, shuffle=False)

                    y_predicted, probabilities = nn_predict_probability(
                        classifier=self.classifier, loader=test_data, device=self.device
                    )
                else:
                    # self.classifier_type == 'BERT'
                    if self.classifier is None:
                        bert_architecture = self.model_information["model_params"].get(
                            "bert_architecture", "bert-base-uncased"
                        )
                        freeze_encoder = self.model_information["model_params"].get(
                            "freeze_encoder", False
                        )

                        add_tokens = self.model_information["model_params"].get(
                            "add_tokens", False
                        )
                        resize_token_embedding = None
                        if add_tokens:
                            # Get the number of added tokens.
                            tokens_added_list = self.model_information[
                                "model_params"
                            ].get("tokens_added", None)
                            size_vocab = len(self.representation)
                            resize_token_embedding = size_vocab + len(tokens_added_list)

                        self.classifier = BERT(
                            bert_architecture=bert_architecture,
                            num_labels=2,
                            freeze_encoder=freeze_encoder,
                            resize=resize_token_embedding,
                        )
                        self.classifier.load_state_dict(
                            torch.load(
                                self.path_to_model_parameters, map_location=self.device
                            )
                        )
                        self.classifier.to(self.device)
                    test_data = Iterator(
                        x_test,
                        batch_size=10,
                        device=self.device,
                        train=False,
                        shuffle=False,
                        sort=False,
                    )
                    y_predicted, probabilities = nn_predict_probability(
                        classifier=self.classifier, loader=test_data, device=self.device
                    )

            if self.classifier_type in ["EmbeddingLSTM", "BERT"]:
                y_predicted = y_predicted.cpu().numpy()
                probabilities = probabilities.cpu().numpy()
            self.predictions[idx_non_stopped_doc] = y_predicted
            self.probabilities[idx_non_stopped_doc] = probabilities
            if type(self.stop_criterion) == SimpleStopCriterion:
                stop_reading = self.stop_criterion.decide(
                    probabilities=probabilities, delay=delay
                )
            elif type(self.stop_criterion) == LearnedDecisionTreeStopCriterion:
                stop_reading = self.stop_criterion.decide(
                    probabilities=probabilities,
                    decisions=y_predicted,
                    raw_posts=documents_not_finished,
                    idx_non_stopped_doc=idx_non_stopped_doc,
                    delay=delay,
                )
            elif type(self.stop_criterion) == HistoricStopCriterion:
                stop_reading = self.stop_criterion.decide(
                    probabilities=probabilities,
                    idx_non_stopped_doc=idx_non_stopped_doc,
                    delay=delay,
                )
            else:
                raise Exception(
                    f'The criterion class "{type(self.stop_criterion)}" is not implemented yet.'
                )
            for j, idx in enumerate(idx_non_stopped_doc):
                # The second condition is to consider the case that `self.is_competition is True`
                if stop_reading[j] and self.delays[idx] == -1:
                    self.delays[idx] = delay
            self.num_post_processed += 1
            prediction_elapsed_time = time.time() - start_time_pred
            prediction_elapsed_time = round(
                prediction_elapsed_time, FP_PRECISION_ELAPSED_TIMES
            )
        return len(idx_non_stopped_doc), feature_elapsed_time, prediction_elapsed_time

    def clear_model_state(self):
        """Clear the internal state of the model.

        Use this function if loading a pre-trained EarlyModel model for the
        first time.
        """
        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None
        self.num_post_processed = 0
        self.stop_criterion.clear_state()

    def save(self, path_json):
        """Save the information and state of the EarlyModel.

        Parameters
        ----------
        path_json : str
            Path to save the information and state of the model.
        """
        # When deploying the model there are a number of attributes that are not set.
        if self.delays is None:
            self.delays = np.empty(0)
            self.already_finished = np.empty(0)
            self.predictions = np.empty(0)
            self.probabilities = np.empty(0)
        model_information = {
            "path_to_model_information": self.path_to_model_information,
            "criterion_class": self.stop_criterion.__class__.__name__,
            "criterion_params": self.stop_criterion.get_parameters(),
            "is_competition": self.is_competition,
            "num_post_processed": self.num_post_processed,
            "delays": self.delays.tolist(),
            "already_finished": self.already_finished.tolist(),
            "predictions": self.predictions.tolist(),
            "probabilities": self.probabilities.tolist(),
            "representation_information": self.representation_information,
            "model_information": self.model_information,
        }
        with open(path_json, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent="\t")

    @staticmethod
    def load(path_json):
        """Load EarlyModel model.

        Parameters
        ----------
        path_json : str
            Path to the file containing the state of the EarlyModel.

        Returns
        --------
        early_model : EarlyModel
            The loaded EarlyModel model.
        """
        with open(path_json, "r") as fp:
            model_information = json.load(fp=fp)
        path_to_model_information = model_information["path_to_model_information"]
        criterion_class = model_information["criterion_class"]
        criterion_params = model_information["criterion_params"]
        if criterion_class == "SimpleStopCriterion":
            stop_criterion = SimpleStopCriterion(**criterion_params)
        elif criterion_class == "LearnedDecisionTreeStopCriterion":
            stop_criterion = LearnedDecisionTreeStopCriterion(**criterion_params)
        elif criterion_class == "HistoricStopCriterion":
            stop_criterion = HistoricStopCriterion(**criterion_params)
        else:
            raise Exception(
                f'Loader for criterion class "{criterion_class}" not implemented yet.'
            )
        is_competition = model_information["is_competition"]
        early_model = EarlyModel(
            path_to_model_information=path_to_model_information,
            stop_criterion=stop_criterion,
            is_competition=is_competition,
        )
        early_model.num_post_processed = model_information["num_post_processed"]
        early_model.delays = np.array(model_information["delays"])
        early_model.already_finished = np.array(model_information["already_finished"])
        early_model.predictions = np.array(model_information["predictions"])
        early_model.probabilities = np.array(model_information["probabilities"])

        return early_model

    def deploy(self, deploy_path, x_train, y_train, x_test=None, y_test=None):
        """Deploy model for usage in the eRisk laboratory.

        The deployment of the model involves:
            - Re-training base model using all the available datasets (reddit and xml files).
            - Copying models' component to the deploy_path.
            - Generating a new `model_information.json` file.

        Parameters
        ----------
        deploy_path : str
            Path where the model will be deployed.

        x_train : list of str
            List of users' publications. Each user publication is separated
            using the token `config.END_OF_POST_TOKEN`.

        y_train : list of int
            List of users' labels.

        x_test : None
            Parameter not used. Only present for API consistency.

        y_test : None
            Parameter not used. Only present for API consistency.
        """
        if self.classifier_type == "BERT" or self.classifier_type == "EmbeddingLSTM":
            raise Exception(
                f"Deploy method not implemented for {self.classifier_type} yet."
            )

        x_train_processed = self.get_representation(x_train)
        print_message(f"Final processed corpus shape: {x_train_processed.shape}")
        self.classifier.fit(x_train_processed, y_train)

        # Copy original model_information to deploy path.
        new_path_to_model_information = os.path.join(
            deploy_path, os.path.basename(self.path_to_model_information)
        )
        print_message(
            f"Copying {self.path_to_model_information} to {new_path_to_model_information}."
        )
        shutil.copy2(self.path_to_model_information, new_path_to_model_information)
        self.path_to_model_information = new_path_to_model_information

        # Save retrained model.
        self.path_to_model_parameters = (
            self.path_to_model_information[: -len("json")] + "pkl"
        )
        print_message(f"Saving retrained model to {self.path_to_model_parameters}.")
        with open(self.path_to_model_parameters, "wb") as f:
            pickle.dump(self.classifier, f, protocol=PICKLE_PROTOCOL)

        # Copy representation files.
        print_message("Copying representation files.")
        representation_files_glob = (
            self.path_to_representation_information[: -len(".json")] + "*"
        )
        representation_files = glob.glob(representation_files_glob)
        for rep_file in representation_files:
            file_name = os.path.basename(rep_file)

            new_path_rep_file = os.path.join(deploy_path, file_name)
            print_message(f"Copying {rep_file} to {new_path_rep_file}.")
            shutil.copy2(rep_file, new_path_rep_file)

        # Save EarlyModel.
        final_model_path = os.path.join(deploy_path, "deployed_earlymodel.json")
        self.save(final_model_path)

        # Generate model_information.json file.
        model_information_path = os.path.join(deploy_path, "model_information.json")
        model_information = {
            "model_class": "EarlyModel",
            "model_path": final_model_path,
        }
        with open(model_information_path, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent="\t")


class BaselineNetwork(nn.Module):
    """Baseline network.

    A network which predicts the average reward observed during a markov
    decision-making process. Weights are updated with respect to the mean
    squared error between its prediction and the observed reward.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    output_size : int
        Size of the output layer.
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.relu(self.fc(x.detach()))
        return b


class Controller(nn.Module):
    """Controller network.

    A network that chooses whether or not enough information
    has been seen to predict a label.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    output_size : int
        Size of the output layer.
    """

    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()

        self.fc = nn.Linear(input_size, output_size)  # Optimized w.r.t. the reward.
        # The buffers have the advantage that they reside in the same device as
        # the model and have `require_grad = False`.
        self.register_buffer("_epsilon", torch.randn(1))
        self.register_buffer("_small_exploration", torch.tensor([0.05]))
        self.register_buffer("_zero_constant", torch.zeros(1))
        self.register_buffer("_small_value", torch.tensor([0.000001]))

    def forward(self, x):
        stopping_probability = torch.sigmoid(self.fc(x))
        stopping_probability = stopping_probability.squeeze()
        # Explore/exploit depending on the value of epsilon.
        stopping_probability = (
            1 - self._epsilon
        ) * stopping_probability + self._epsilon * self._small_exploration
        # Sum a small amount if zero, which causes the log to return inf.
        stopping_probability = torch.where(
            torch.isclose(stopping_probability, self._zero_constant),
            stopping_probability + self._small_value,
            stopping_probability,
        )

        m = Bernoulli(probs=stopping_probability)
        # Sample an action.
        action = m.sample()
        # Compute log probability of sampled action.
        log_pi = m.log_prob(action)
        return action, log_pi, -torch.log(stopping_probability)


class EARLIEST(nn.Module, CompetitionModel):
    """Early and Adaptive Recurrent Label ESTimator (EARLIEST).

    Code adapted from https://github.com/Thartvigsen/EARLIEST
    to work with text data.

    Parameters
    ----------
    n_inputs : int
        The number of features in the input data.
    n_classes : int, default=1
        The number of classes in the input labels.
    n_hidden : int, default=50
        The number of dimensions in the RNN's hidden states.
    n_layers : int, default=1
        The number of layers in the RNN.
    lam : float, default=0.0001
        Earliness weight, i.e., emphasis on earliness.
    num_epochs : int, default=600
        The number of epochs use during training.
    drop_p : float, default=0
        Dropout probability. If non-zero, introduces a dropout layer on
        the outputs of each LSTM layer except the last layer.
    weights : torch.Tensor, default=None
        Weights for each class.
    input_type : {'doc2vec'}, default='doc2vec'
        The type of the input.
    device : str, default='cpu'
        Device to run the model from.
    representation_path : str, default=None
        Path to the trained representation.
    max_sequence_length : int, default=200
        Maximum number of posts to represent per user.
    is_competition : bool, default=True
        Flag to indicate if run in the challenge.
    """

    def __init__(
        self,
        n_inputs=1,
        n_classes=1,
        n_hidden=50,
        n_layers=1,
        lam=0.0001,
        num_epochs=600,
        drop_p=0,
        weights=None,
        input_type="doc2vec",
        device="cpu",
        representation_path=None,
        max_sequence_length=200,
        is_competition=True,
    ):
        super(EARLIEST, self).__init__()

        # Hyper-parameters.
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.drop_p = drop_p
        self.input_type = input_type
        self.device = device

        self.is_competition = is_competition

        self.representation_path = representation_path
        self.representation = None
        self.last_docs_rep = None
        self.last_num_posts_processed = None
        self.num_post_processed = 0
        self.max_sequence_length = max_sequence_length
        self.last_idx_non_stopped_doc = []

        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None

        # The buffers have the advantage that they reside in the same device as
        # the model, and have `require_grad = False`.
        self.register_buffer("lam", torch.tensor([lam]))
        # The time buffer will, later on, change shape depending on the batch.
        # Using buffer makes it reside in the same device as the model.
        self.register_buffer("time", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("_exponentials", self.exponential_decay(num_epochs))
        self.register_buffer("weights", weights)

        # Sub-networks.
        self.Controller = Controller(n_hidden + 1, 1)
        self.BaselineNetwork = BaselineNetwork(n_hidden + 1, 1)
        if input_type == "doc2vec":
            self.RNN = nn.LSTM(
                input_size=n_inputs,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=drop_p,
            )
        else:
            raise Exception(
                f'The input type "{self.input_type}" not allowed. Use "doc2vec".'
            )
        self.out = nn.Linear(n_hidden, n_classes)

    def __repr__(self):
        model_information = "-" * 50 + "\n"
        model_information += "EARLIEST model\n"
        model_information += f"n_inputs: {self.n_inputs}\n"
        model_information += f"n_classes: {self.n_classes}\n"
        model_information += f"n_hidden: {self.n_hidden}\n"
        model_information += f"n_layers: {self.n_layers}\n"
        model_information += f"drop_p: {self.drop_p}\n"
        model_information += f"input_type: {self.input_type}\n"
        model_information += f"device: {self.device}\n"
        model_information += f"representation_path: {self.representation_path}\n"
        model_information += f"max_sequence_length: {self.max_sequence_length}\n"
        model_information += f"is_competition: {self.is_competition}\n"
        model_information += "-" * 50 + "\n"
        return model_information

    @staticmethod
    def exponential_decay(n):
        """Calculate samples from the exponential decay.

        Parameters
        ----------
        n : int
            The number of samples to take.

        Returns
        -------
        y : torch.Tensor
            Tensor with `n` samples of the exponential decay.
        """
        tau = 1
        tmax = 7
        t = np.linspace(0, tmax, n)
        y = torch.tensor(np.exp(-t / tau), dtype=torch.float)
        return y

    def save(self, path_json):
        """Save the model's state and learnable parameters to disk.

        Parameters
        ----------
        path_json : str
            Path to save the state of the model.

        Notes
        -----
        The learnable parameters are stored in a file with the same name
        as `path_json` but with extension `.pt`.
        """
        filename, file_extension = os.path.splitext(path_json)
        state_dict_path = filename + ".pt"
        model_params = {
            "n_inputs": self.n_inputs,
            "n_classes": self.n_classes,
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "drop_p": self.drop_p,
            "input_type": self.input_type,
            "device": self.device,
            "representation_path": self.representation_path,
            "max_sequence_length": self.max_sequence_length,
            "is_competition": self.is_competition,
        }
        model_information = {
            "model_params": model_params,
            "state_dict_path": state_dict_path,
        }
        if self.is_competition:
            model_information.update(
                {
                    "last_docs_rep": self.last_docs_rep.tolist(),
                    "last_num_posts_processed": self.last_num_posts_processed,
                    "num_post_processed": self.num_post_processed,
                    "last_idx_non_stopped_doc": self.last_idx_non_stopped_doc,
                    "predictions": self.predictions.tolist(),
                    "probabilities": self.probabilities.tolist(),
                    "delays": self.delays.tolist(),
                    "already_finished": self.already_finished.tolist(),
                }
            )
        with open(path_json, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent="\t")

        if not self.is_competition:
            # Since the PyTorch version used hasn't implemented the `persistent=False`
            # parameter for the `register_buffer` function, before saving the model
            # remove them from the `stated_dict`.
            del self.Controller._epsilon
            del self.lam
            del self.time
            del self._exponentials
            del self.weights
            torch.save(self.state_dict(), state_dict_path)
        else:
            # During the eRisk laboratory the buffers are desired, thus make a
            # copy of the model.
            new_model = copy.deepcopy(self)
            del new_model.Controller._epsilon
            del new_model.lam
            del new_model.time
            del new_model._exponentials
            del new_model.weights

            torch.save(new_model.state_dict(), state_dict_path)

            del new_model

    @staticmethod
    def load(path_json, for_competition=False):
        """Load EARLIEST model.

        Parameters
        ----------
        path_json : str
            Path to the file containing the state of the EARLIEST model.
        for_competition : bool
            Flag to indicate if it is for competition or not.

        Returns
        --------
        earliest_model : EARLIEST
            The loaded EARLIEST model.
        """
        with open(path_json, "r") as fp:
            model_information = json.load(fp=fp)
        model_params = model_information["model_params"]
        earliest_model = EARLIEST(**model_params)
        state_dict_path = model_information["state_dict_path"]
        earliest_model.load_state_dict(
            torch.load(state_dict_path, map_location=earliest_model.device),
            strict=False,
        )

        if earliest_model.is_competition:
            earliest_model.num_post_processed = model_information["num_post_processed"]
            earliest_model.last_num_posts_processed = model_information[
                "last_num_posts_processed"
            ]
            earliest_model.last_idx_non_stopped_doc = model_information[
                "last_idx_non_stopped_doc"
            ]
            earliest_model.last_docs_rep = np.array(model_information["last_docs_rep"])
            earliest_model.delays = np.array(model_information["delays"])
            earliest_model.already_finished = np.array(
                model_information["already_finished"]
            )
            earliest_model.predictions = np.array(model_information["predictions"])
            earliest_model.probabilities = np.array(model_information["probabilities"])
        if for_competition:
            earliest_model.is_competition = True

        return earliest_model

    def clear_model_state(self):
        """Clear the internal state of the model.

        Use this function if loading a pre-trained EARLIEST model for the
        first time.
        """
        self.last_docs_rep = None
        self.last_num_posts_processed = None
        self.num_post_processed = 0
        self.last_idx_non_stopped_doc = []
        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None

    def get_representation(self, documents, idx_to_remove):
        """Get representation of the documents.

        Parameters
        ----------
        documents : list of str
            Raw documents to get the representation of.
        idx_to_remove : list of int
            Indexes of documents already finished.

        Returns
        -------
        last_docs_rep : numpy.ndarray
            Representation of the documents.
        """
        if self.input_type == "doc2vec":
            if self.representation is None:
                self.representation = gensim.models.doc2vec.Doc2Vec.load(
                    self.representation_path
                )
            if self.last_docs_rep is None:
                max_num_post = max(
                    [len(posts.split(END_OF_POST_TOKEN)) for posts in documents]
                )

                # XXX: This only happens if the snapshot of the run gets corrupted.
                #      For example, by a system reboot while saving the model.
                if max_num_post > self.max_sequence_length:
                    initial_subset_docs_idx = max_num_post - self.max_sequence_length
                else:
                    initial_subset_docs_idx = 0
                current_num_posts = max_num_post - initial_subset_docs_idx
                current_documents = [
                    END_OF_POST_TOKEN.join(
                        posts.split(END_OF_POST_TOKEN)[
                            initial_subset_docs_idx:max_num_post
                        ]
                    )
                    for posts in documents
                ]

                self.last_docs_rep = get_doc2vec_representation(
                    documents=current_documents,
                    doc2vec_model=self.representation,
                    sequential=True,
                    is_competition=True,
                )
                self.last_num_posts_processed = max_num_post
            else:
                replace_old_rep = False
                max_num_post = max(
                    [len(posts.split(END_OF_POST_TOKEN)) for posts in documents]
                )
                if (
                    max_num_post - self.last_num_posts_processed
                    > self.max_sequence_length
                ):
                    initial_subset_docs_idx = max_num_post - self.max_sequence_length
                    replace_old_rep = True
                else:
                    initial_subset_docs_idx = self.last_num_posts_processed
                current_num_posts = max_num_post - initial_subset_docs_idx
                current_documents = [
                    END_OF_POST_TOKEN.join(
                        posts.split(END_OF_POST_TOKEN)[
                            initial_subset_docs_idx:max_num_post
                        ]
                    )
                    for posts in documents
                ]
                current_rep = get_doc2vec_representation(
                    documents=current_documents,
                    doc2vec_model=self.representation,
                    sequential=True,
                    is_competition=True,
                )
                old_seq_length = self.last_docs_rep.shape[1]
                current_seq_length = current_rep.shape[1]
                assert current_num_posts == current_seq_length

                if replace_old_rep:
                    self.last_docs_rep = current_rep
                else:
                    if idx_to_remove:
                        current_num_doc = self.last_docs_rep.shape[0]
                        mask = np.ones(current_num_doc, dtype=bool)
                        remap_idx_to_remove = [
                            i
                            for i, v in enumerate(self.last_idx_non_stopped_doc)
                            if v in idx_to_remove
                        ]
                        mask[remap_idx_to_remove] = False
                        self.last_docs_rep = self.last_docs_rep[mask, :, :]
                    overflow_num = (
                        old_seq_length + current_seq_length - self.max_sequence_length
                    )
                    if overflow_num > 0:
                        self.last_docs_rep = self.last_docs_rep[:, overflow_num:, :]
                    self.last_docs_rep = np.concatenate(
                        (self.last_docs_rep, current_rep), axis=1
                    )
                self.last_num_posts_processed = max_num_post
            assert self.last_docs_rep.shape[1] <= self.max_sequence_length
            return self.last_docs_rep
        else:
            raise Exception(f'Representation "{self.input_type}" not implemented yet.')

    def predict(self, documents_test, delay):
        """Predict the class for the current users' posts.

        Parameters
        ----------
        documents_test : list of str
            List of users' posts.
        delay : int
            Current delay, i.e., post number being processed.

        Returns
        -------
        len_active_users : int
            The number of documents with more posts to process.
        feature_elapsed_time : float
            Time elapsed building the features for the input.
        prediction_elapsed_time : float
            Time elapsed while predicting the input.
        """
        feature_elapsed_time = 0
        prediction_elapsed_time = 0
        start_time = time.time()
        # The first time this function is called, initialize the attributes of
        # the class.
        if self.predictions is None:
            self.predictions = np.array([-1] * len(documents_test))
            self.probabilities = -np.ones_like(self.predictions, dtype=float)
            self.delays = -np.ones_like(self.predictions)
            self.already_finished = np.zeros_like(self.predictions)

        # For the users with no more posts and for which a decision has not been
        # made, issue the last label and store the delay.
        cant_posts_docs = [len(doc.split(END_OF_POST_TOKEN)) for doc in documents_test]
        for j, num_posts in enumerate(cant_posts_docs):
            if num_posts < delay:
                self.already_finished[j] = 1
                if self.delays[j] == -1:
                    self.delays[j] = delay - 1

        # Keep reporting the scores of already flag users for the laboratory eRisk.
        idx_non_stopped_doc = [
            j
            for j, has_finished in enumerate(self.already_finished)
            if not has_finished
        ]
        if self.last_idx_non_stopped_doc:
            idx_newly_stopped_docs = set(self.last_idx_non_stopped_doc) - set(
                idx_non_stopped_doc
            )
            idx_newly_stopped_docs = list(idx_newly_stopped_docs)
        else:
            idx_newly_stopped_docs = []

        if len(idx_non_stopped_doc) > 0:
            documents_not_finished = [documents_test[j] for j in idx_non_stopped_doc]

            x_test = self.get_representation(
                documents_not_finished, idx_newly_stopped_docs
            )
            self.last_idx_non_stopped_doc = idx_non_stopped_doc

            test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float))
            test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

            feature_elapsed_time = time.time() - start_time
            feature_elapsed_time = round(
                feature_elapsed_time, FP_PRECISION_ELAPSED_TIMES
            )

            start_time_pred = time.time()
            y_predicted, probabilities, halting_points = nn_predict_probability(
                classifier=self, loader=test_loader, device=self.device
            )

            y_predicted = y_predicted.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            halting_points = halting_points.cpu().numpy()

            self.predictions[idx_non_stopped_doc] = y_predicted
            self.probabilities[idx_non_stopped_doc] = probabilities

            for j, idx in enumerate(idx_non_stopped_doc):
                if halting_points[j] != -1 and self.delays[idx] == -1 and delay > 4:
                    self.delays[idx] = delay
            self.num_post_processed += 1
            prediction_elapsed_time = time.time() - start_time_pred
            prediction_elapsed_time = round(
                prediction_elapsed_time, FP_PRECISION_ELAPSED_TIMES
            )
        return len(idx_non_stopped_doc), feature_elapsed_time, prediction_elapsed_time

    def forward(self, x, epoch=0, test=False):
        """Compute halting points and predictions."""
        if test:
            # No random decisions during testing.
            self.Controller._epsilon = x.new_zeros(1).float()
        else:
            self.Controller._epsilon = self._exponentials[epoch]  # Explore/exploit

        if self.input_type == "padded_sequential":
            T, B = x.shape
        else:
            T, B, V = x.shape

        baselines = []  # Predicted baselines.
        actions = []  # Which classes to halt at each step.
        log_pi = []  # Log probability of chosen actions.
        halt_probs = []
        hidden = self.init_hidden(B)
        # The function `Tensor.new()` create a new tensor of the same type and
        # in the same device.
        halt_points = -x.new_ones(B).float()
        predictions = x.new_zeros((B, self.n_classes), requires_grad=True).float()

        logits = 0.0

        # For each time-step, select a set of actions.
        for t in range(T):
            # Run Base RNN on new data at step t.
            # Add a new dimension to the tensor for the time step.
            rnn_input = x[t].unsqueeze(0)

            # If (h_0, c_0) are not provided, both h_0 and c_0 default to zero.
            # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
            # In this case, the hidden state must be initialized with the previous hidden state
            # in order to maintain the history of the input.
            output, hidden = self.RNN(rnn_input, hidden)

            # Predict logits for all elements in the batch.
            logits = self.out(output)

            # Compute halting probability and sample an action.
            self.time = self.time.new([t]).view(1, 1, 1).repeat(1, B, 1)

            c_in = torch.cat((output, self.time), dim=2).detach()
            a_t, p_t, w_t = self.Controller(c_in)

            # In order to operate with other tensor we add a new dimension.
            a_t_new_dimension = a_t.unsqueeze(1)

            # If a_t == 1 and this class hasn't been halted, save its logits.
            predictions = torch.where(
                (a_t_new_dimension == 1) & (predictions == 0), logits, predictions
            )

            # If a_t == 1 and this class hasn't been halted, save the time.
            halt_points = torch.where(
                (halt_points == -1) & (a_t == 1), self.time.squeeze(), halt_points
            )

            # Compute baseline.
            b_t = self.BaselineNetwork(torch.cat((output, self.time), dim=2).detach())

            actions.append(a_t)
            baselines.append(b_t.squeeze())
            log_pi.append(p_t)
            halt_probs.append(w_t)
            # If no negative values, every input has been halted.
            if (halt_points == -1).sum() == 0:
                break

        # If one element in the batch has not been halting, use its final prediction
        logits = torch.where(predictions == 0.0, logits, predictions).squeeze(0)
        if not self.is_competition:
            halt_points = torch.where(
                halt_points == -1, self.time.squeeze(), halt_points
            )

        self.baselines = torch.stack(baselines)
        self.log_pi = torch.stack(log_pi)
        self.halt_probs = torch.stack(halt_probs)
        self.actions = torch.stack(actions)

        # Compute mask for where actions are updated.
        # This lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = torch.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[: (1 + halt_points[b]).long(), b] = 1
        return logits, halt_points + 1, (1 + halt_points).mean() / (T + 1)

    def init_hidden(self, bsz):
        """Initialize hidden states."""
        # The function `Tensor.new()` create a new tensor of the same type and
        # in the same device.
        h = (
            self.lam.new_zeros(self.n_layers, bsz, self.n_hidden),
            self.lam.new_zeros(self.n_layers, bsz, self.n_hidden),
        )
        return h

    def compute_loss(self, logits, y):
        """Compute loss."""
        # Compute reward.
        if self.n_classes > 1:
            _, y_hat = torch.max(nn.functional.softmax(logits, dim=1), dim=1)
        else:
            y_hat = torch.round(torch.sigmoid(logits))
            # nn.BCEWithLogitsLoss requires floats.
            y = y.float()
        # Check the correctly classified inputs.
        # For a correct classification the tensor will have a one (1), while
        # for the incorrect it will have a minus one (-1).
        self.r = (2 * (y_hat.float().round() == y.float()).float() - 1).detach()
        # If the batch has only one element, we squeeze the tensor so that
        # the broadcasting keeps working as expected.
        self.grad_mask = self.grad_mask.squeeze(1)
        self.R = self.r * self.grad_mask

        # Rescale reward with baseline.
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # If you want a discount factor, that goes here!
        # It is used in the original implementation.

        # Compute losses.
        mse_loss_function = nn.MSELoss()
        ce_loss_function = (
            nn.BCEWithLogitsLoss(pos_weight=self.weights)
            if self.n_classes == 1
            else nn.CrossEntropyLoss(weight=self.weights)
        )
        # Baseline should approximate mean reward
        self.loss_b = mse_loss_function(b, self.R)
        # RL loss
        self.loss_r = (-self.log_pi * self.adjusted_reward).sum() / self.log_pi.shape[0]
        # Classification loss
        self.loss_c = ce_loss_function(logits, y)
        # Penalize late predictions
        self.wait_penalty = self.halt_probs.sum(1).mean()

        loss = self.loss_r + self.loss_b + self.loss_c + self.lam * (self.wait_penalty)

        # It can help to add a larger weight to self.loss_c so early training
        # focuses on classification: ... + 10*self.loss_c + ...
        return loss, self.loss_r, self.loss_b, self.loss_c, self.lam * self.wait_penalty

    def deploy(self, deploy_path, x_train, y_train, x_test, y_test):
        """Deploy model for usage in competition.

        The deployment of the model involves:
            - Re-training base model using all the available datasets.
            - Copying models' component to the deploy_path.
            - Generating a new `model_information.json` file.

        Parameters
        ----------
        deploy_path : str
            Path where the model will be deployed.

        x_train : list of str
            List of users' publications for training. Each user publication is
            separated using the token `config.END_OF_POST_TOKEN`.

        y_train : list of int
            List of users' labels for training.

        x_test : list of str
            List of users' publications for testing. Each user publication is
            separated using the token `config.END_OF_POST_TOKEN`.

        y_test : list of int
            List of users' labels for testing.
        """
        if self.input_type != "doc2vec":
            raise Exception(f'Representation "{self.input_type}" not implemented yet.')

        # To train we have to set the model to `is_competition = False`.
        self.is_competition = False

        # Set the random seeds.
        random_seed = 30
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Load the representation.
        self.representation = gensim.models.doc2vec.Doc2Vec.load(
            self.representation_path
        )

        # Apply the representation algorithm to each corpus.
        x_train = get_doc2vec_representation(
            documents=x_train,
            doc2vec_model=self.representation,
            sequential=True,
            max_sequence_length=MAX_SEQ_LENGTH,
        )
        train_data = TensorDataset(
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(x_train, dtype=torch.float),
        )

        x_valid = get_doc2vec_representation(
            documents=x_test,
            doc2vec_model=self.representation,
            sequential=True,
            max_sequence_length=MAX_SEQ_LENGTH,
        )
        valid_data = TensorDataset(
            torch.tensor(y_test, dtype=torch.long),
            torch.tensor(x_valid, dtype=torch.float),
        )

        print_message(f"Final processed training corpus: {x_train.shape}")
        print_message(f"Final processed validation corpus: {x_valid.shape}")

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
        valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

        # Calculate the weights to use to account for the imbalanced corpus.
        num_positives = valid_data[:][0].sum()
        num_negatives = len(valid_data) - num_positives

        w = torch.tensor([num_negatives, num_positives], dtype=torch.float32)
        w = w / w.sum()
        w = 1.0 / w

        # Send the model to the device.
        self.to(self.device)

        current_weights = w.clone()
        if self.n_classes == 1:
            current_weights = w[1]
        current_weights.to(self.device)
        self.weights = current_weights

        # XXX: Hack. In the future store them as attributes of the model.
        # TODO: For now, you'll need to select the best learning rate obtained.
        lr = 0.01 if "depression" in deploy_path else 0.1
        num_epochs = 600

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # Training.
        best_validation_loss = float("inf")
        model_state_dict_path = os.path.join(
            deploy_path, "training_earliest_state_dict.pt"
        )
        for epoch in range(num_epochs):
            one_epoch_train_earliest_model(
                earliest_model=self,
                earliest_optimizer=optimizer,
                earliest_scheduler=scheduler,
                loader=train_loader,
                current_epoch=epoch,
                device=self.device,
                num_epochs=num_epochs,
            )

            validation_epoch_loss, _, _, _, _ = validate_earliest_model(
                earliest_model=self, loader=valid_loader, device=self.device
            )

            if epoch > 25 and validation_epoch_loss < best_validation_loss:
                best_validation_loss = validation_epoch_loss
                torch.save(self.state_dict(), model_state_dict_path)
                model_save_epoch = epoch

        # Load the model with the lowest loss in the validation set.
        print_message(f"Loading model from epoch {model_save_epoch}")
        self.load_state_dict(torch.load(model_state_dict_path))

        # Save EARLIEST.
        final_model_path = os.path.join(deploy_path, "deployed_earliest.json")
        self.save(final_model_path)

        # Generate model_information.json file.
        model_information_path = os.path.join(deploy_path, "model_information.json")
        model_information = {
            "model_class": "EARLIEST",
            "model_path": final_model_path,
        }
        with open(model_information_path, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent="\t")


def one_epoch_train_earliest_model(
    earliest_model,
    earliest_optimizer,
    earliest_scheduler,
    loader,
    current_epoch,
    device,
    num_epochs,
):
    """Train EARLIEST model.

    Parameters
    ----------
    earliest_model : EARLIEST
        The EARLIEST model to train.
    earliest_optimizer : torch.optim.Optimizer
        The optimizer to use.
    earliest_scheduler : torch.optim._LRScheduler
        The scheduler for the learning rate to use.
    loader : torch.utils.data.DataLoader
        The DataLoader with the training data.
    current_epoch : int
        The current epoch of training.
    device : torch.device
        Torch device where the model will be trained.
    num_epochs : int
        Total number of times this function will be called.

    Returns
    -------
    delays : torch.Tensor
        The training delays.
    loss : float
        The sum of the loss.
    loss_r : float
        The sum of the reinforcement loss.
    loss_b : float
        The sum of the baseline loss.
    loss_c : float
        The sum of the classification loss.
    loss_e : float
        The sum of the delay loss.
    """
    delays = []
    _loss_sum = 0.0
    _loss_sum_r = 0.0
    _loss_sum_b = 0.0
    _loss_sum_c = 0.0
    _loss_sum_e = 0.0

    earliest_model.train()
    for i, batch in enumerate(loader):
        if type(earliest_model.RNN) == torch.nn.LSTM:
            y, x = batch
        else:
            raise Exception(
                f'model.RNN type "{type(earliest_model.RNN)}" not implemented completely.'
            )
        x, y = x.to(device), y.to(device)

        x = torch.transpose(x, 0, 1)
        # Forward pass.
        logits, halting_points, halting_points_mean = earliest_model(x, current_epoch)

        delays.append(halting_points)

        # Compute gradients and update weights.
        earliest_optimizer.zero_grad()
        loss, loss_r, loss_b, loss_c, loss_e = earliest_model.compute_loss(
            logits.squeeze(1), y
        )
        loss.backward()

        _loss_sum += loss.item()
        _loss_sum_r += loss_r.item()
        _loss_sum_b += loss_b.item()
        _loss_sum_c += loss_c.item()
        _loss_sum_e += loss_e.item()

        earliest_optimizer.step()

        if (i + 1) % 10 == 0:
            print_message(
                f"Epoch [{current_epoch + 1}/{num_epochs}], "
                f"Batch [{i + 1}/{len(loader)}], Loss: {loss.item():.4f}"
            )
    earliest_scheduler.step()

    return (
        torch.cat(delays),
        _loss_sum,
        _loss_sum_r,
        _loss_sum_b,
        _loss_sum_c,
        _loss_sum_e,
    )


def validate_earliest_model(earliest_model, loader, device):
    """Validate EARLIEST model.

    Parameters
    ----------
    earliest_model : EARLIEST
        The EARLIEST model to train.
    loader : torch.utils.data.DataLoader
        The DataLoader with the validation data.
    device : torch.device
        Torch device where the model will be trained.

    Returns
    -------
    loss : float
        The sum of the loss.
    loss_r : float
        The sum of the reinforcement loss.
    loss_b : float
        The sum of the baseline loss.
    loss_c : float
        The sum of the classification loss.
    loss_e : float
        The sum of the delay loss.
    """
    _validation_epoch_loss = 0.0
    _val_loss_sum_r = 0.0
    _val_loss_sum_b = 0.0
    _val_loss_sum_c = 0.0
    _val_loss_sum_e = 0.0

    earliest_model.eval()
    with torch.no_grad():
        for batch in loader:
            if type(earliest_model.RNN) == torch.nn.LSTM:
                y_val, x_val = batch
            else:
                raise Exception(
                    f'model.RNN type "{type(earliest_model.RNN)}" not implemented completely.'
                )
            x_val, y_val = x_val.to(device), y_val.to(device)

            x_val = torch.transpose(x_val, 0, 1)
            logits_val, _, _ = earliest_model(x_val, test=True)

            (
                loss_val,
                loss_val_r,
                loss_val_b,
                loss_val_c,
                loss_val_e,
            ) = earliest_model.compute_loss(logits_val.squeeze(1), y_val)
            _validation_epoch_loss += loss_val.item()
            _val_loss_sum_r += loss_val_r
            _val_loss_sum_b += loss_val_b
            _val_loss_sum_c += loss_val_c
            _val_loss_sum_e += loss_val_e

    return (
        _validation_epoch_loss,
        _val_loss_sum_r,
        _val_loss_sum_b,
        _val_loss_sum_c,
        _val_loss_sum_e,
    )


def test_earliest_model(earliest_model, loader, device):
    """Test EARLIEST model.

    Parameters
    ----------
    earliest_model : EARLIEST
        The EARLIEST model to train.
    loader : torch.utils.data.DataLoader
        The DataLoader with the validation data.
    device : torch.device
        Torch device where the model will be trained.

    Returns
    -------
    predictions : list
        Input predictions.
    labels : list
        Input true labels.
    delays : list
        Delays for input prediction.
    """
    labels = []
    predictions = []
    delays = []

    earliest_model.eval()
    with torch.no_grad():
        for batch in loader:
            if type(earliest_model.RNN) == torch.nn.LSTM:
                y, x = batch
            else:
                raise Exception(
                    f'model.RNN type "{type(earliest_model.RNN)}" not implemented completely.'
                )
            x, y = x.to(device), y.to(device)

            x = torch.transpose(x, 0, 1)
            logits, halting_points, _ = earliest_model(x, test=True)

            if earliest_model.n_classes > 1:
                _, preds = torch.max(nn.functional.softmax(logits, dim=1), dim=1)
            else:
                preds = torch.round(torch.sigmoid(logits)).int()

            delays.append(halting_points)
            predictions.append(preds)
            labels.append(y)
    return predictions, labels, delays


class SS3(CompetitionModel):
    """SS3 based model for early classification.

    Parameters
    ----------
    ss3_model : pyss3.SS3
        Trained SS3 model.
    policy_value : float
        Policy value (gamma) that affects the final decision for each
        user. Note that, if the final score of a user is greater than
        median(scores) + policy_value · MAD(scores) then the user is
        flag as positive, otherwise is flag as negative.
    normalize_score : int, default=0
        Method to normalize the scores.
        If `normalize_score == 0` the score is not normalized and is calculated
        as:
            acc_cv[i][pos_i] - acc_cv[i][neg_i].
        If `normalize_score == 1`  the score of each user is normalized
        using the following formula:
            softmax([acc_cv[i][pos_i] / delay, acc_cv[i][neg_i] / delay]).
        The delay for each user correspond to the last number of post send
        by the user. Note that there are users that stop sending
        publications earlier.
        After that, we retrieve only the score for the positive class.
        If `normalize_score == 2`  the score of each user is normalized
        using the following formula:
            acc_cv[i][pos_i] / (acc_cv[i][pos_i] + acc_cv[i][neg_i])
        In case the sum of both number is close to zero, `0.1` is returned.

    References
    ----------
    .. [1] `Burdisso, S. G., Errecalde, M., & Montes-y-Gómez, M. (2019).
        A text classification framework for simple and effective early
        depression detection over social media streams. Expert Systems
        with Applications, 133, 182-197.
        <https://arxiv.org/abs/1905.08772>`_

    .. [2] `pySS3: A Python package implementing a new model for text
        classification with visualization tools for Explainable AI
        <https://github.com/sergioburdisso/pyss3>`_
    """

    __model__ = None
    __policy_value__ = 2

    # State.
    __acc_cv__ = None

    def __init__(self, ss3_model, policy_value, normalize_score=0):
        self.__model__ = ss3_model
        self.__policy_value__ = policy_value
        self.__normalize_score__ = normalize_score

    def __repr__(self):
        model_information = "-" * 50 + "\n"
        model_information += "SS3 model\n"
        model_information += f"__policy_value__: {self.__policy_value__}\n"
        model_information += f"__normalize_score__: {self.__normalize_score__}\n"
        model_information += "-" * 50 + "\n"
        return model_information

    @staticmethod
    def mad_median(values):
        """Median and median absolute deviation of the scores.

        Parameters
        ----------
        users_scores : list of float
            Users scores.

        Returns
        -------
        m : float
            Median of the users scores.
        sd : float
            Median absolute deviation of the scores.
        """
        values = sorted(values)[::-1]
        n = len(values)
        if n == 2:
            return (values[0], values[0])

        values_m = n // 2 if n % 2 else n // 2 - 1
        m = values[values_m]  # Median
        diffs = sorted([abs(m - v) for v in values])
        sd = diffs[len(diffs) // 2]  # sd Mean
        return m, sd

    @staticmethod
    def load(state_path, model_folder_path, model_name, policy_value, normalize_score):
        """Load SS3 model.

        Parameters
        ----------
        state_path : str
            Path to the file containing the state of the SS3 model.
        model_folder_path : str
            Path to load the model from. Note that, by default, pyss3
            assumes the model checkpoint is placed in a folder named
            "ss3_models". Thus, `model_folder_path` should point to the
            parent folder of the "ss3_models" directory.
        model_name : str
            Name of the model to load.
        policy_value : float
            Policy value (gamma) that affects the final decision for each
            user. Note that, if the final score of a user is greater than
            median(scores) + policy_value · MAD(scores) then the user is
            flag as positive, otherwise is flag as negative.
        normalize_score : int
            If `normalize_score == 0` the score is not normalized and is
            calculated as:
                acc_cv[i][pos_i] - acc_cv[i][neg_i].
            If `normalize_score == 1`  the score of each user is normalized
            using the following formula:
                softmax([acc_cv[i][pos_i] / delay, acc_cv[i][neg_i] / delay]).
            The delay for each user correspond to the last number of post send
            by the user. Note that there are users that stop sending
            publications earlier.
            After that, we retrieve only the score for the positive class.
            If `normalize_score == 2`  the score of each user is normalized
            using the following formula:
                acc_cv[i][pos_i] / (acc_cv[i][pos_i] + acc_cv[i][neg_i])
            In case the sum of both number is close to zero, `0.1` is returned.

        Returns
        -------
        model : pyss3.SS3
            SS3 loaded model.
        """
        # Load model.
        clf = pyss3.SS3(name=model_name)
        clf.load_model(model_folder_path)
        model = SS3(clf, policy_value, normalize_score)

        # Load state.
        try:
            with open(state_path, "r") as fp:
                state = json.load(fp=fp)
            model.__acc_cv__ = state["acc_cv"]
            model.__finished_delay__ = state["finished_delay"]
        except Exception:
            pass

        return model

    def save(self, state_path):
        """Save SS3 model's state to disk.

        Parameters
        ----------
        state_path : str
            Path to save the state of the model.
        """
        state = {
            "acc_cv": self.__acc_cv__,
            "finished_delay": self.__finished_delay__,
        }
        with open(state_path, "w") as fp:
            json.dump(fp=fp, obj=state, indent="\t")

    def predict(self, documents_test, delay):
        """Predict the class for the current users' posts.

        Parameters
        ----------
        documents_test : list of str
            List of users' posts.
        delay : int
            Current delay, i.e., post number being processed.

        Returns
        -------
        decisions : list of int
            List of predicted class for each user. The value 1 indicates
            that the user is at-risk, while 0 indicates the user is not
            at-risk.
        scores : list of float
            List of scores for each user. The score represents the
            estimated level of risk of a user.
        """
        start_time = time.time()
        n_users = len(documents_test)

        # If these are the first posts, then initialize the internal state.
        if delay - 1 == 0:
            self.__acc_cv__ = [None] * n_users
            self.__finished_delay__ = [None] * n_users

        clf = self.__model__
        for i, post in enumerate(documents_test):
            if not post:
                if self.__finished_delay__[i] is None:
                    self.__finished_delay__[i] = delay
                continue

            cv = clf.classify(post, sort=False)
            accumulated_cv = self.__acc_cv__[i]

            if accumulated_cv is None:
                self.__acc_cv__[i] = list(cv)
            else:
                self.__acc_cv__[i] = list(
                    map(float.__add__, map(float, accumulated_cv), map(float, cv))
                )

        pos_i = clf.get_category_index("positive")
        neg_i = clf.get_category_index("negative")

        acc_cv = self.__acc_cv__
        if self.__normalize_score__ == 0:
            scores = [acc_cv[i][pos_i] - acc_cv[i][neg_i] for i in range(len(acc_cv))]
        elif self.__normalize_score__ == 1:
            # Note that we have to take special care when the user stopped
            # writing posts. If we keep updating the acc_cv, her score will tend
            # to 0.5. We keep the scores from all users, even those that already
            # finished posting, but we divide for the last delay they
            # participated.
            scores = [
                softmax([acc_cv[i][pos_i] / delay, acc_cv[i][neg_i] / delay])[0].item()
                if self.__finished_delay__[i] is None
                else softmax(
                    [
                        acc_cv[i][pos_i] / self.__finished_delay__[i],
                        acc_cv[i][neg_i] / self.__finished_delay__[i],
                    ]
                )[0].item()
                for i in range(len(acc_cv))
            ]
        elif self.__normalize_score__ == 2:
            scores = [
                acc_cv[i][pos_i] / (acc_cv[i][pos_i] + acc_cv[i][neg_i])
                if not np.isclose(acc_cv[i][pos_i] + acc_cv[i][neg_i], 0.0)
                else 0.1
                for i in range(len(acc_cv))
            ]
        else:
            raise Exception(
                f"The normalize_score ({self.__normalize_score__}) is invalid. Valid values are 0, 1 or 2"
            )

        # Get median and mad from the scores (ranking).
        m, mad = SS3.mad_median(scores)

        decisions = [0] * len(scores)
        mad_threshold = self.__policy_value__
        for i in range(len(scores)):
            user_at_risk = (scores[i] > m + mad_threshold * mad) and delay > 2
            if user_at_risk:
                decisions[i] = 1

        prediction_elapsed_time = time.time() - start_time
        prediction_elapsed_time = round(
            prediction_elapsed_time, FP_PRECISION_ELAPSED_TIMES
        )

        return decisions, scores, prediction_elapsed_time

    def clear_model_state(self):
        pass

    def deploy(self, deploy_path, x_train, y_train, x_test=None, y_test=None):
        """Deploy model for usage in competition.

        The deployment of the model involves:
            - Re-training base model using all the available datasets.
            - Copying models' component to the deploy_path.
            - Generating a new `model_information.json` file.

        Parameters
        ----------
        deploy_path : str
            Path where the model will be deployed.

        x_train : list of str
            List of users' publications. Each user publication is separated
            using the token `config.END_OF_POST_TOKEN`.

        y_train : list of int
            List of users' labels.

        x_test : None
            Parameter not used. Only present for API consistency.

        y_test : None
            Parameter not used. Only present for API consistency.
        """
        x_train_processed = [
            " ".join(document.split(END_OF_POST_TOKEN)) for document in x_train
        ]
        print_message(f"Final processed corpus length: {len(x_train_processed)}.")
        self.__model__.fit(x_train_processed, y_train)

        # Save retrained model.
        self.__model__.save_model(deploy_path)

        # Generate model_information.json file.
        model_information_path = os.path.join(deploy_path, "model_information.json")
        model_information = {
            "model_class": "SS3",
            "model_name": self.__model__.__name__,
            "model_path": "state.json",
            "policy_value": self.__policy_value__,
            "normalize_score": self.__normalize_score__,
        }
        with open(model_information_path, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent="\t")
