"""
Train the atemporal models.
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
import faulthandler  # Used to catch any segmentation fault errors in the logs.
import glob
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer, RobertaTokenizer

from src.config import (
    PATH_INTERIM_CORPUS,
    PATH_PROCESSED_CORPUS,
    PATH_MODELS,
    PICKLE_PROTOCOL,
    MAX_SEQ_LEN_BERT,
)
from src.models.model import EmbeddingLSTM, BERT
from src.utils.utilities import print_message, print_elapsed_time, have_same_parameters


# Ensure that the PyTorch execution is deterministic, or at least as
# deterministic as possible ^^.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Scikit-learn models to consider.
MODELS_LIST = [
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "SVC",
    "LogisticRegression",
    "MLPClassifier",
    "RandomForestClassifier",
]


def get_classifiers_params():
    """Get a list of tuples with the name and parameters of a sklearn model."""
    params_list = []

    for model_name in MODELS_LIST:
        if model_name == "DecisionTreeClassifier":
            criterion_list = ["gini", "entropy"]
            class_weight_list = [None, "balanced"]
            for criterion in criterion_list:
                for class_weight in class_weight_list:
                    params = {
                        "class_weight": class_weight,
                        "criterion": criterion,
                        "random_state": 30,
                    }
                    params_list.append((model_name, params))

        elif model_name == "KNeighborsClassifier":
            n_neighbors_list = [1, 3, 5, 10]
            w_list = ["uniform", "distance"]
            for n_neighbors in n_neighbors_list:
                for w in w_list:
                    params = {
                        "weights": w,
                        "n_neighbors": n_neighbors,
                    }
                    params_list.append((model_name, params))

        elif model_name == "SVC":
            c_list = [2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9]
            gamma_list = [
                "scale",
                2 ** -15,
                2 ** -13,
                2 ** -11,
                2 ** -9,
                2 ** -7,
                2 ** -5,
                2 ** -3,
                2 ** -1,
                2,
                2 ** 3,
            ]
            class_weight_list = [None, "balanced"]
            for c in c_list:
                for gamma in gamma_list:
                    for class_weight in class_weight_list:
                        params = {
                            "C": c,
                            "gamma": gamma,
                            "probability": True,
                            "class_weight": class_weight,
                            "random_state": 30,
                        }
                        params_list.append((model_name, params))

        elif model_name == "LogisticRegression":
            c_list = [2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9]
            class_weight_list = [None, "balanced"]
            for c in c_list:
                for class_weight in class_weight_list:
                    params = {
                        "C": c,
                        "class_weight": class_weight,
                        "max_iter": 500,
                        "solver": "liblinear",
                    }
                    params_list.append((model_name, params))

        elif model_name == "MLPClassifier":
            hidden_layer_sizes_list = [(50,), (50, 50), (100,), (100, 100), (250,)]
            solver_list = ["adam"]

            for hidden_layer_sizes in hidden_layer_sizes_list:
                for solver in solver_list:
                    params = {
                        "hidden_layer_sizes": hidden_layer_sizes,
                        "solver": solver,
                        "random_state": 30,
                        "max_iter": 500,
                    }
                    params_list.append((model_name, params))

        elif model_name == "RandomForestClassifier":
            n_estimators_list = [50, 100, 200]
            criterion_list = ["gini", "entropy"]
            class_weight_list = [None, "balanced"]
            for n_estimators in n_estimators_list:
                for criterion in criterion_list:
                    for class_weight in class_weight_list:
                        params = {
                            "n_estimators": n_estimators,
                            "class_weight": class_weight,
                            "criterion": criterion,
                            "random_state": 30,
                        }
                        params_list.append((model_name, params))
        else:
            raise ValueError(f'The model "{model_name}" is not supported yet.')
    return params_list


def get_classifier(model_name, params):
    """Get the instantiated classifier."""
    if model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**params)

    elif model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(**params)

    elif model_name == "SVC":
        return SVC(**params)

    elif model_name == "LogisticRegression":
        return LogisticRegression(**params)

    elif model_name == "MLPClassifier":
        return MLPClassifier(**params)

    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(**params)


def train_eval_model(
    classifier,
    train_file_path,
    x_train,
    y_train,
    x_test,
    y_test,
    corpus_name,
    corpus_kind,
    path_to_models,
    representation="bow",
    replace_old=True,
    random_seed=30,
):
    """Train and evaluate a sklearn model.

    Parameters
    ----------
    classifier : sklearn model
        Classifier to train and evaluate.
    train_file_path : str
        Training corpus' path.
    x_train : numpy.ndarray
        Training vectors.
    y_train : numpy.ndarray
        Training target values.
    x_test : numpy.ndarray
        Testing vectors.
    y_test : numpy.ndarray
        Testing target values.
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    path_to_models : str
        Base path used to store the trained models.
    representation : {'bow', 'lda', 'lsa', 'doc2vec'}, default='bow'
        Document representation.
    replace_old : bool, default=True
        If `replace_old=True` replace the last generated model if it exists.
        If `replace_old=False` check if a trained model exists, if that is the
        case print an error message, otherwise, train and evaluate the model.
    random_seed : int, default=30
        Random seed to use.
    """
    print_message(f"Training model:\n{classifier.__repr__()}")
    # Set the random seeds not related to the model.
    np.random.seed(random_seed)
    random.seed(random_seed)

    representation_information = {}
    if representation == "bow":
        json_file_path = train_file_path[:-10] + ".json"
        with open(json_file_path, "r") as f:
            json_dictionary = json.load(fp=f)
        representation_information.update(json_dictionary)
    elif representation == "lda":
        number_topics = int(train_file_path[-18:-16])
        representation_information.update({"number_topics": number_topics})
    elif representation == "lsa":
        number_factors = int(train_file_path[-20:-17])
        representation_information.update({"number_factors": number_factors})
    elif representation == "doc2vec":
        json_file_path = train_file_path[:-10] + ".json"
        with open(json_file_path, "r") as f:
            json_dictionary = json.load(fp=f)
        representation_information.update(json_dictionary)
    else:
        raise Exception(f'Representation "{representation}" not implemented yet.')

    model_information = {
        "classifier_type": classifier.__class__.__name__,
        "classifier_params": classifier.get_params(),
        "corpus_kind": corpus_kind,
        "corpus_name": corpus_name,
        "representation": representation,
        "representation_information": representation_information,
        "train_file_path": train_file_path,
        "random_seed": random_seed,
    }

    base_output_path = os.path.join(
        path_to_models, corpus_kind, corpus_name, representation
    )
    model_information_file_suffix = "_model_information.json"

    possible_files = glob.glob(f"{base_output_path}/*{model_information_file_suffix}")
    max_id = 0
    current_id = 0
    already_exists = False
    for file_path in possible_files:
        sup_lim = len(model_information_file_suffix)
        inf_lim = sup_lim + 4
        current_id = int(file_path[-inf_lim:-sup_lim])
        if current_id > max_id:
            max_id = current_id
        already_exists = have_same_parameters(model_information, file_path)
        if already_exists:
            if replace_old:
                print_message(f"Cleaning the model {file_path} previously trained.")
                os.remove(file_path)
                pickle_model = file_path[:-sup_lim] + "_model_and_report.pkl"
                os.remove(pickle_model)
            else:
                print_message(
                    f"The model {file_path} already exist. Delete it beforehand or call this "
                    "function with the parameter `replace_old=True`."
                )
                return
            break
    model_id = current_id if already_exists else max_id + 1

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    classification_report = metrics.classification_report(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average="macro")
    recall = metrics.recall_score(y_test, y_pred, average="macro")
    f1 = metrics.f1_score(y_test, y_pred, average="macro")
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    print_message(classification_report)

    elapsed_mins, elapsed_secs = print_elapsed_time("Model training")

    model_output_path = os.path.join(
        base_output_path, f"{model_id:04d}_model_and_report.pkl"
    )
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    print_message(
        f'Saving the trained model and the obtained performance metrics in "{model_output_path}".'
    )
    with open(model_output_path, "wb") as f:
        pickle.dump(
            (
                classifier,
                classification_report,
                precision,
                recall,
                f1,
                accuracy,
                confusion_matrix,
                elapsed_mins,
                elapsed_secs,
            ),
            f,
            protocol=PICKLE_PROTOCOL,
        )

    model_information_path = os.path.join(
        base_output_path, f"{model_id:04d}_model_information.json"
    )
    print_message(f'Saving the model information in "{model_information_path}".')
    with open(model_information_path, "w") as f:
        json.dump(fp=f, obj=model_information, indent="\t")


def binary_accuracy(preds, y):
    """Get the accuracy per batch.

    If you get 8/10 right, this returns 0.8, not 8.
    """
    y = y.reshape((-1))
    preds = preds.reshape((-1))
    correct = (preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train_neural_network(model, loader, optimizer, criterion, device, clip=5):
    """Train PyTorch model."""
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in loader:
        if type(model) == EmbeddingLSTM:
            labels, inputs = batch
        elif type(model) == BERT:
            labels, inputs = batch.label, batch.posts
            labels = labels.type(torch.long)
        else:
            raise Exception(f'Model "{type(model)}" not implemented yet.')

        labels, inputs = labels.to(device), inputs.to(device)
        optimizer.zero_grad()

        if type(model) == EmbeddingLSTM:
            logits, h = model(inputs)
            logits = logits.squeeze()
            labels = labels.reshape(logits.size())
            loss = criterion(logits, labels.float())
            # round predictions to the closest integer
            predictions = torch.round(torch.sigmoid(logits))
        else:
            # type(model) == BERT
            logits = model(inputs)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

        acc = binary_accuracy(predictions, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate_neural_network(model, loader, criterion, device):
    """Evaluate PyTorch model."""
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in loader:
            if type(model) == EmbeddingLSTM:
                labels, inputs = batch
                labels, inputs = labels.to(device), inputs.to(device)
                logits, h = model(inputs)
                logits = logits.squeeze()
                labels = labels.reshape(logits.size())
                loss = criterion(logits, labels.float())
                # round predictions to the closest integer
                predictions = torch.round(torch.sigmoid(logits))
            elif type(model) == BERT:
                labels, inputs = batch.label, batch.posts
                labels = labels.type(torch.long)
                labels, inputs = labels.to(device), inputs.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
            else:
                raise Exception(f'Model "{type(model)}" not implemented yet.')

            acc = binary_accuracy(predictions, labels.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def test_classification_report(model, loader, device):
    """Evaluate PyTorch model and get predicted labels.

    Similar to the function `evaluate_neural_network`. In this case,
    all the predicted labels and the true labels are returned.
    """
    model.eval()
    all_labels = None
    all_predictions = None

    with torch.no_grad():
        for batch in loader:
            if type(model) == EmbeddingLSTM:
                labels, inputs = batch
                labels, inputs = labels.to(device), inputs.to(device)
                logits, h = model(inputs)
                logits = logits.squeeze()
                labels = labels.reshape(logits.size())
                logits = logits.reshape((-1))
                labels = labels.reshape((-1))
                # round predictions to the closest integer
                predictions = torch.round(torch.sigmoid(logits))
            elif type(model) == BERT:
                labels, inputs = batch.label, batch.posts
                labels = labels.type(torch.long)
                labels, inputs = labels.to(device), inputs.to(device)
                logits = model(inputs)
                predictions = torch.argmax(logits, dim=1)

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            if all_labels is None:
                all_labels = labels
                all_predictions = predictions
            else:
                all_labels = np.concatenate((all_labels, labels))
                all_predictions = np.concatenate((all_predictions, predictions))
        return all_labels, all_predictions


def epoch_time(start_time, end_time):
    """Get the elapsed minutes and remaining seconds from two times."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    """Get the number of trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_eval_pytorch_model(
    train_file_path,
    train_data,
    test_data,
    corpus_name,
    corpus_kind,
    path_to_models,
    representation="padded_sequential",
    replace_old=True,
    model_params=None,
    random_seed=30,
):
    """Train and evaluate a PyTorch model.

    Parameters
    ----------
    train_file_path : str
        Training corpus' path.
    train_data : torch.utils.data.TensorDataset
        Training corpus (labels and vectors).
    test_data : torch.utils.data.TensorDataset
        Testing corpus (labels and vectors).
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    path_to_models : str
        Base path used to store the trained models.
    representation : {'padded_sequential'}, default='padded_sequential'
        Document representation.
    replace_old : bool, default=True
        If `replace_old=True` replace the last generated model if it exists.
        If `replace_old=False` check if a trained model exists, if that is the
        case print an error message, otherwise, train and evaluate the model.
    model_params : dict, default=None
        Model parameters.
    random_seed : int, default=30
        Random seed to use.
    """
    # Set the random seeds.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    batch_size = 5
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    representation_information = {}

    if representation == "padded_sequential":
        seq_length = int(train_file_path[-15:-9])
        representation_information.update({"seq_length": seq_length})

        vocabulary_path = train_file_path[:-8] + "vocabulary.pkl"
        with open(vocabulary_path, "rb") as f:
            _, vocab_to_int, _ = pickle.load(f)

        # Sum two (2) to the number of vocabulary to integer entries to account
        # for the padding and unknown tokens.
        n_vocab = len(vocab_to_int) + 2
        n_embed = 200
        n_hidden = 256
        n_output = 1  # 1 ("positive") or 0 ("negative")
        n_layers = model_params.get("n_layers", 1)

        model = EmbeddingLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers)
    else:
        raise Exception(f'Representation "{representation}" not implemented yet.')

    print_message(
        f"Training the model {model.__class__.__name__} with a {representation} representation "
        f"of length {seq_length}."
    )
    print_message("The architecture of the model is:")
    print_message(model)
    print_message(
        f"The model has {count_parameters(model):_} parameters that change while training."
    )

    classifier_params = {
        "model_architecture": model.__repr__(),
        "model_params": model_params,
    }
    model_information = {
        "classifier_type": model.__class__.__name__,
        "classifier_params": classifier_params,
        "corpus_kind": corpus_kind,
        "corpus_name": corpus_name,
        "representation": representation,
        "representation_information": representation_information,
        "train_file_path": train_file_path,
        "random_seed": random_seed,
    }

    base_output_path = os.path.join(
        path_to_models, corpus_kind, corpus_name, representation
    )
    model_information_file_suffix = "_model_information.json"

    possible_files = glob.glob(f"{base_output_path}/*{model_information_file_suffix}")
    max_id = 0
    current_id = 0
    already_exists = False
    for file_path in possible_files:
        sup_lim = len(model_information_file_suffix)
        inf_lim = sup_lim + 2
        current_id = int(file_path[-inf_lim:-sup_lim])
        if current_id > max_id:
            max_id = current_id
        already_exists = have_same_parameters(model_information, file_path)
        if already_exists:
            if replace_old:
                print_message(f"Cleaning the model {file_path} previously trained.")
                os.remove(file_path)
                model_parameters_path = file_path[:-sup_lim] + "_model_parameters.pt"
                os.remove(model_parameters_path)
                pickle_model = file_path[:-sup_lim] + "_model_and_report.pkl"
                os.remove(pickle_model)
            else:
                print_message(
                    f"The model {file_path} already exist. Delete it beforehand or call this "
                    "function with the parameter `replace_old=True`."
                )
                return
            break
    model_id = current_id if already_exists else max_id + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Device --> {device}")

    model = model.to(device)

    current_weights = model_params.get("weights")
    if current_weights is None:
        # Calculate the weights to use to account for the imbalanced corpus.
        num_positives = train_data.dataset[:][0].sum()
        num_negatives = len(train_data.dataset) - num_positives

        w = torch.tensor([num_negatives, num_positives], dtype=torch.float32)
        w = w / w.sum()
        w = 1.0 / w

        if n_output == 1:
            w = w[1]
    else:
        w = torch.tensor(current_weights, dtype=torch.float32)

    w.to(device)

    learning_rate = model_params.get("lr", 0.0001)
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=w)
        if n_output == 1
        else nn.CrossEntropyLoss(weight=w)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = criterion.to(device)

    n_epochs = 25
    best_test_loss = float("inf")

    model_parameters_path = os.path.join(
        base_output_path, f"{model_id:02d}_model_parameters.pt"
    )
    os.makedirs(os.path.dirname(model_parameters_path), exist_ok=True)

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = train_neural_network(
            model, train_data, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate_neural_network(
            model, test_data, criterion, device
        )

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

            torch.save(model.state_dict(), model_parameters_path)

        print_message(
            f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
        )
        print_message(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
        )
        print_message(
            f"\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%"
        )

    # Load the model with best performance in the validation set.
    model.load_state_dict(torch.load(model_parameters_path))

    test_loss, test_acc = evaluate_neural_network(model, test_data, criterion, device)

    # Calculate the loss in the test corpus with the best parameters obtained.
    print_message("·" * 50)
    print_message("Loss for the testing corpus with the best parameters obtained.")
    print_message(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%")
    print_message("·" * 50)

    # Calculate the performance metrics.
    all_labels, all_predictions = test_classification_report(model, test_data, device)

    classification_report = metrics.classification_report(all_labels, all_predictions)
    precision = metrics.precision_score(all_labels, all_predictions, average="macro")
    recall = metrics.recall_score(all_labels, all_predictions, average="macro")
    f1 = metrics.f1_score(all_labels, all_predictions, average="macro")
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    confusion_matrix = metrics.confusion_matrix(all_labels, all_predictions)

    print_message(classification_report)

    elapsed_mins, elapsed_secs = print_elapsed_time("Model training ended")

    model_output_path = os.path.join(
        base_output_path, f"{model_id:02d}_model_and_report.pkl"
    )
    print_message(
        f'Saving the trained models and its metrics in "{model_output_path}".'
    )
    with open(model_output_path, "wb") as f:
        # The first element of the tuple refers to the model.
        # As in this case the parameters are stored in another file we do not place it here.
        pickle.dump(
            (
                None,
                classification_report,
                precision,
                recall,
                f1,
                accuracy,
                confusion_matrix,
                elapsed_mins,
                elapsed_secs,
            ),
            f,
            protocol=PICKLE_PROTOCOL,
        )

    model_information_path = os.path.join(
        base_output_path, f"{model_id:02d}_model_information.json"
    )
    print_message(f'Saving the model information in "{model_information_path}".')
    with open(model_information_path, "w") as f:
        json.dump(fp=f, obj=model_information, indent="\t")


def train_eval_bert_model(
    corpus_name,
    corpus_kind,
    path_to_models,
    replace_old=True,
    model_params=None,
    random_seed=30,
):
    """Train and evaluate a BERT model.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : {'xml', 'reddit'}
        Corpus kind.
    path_to_models : str
        Base path used to store the trained models.
    replace_old : bool, default=True
        If `replace_old=True` replace the last generated model if it exists.
        If `replace_old=False` check if a trained model exists, if that is the
        case print an error message, otherwise, train and evaluate the model.
    model_params : dict, default=None
        Model parameters.
    random_seed : int, default=30
        Random seed to use.
    """
    # Set the random seeds.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Device --> {device}")

    input_file_path = os.path.join(PATH_INTERIM_CORPUS, corpus_kind, corpus_name)
    input_file_name_train = f"{corpus_name}_truncated_train.csv"
    input_file_name_test = f"{corpus_name}_truncated_test.csv"

    bert_architecture = model_params.get("bert_architecture", "bert-base-uncased")

    # Generate the corpus with BERT tokens.
    if bert_architecture == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(bert_architecture)
    elif bert_architecture == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(bert_architecture)
    else:
        raise Exception(f'The architecture "{bert_architecture}" is not supported yet.')

    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    label_field = Field(
        sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
    )
    posts_field = Field(
        use_vocab=False,
        tokenize=tokenizer.encode,
        lower=False,
        include_lengths=False,
        batch_first=True,
        fix_length=MAX_SEQ_LEN_BERT,
        pad_token=pad_index,
        unk_token=unk_index,
    )
    fields = [("label", label_field), ("posts", posts_field)]

    train_data, test_data = TabularDataset.splits(
        path=input_file_path,
        train=input_file_name_train,
        validation=None,
        test=input_file_name_test,
        format="CSV",
        fields=fields,
        skip_header=True,
    )

    batch_size = 3 if bert_architecture == "bert-large-uncased" else 10

    train_iter = BucketIterator(
        train_data,
        batch_size=batch_size,
        sort_key=lambda x: len(x.posts),
        device=device,
        train=True,
        sort=True,
        sort_within_batch=True,
    )
    test_iter = Iterator(
        test_data,
        batch_size=batch_size,
        device=device,
        train=False,
        shuffle=False,
        sort=False,
    )

    representation_information = tokenizer.__repr__()

    # Get the number of label of the corpus.
    df_train = pd.read_csv(os.path.join(input_file_path, input_file_name_train))
    num_labels = len(df_train.label.unique())

    model = BERT(
        bert_architecture=bert_architecture,
        num_labels=num_labels,
        freeze_encoder=model_params.get("freeze_encoder", False),
    )

    print_message(f"Training the model {model.__class__.__name__}.")
    print_message("The architecture of the model is:")
    print_message(model)
    print_message(
        f"The model has {count_parameters(model):_} parameters that change while training."
    )

    representation = "bert_tokenizer"
    classifier_params = {
        "model_architecture": model.__repr__(),
        "model_params": model_params,
    }
    model_information = {
        "classifier_type": model.__class__.__name__,
        "classifier_params": classifier_params,
        "corpus_kind": corpus_kind,
        "corpus_name": corpus_name,
        "representation": representation,
        "representation_information": representation_information,
        "train_file_path": os.path.join(input_file_path, input_file_name_train),
        "random_seed": random_seed,
    }

    base_output_path = os.path.join(
        path_to_models, corpus_kind, corpus_name, representation
    )
    model_information_file_suffix = "_model_information.json"

    possible_files = glob.glob(f"{base_output_path}/*{model_information_file_suffix}")
    max_id = 0
    current_id = 0
    already_exists = False
    for file_path in possible_files:
        sup_lim = len(model_information_file_suffix)
        inf_lim = sup_lim + 2
        current_id = int(file_path[-inf_lim:-sup_lim])
        if current_id > max_id:
            max_id = current_id
        already_exists = have_same_parameters(model_information, file_path)
        if already_exists:
            if replace_old:
                print_message(f"Cleaning the model {file_path} previously trained.")
                os.remove(file_path)
                model_parameters_path = file_path[:-sup_lim] + "_model_parameters.pt"
                os.remove(model_parameters_path)
                pickle_model = file_path[:-sup_lim] + "_model_and_report.pkl"
                os.remove(pickle_model)
            else:
                print_message(
                    f"The model {file_path} already exist. Delete it beforehand or call this "
                    "function with the parameter `replace_old=True`."
                )
                return
            break
    model_id = current_id if already_exists else max_id + 1

    current_weights = model_params.get("weights")
    if current_weights is None:
        # Calculate the weights to use to account for the imbalanced corpus.
        num_positives = df_train.label.sum()
        num_negatives = len(df_train) - num_positives

        w = torch.tensor([num_negatives, num_positives], dtype=torch.float32)
        w = w / w.sum()
        w = 1.0 / w

        if model_params.get("normalize_weights"):
            w = w / w.sum()
    else:
        assert type(current_weights) is list
        assert len(current_weights) == num_labels

        w = torch.tensor(current_weights, dtype=torch.float32)

    w.to(device)

    learning_rate = model_params.get("lr", 1e-5)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 20 if model_params.get("freeze_encoder") else 10
    best_test_loss = float("inf")

    model_parameters_path = os.path.join(
        base_output_path, f"{model_id:02d}_model_parameters.pt"
    )
    os.makedirs(os.path.dirname(model_parameters_path), exist_ok=True)

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = train_neural_network(
            model, train_iter, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate_neural_network(
            model, test_iter, criterion, device
        )

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

            torch.save(model.state_dict(), model_parameters_path)

        print_message(
            f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
        )
        print_message(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
        )
        print_message(
            f"\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%"
        )

    # Load the model with best performance in the validation set.
    model.load_state_dict(torch.load(model_parameters_path))

    test_loss, test_acc = evaluate_neural_network(model, test_iter, criterion, device)

    # Calculate the loss in the test corpus with the best parameters obtained.
    print_message("·" * 50)
    print_message("Loss for the testing corpus with the best parameters obtained.")
    print_message(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%")
    print_message("·" * 50)

    # Calculate the performance metrics.
    all_labels, all_predictions = test_classification_report(model, test_iter, device)

    classification_report = metrics.classification_report(all_labels, all_predictions)
    precision = metrics.precision_score(all_labels, all_predictions, average="macro")
    recall = metrics.recall_score(all_labels, all_predictions, average="macro")
    f1 = metrics.f1_score(all_labels, all_predictions, average="macro")
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    confusion_matrix = metrics.confusion_matrix(all_labels, all_predictions)

    print_message(classification_report)

    elapsed_mins, elapsed_secs = print_elapsed_time("Model training ended")

    model_output_path = os.path.join(
        base_output_path, f"{model_id:02d}_model_and_report.pkl"
    )
    print_message(
        f'Saving the trained models and its metrics in "{model_output_path}".'
    )
    with open(model_output_path, "wb") as f:
        # The first element of the tuple refers to the model.
        # As in this case the parameters are stored in another file we do not place it here.
        pickle.dump(
            (
                None,
                classification_report,
                precision,
                recall,
                f1,
                accuracy,
                confusion_matrix,
                elapsed_mins,
                elapsed_secs,
            ),
            f,
            protocol=PICKLE_PROTOCOL,
        )

    model_information_path = os.path.join(
        base_output_path, f"{model_id:02d}_model_information.json"
    )
    print_message(f'Saving the model information in "{model_information_path}".')
    with open(model_information_path, "w") as f:
        json.dump(fp=f, obj=model_information, indent="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the atemporal models using different document representations."
    )
    parser.add_argument(
        "corpus", help="eRisk task corpus name", choices=["depression", "gambling"]
    )
    args = parser.parse_args()
    kind = "reddit"
    faulthandler.enable()

    partial_corpus_path = os.path.join(PATH_PROCESSED_CORPUS, kind, args.corpus)

    # Initialize the timer.
    _, _ = print_elapsed_time()

    # Training models using the Bag of Words (BoW) representation.
    print_message("Training models using the Bag of Words (BoW) representation.")
    bow_corpus_paths = glob.glob(
        f"{partial_corpus_path}/bow/{args.corpus}_bow_*_train.pkl"
    )
    for train_path in bow_corpus_paths:
        test_file_path = train_path[:-9] + "test.pkl"

        with open(train_path, "rb") as fp:
            xtrain, ytrain = pickle.load(fp)

        with open(test_file_path, "rb") as fp:
            xtest, ytest = pickle.load(fp)

        for i, (clf_name, clf_params) in enumerate(get_classifiers_params()):
            clf = get_classifier(clf_name, clf_params)
            train_eval_model(
                classifier=clf,
                train_file_path=train_path,
                x_train=xtrain,
                y_train=ytrain,
                x_test=xtest,
                y_test=ytest,
                corpus_name=args.corpus,
                corpus_kind=kind,
                representation="bow",
                replace_old=False,
                path_to_models=PATH_MODELS,
            )
            print_message("  -" * 20)

    print_message("=" * 60)
    # Training models using the Latent Dirichlet Allocation (LDA) representation.
    print_message(
        "Training models using the Latent Dirichlet Allocation (LDA) representation."
    )
    lda_corpus_paths = glob.glob(
        f"{partial_corpus_path}/lda/{args.corpus}_lda_corpus_*topics_train.pkl"
    )
    for train_path in lda_corpus_paths:
        test_file_path = train_path[:-9] + "test.pkl"

        with open(train_path, "rb") as fp:
            xtrain, ytrain = pickle.load(fp)

        with open(test_file_path, "rb") as fp:
            xtest, ytest = pickle.load(fp)

        for i, (clf_name, clf_params) in enumerate(get_classifiers_params()):
            clf = get_classifier(clf_name, clf_params)
            train_eval_model(
                classifier=clf,
                train_file_path=train_path,
                x_train=xtrain,
                y_train=ytrain,
                x_test=xtest,
                y_test=ytest,
                corpus_name=args.corpus,
                corpus_kind=kind,
                representation="lda",
                replace_old=False,
                path_to_models=PATH_MODELS,
            )
            print_message("  -" * 20)

    print_message("=" * 60)
    # Training models using the Latent Semantic Analysis (LSA) representation
    print_message(
        "Training models using the Latent Semantic Analysis (LSA) representation."
    )
    lsa_corpus_paths = glob.glob(
        f"{partial_corpus_path}/lsa/{args.corpus}_lsa_corpus_*factors_train.pkl"
    )
    for train_path in lsa_corpus_paths:
        test_file_path = train_path[:-9] + "test.pkl"

        with open(train_path, "rb") as fp:
            xtrain, ytrain = pickle.load(fp)

        with open(test_file_path, "rb") as fp:
            xtest, ytest = pickle.load(fp)

        for i, (clf_name, clf_params) in enumerate(get_classifiers_params()):
            clf = get_classifier(clf_name, clf_params)
            train_eval_model(
                classifier=clf,
                train_file_path=train_path,
                x_train=xtrain,
                y_train=ytrain,
                x_test=xtest,
                y_test=ytest,
                corpus_name=args.corpus,
                corpus_kind=kind,
                representation="lsa",
                replace_old=False,
                path_to_models=PATH_MODELS,
            )
            print_message("  -" * 20)

    print_message("=" * 60)
    # Training models using the doc2vec representation.
    print_message("Training models using the doc2vec representation.")
    doc2vec_corpus_paths = glob.glob(
        f"{partial_corpus_path}/doc2vec/{args.corpus}_doc2vec_*_train.pkl"
    )
    for train_path in doc2vec_corpus_paths:
        test_file_path = train_path[:-9] + "test.pkl"

        with open(train_path, "rb") as fp:
            xtrain, ytrain = pickle.load(fp)

        with open(test_file_path, "rb") as fp:
            xtest, ytest = pickle.load(fp)

        for i, (clf_name, clf_params) in enumerate(get_classifiers_params()):
            clf = get_classifier(clf_name, clf_params)
            train_eval_model(
                classifier=clf,
                train_file_path=train_path,
                x_train=xtrain,
                y_train=ytrain,
                x_test=xtest,
                y_test=ytest,
                corpus_name=args.corpus,
                corpus_kind=kind,
                representation="doc2vec",
                replace_old=False,
                path_to_models=PATH_MODELS,
            )
            print_message("  -" * 20)

    print_message("=" * 60)
    # Training models using the padded sequential representation.
    print_message("Training models using the padded sequential representation.")
    padded_sequential_train_corpus_paths = glob.glob(
        f"{partial_corpus_path}/padded_sequential/"
        f"{args.corpus}_padded_sequential_*_train.pt"
    )

    for train_path in padded_sequential_train_corpus_paths:
        test_file_path = train_path[:-8] + "test.pt"

        traindata = torch.load(f=train_path)
        testdata = torch.load(f=test_file_path)

        n_layers_list = [1]
        lr_list = [0.0001]
        weights_list = [None, 1.0]

        for lr in lr_list:
            for num_layers in n_layers_list:
                for weights in weights_list:
                    model_parameters = {
                        "n_layers": num_layers,
                        "lr": lr,
                        "weights": weights,
                    }
                    train_eval_pytorch_model(
                        train_file_path=train_path,
                        train_data=traindata,
                        test_data=testdata,
                        corpus_name=args.corpus,
                        corpus_kind=kind,
                        representation="padded_sequential",
                        replace_old=False,
                        model_params=model_parameters,
                        path_to_models=PATH_MODELS,
                    )
                    print_message("  -" * 20)

    print_message("=" * 60)
    # Training BERT models.
    print_message("Training BERT models.")

    bert_architecture_list = ["bert-base-uncased", "roberta-base"]
    lr_list = [0.0001, 0.00001]
    weights_list = [None, [1.0, 1.0]]
    normalize_weights_list = [True, False]
    freeze_encoder_list = [True, False]

    for bert_arch in bert_architecture_list:
        for lr in lr_list:
            for weights in weights_list:
                for normalize_weights in normalize_weights_list:
                    for freeze_encoder_flag in freeze_encoder_list:
                        model_parameters = {
                            "bert_architecture": bert_arch,
                            "lr": lr,
                            "weights": weights,
                            "freeze_encoder": freeze_encoder_flag,
                        }
                        if weights is None:
                            model_parameters.update(
                                {"normalize_weights": normalize_weights}
                            )
                        train_eval_bert_model(
                            corpus_name=args.corpus,
                            corpus_kind=kind,
                            replace_old=False,
                            model_params=model_parameters,
                            path_to_models=PATH_MODELS,
                        )
                        print_message("  -" * 20)

    print_message("#" * 50)
    print_message("END OF SCRIPT")
