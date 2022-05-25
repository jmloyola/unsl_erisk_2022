"""
Retrain best models.
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
import datetime
import faulthandler
import glob
import json
import os
import pickle
import pprint

import torch
from sklearn.utils import shuffle

from src.config import PATH_BEST_MODELS, PATH_MODELS, PATH_PROCESSED_CORPUS
from src.models.train_model import (
    get_classifier,
    have_same_parameters,
    train_eval_bert_model,
    train_eval_model,
    train_eval_pytorch_model,
)
from src.utils.utilities import print_elapsed_time, print_message


def get_dataframe_best_models(corpus_name, corpus_kind="reddit", measure="f1"):
    """Get DataFrame with the best models.

    Parameters
    ----------
    corpus_name : {'depression', 'gambling'}
        Corpus name.
    corpus_kind : str, default='reddit'
        Corpus kind.
    measure : str, default='f1'

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the best models.
    """
    base_path = os.path.join(PATH_MODELS, corpus_kind, corpus_name)
    sufix = f"_best_models_{measure}.pkl"
    possible_files = glob.glob(f"{base_path}/*{sufix}")
    last_date = None

    for file in possible_files:
        date_str_sup_lim = -len(sufix)
        date_str_inf_lim = -(len(sufix) + len("2020_10_26"))  # random date
        date_str = file[date_str_inf_lim:date_str_sup_lim]
        current_date = datetime.datetime.strptime(date_str, "%Y_%m_%d")

        if last_date is None or last_date < current_date:
            last_date = current_date

    if last_date is None:
        print_message("No file with the best models was found.")
        return None
    file_path = os.path.join(base_path, f'{last_date.strftime("%Y_%m_%d")}{sufix}')
    print_message(
        f"Loading DataFrame with the best models for the metric {measure}: {file_path}"
    )
    with open(file_path, "rb") as f:
        dataframe = pickle.load(f)
    return dataframe


def get_file_paths(dataframe):
    """Get file path for each row of the DataFrame."""
    return os.path.join(
        PATH_MODELS,
        dataframe["corpus_kind"],
        dataframe["corpus_name"],
        dataframe["representation"],
        dataframe["file_name"] + ".json",
    )


def replace_seed_sklearn_model(old_params, new_seed):
    """Replace random seed in parameters dictionary."""
    if "random_state" in old_params:
        old_params["random_state"] = new_seed
        has_seed = True
    else:
        print_message('The parameter "random_state" was not found.')
        has_seed = False
    return old_params, has_seed


def find_representation_json(
    representation, representation_information, corpus_name, corpus_kind
):
    """Find json file related to representation."""
    json_path = None
    possible_files = glob.glob(
        f"{PATH_PROCESSED_CORPUS}/{corpus_kind}/{corpus_name}/{representation}/*.json"
    )

    for file_path in possible_files:
        if have_same_parameters(representation_information, file_path):
            json_path = file_path
    return json_path


def get_corpus_paths(model_information):
    """Get training and testing corpus paths."""
    corpus_name = model_information["corpus_name"]
    corpus_kind = model_information["corpus_kind"]
    representation = model_information["representation"]
    representation_information = model_information["representation_information"]
    train_file_name = None
    test_file_name = None

    if representation in ["bow", "doc2vec"]:
        json_path = find_representation_json(
            representation=representation,
            representation_information=representation_information,
            corpus_name=corpus_name,
            corpus_kind=corpus_kind,
        )
        train_file_path = json_path[: -len(".json")] + "_train.pkl"
        test_file_path = json_path[: -len(".json")] + "_test.pkl"
    elif representation == "bert_tokenizer":
        train_file_path, test_file_path = None, None
    else:
        base_path = os.path.join(
            PATH_PROCESSED_CORPUS, corpus_kind, corpus_name, representation
        )
        if representation == "lda":
            number_topics = representation_information["number_topics"]
            train_file_name = (
                f"{corpus_name}_lda_corpus_{number_topics:02d}topics_train.pkl"
            )
            test_file_name = (
                f"{corpus_name}_lda_corpus_{number_topics:02d}topics_test.pkl"
            )
        elif representation == "lsa":
            number_factors = representation_information["number_factors"]
            train_file_name = (
                f"{corpus_name}_lsa_corpus_{number_factors:03d}factors_train.pkl"
            )
            test_file_name = (
                f"{corpus_name}_lsa_corpus_{number_factors:03d}factors_test.pkl"
            )
        elif representation == "padded_sequential":
            seq_length = representation_information["seq_length"]
            train_file_name = (
                f"{corpus_name}_padded_sequential_{seq_length:06d}_train.pt"
            )
            test_file_name = f"{corpus_name}_padded_sequential_{seq_length:06d}_test.pt"

        train_file_path = os.path.join(base_path, train_file_name)
        test_file_path = os.path.join(base_path, test_file_name)

    return train_file_path, test_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to retrain models using different document representations."
    )
    parser.add_argument(
        "corpus", help="eRisk task corpus name", choices=["depression", "gambling"]
    )
    args = parser.parse_args()

    corpus = args.corpus
    kind = "reddit"

    faulthandler.enable()
    metrics = ["f1", "positive_f1"]
    random_seeds = [value for value in range(1, 31)]

    # Initialize the timer.
    _, _ = print_elapsed_time()

    for measure in metrics:
        print_message("=" * 60)
        df = get_dataframe_best_models(
            corpus_name=corpus, corpus_kind=kind, measure=measure
        )

        file_paths = list(df.apply(get_file_paths, axis=1))

        path_to_best_models = os.path.join(PATH_BEST_MODELS, measure)

        for json_file_path in file_paths:
            print_message("-" * 60)
            print_message(f"Training the model in {json_file_path} multiple times.")

            with open(json_file_path, "r") as fp:
                json_dictionary = json.load(fp=fp)

            classifier_type = json_dictionary["classifier_type"]
            classifier_params = json_dictionary["classifier_params"]
            current_representation = json_dictionary["representation"]

            print_message(f"Classifier type: {classifier_type}")
            print_message("Base parameters: ")
            pprint.pprint(classifier_params)

            train_path, test_path = get_corpus_paths(json_dictionary)

            for i, rnd_seed in enumerate(random_seeds):
                print_message(f"##### Random seed: {rnd_seed} #####")

                if classifier_type == "EmbeddingLSTM":
                    # Since each training stage takes too long, we retrain less times.
                    if i > 4:
                        break

                    train_data = torch.load(f=train_path)
                    test_data = torch.load(f=test_path)

                    train_eval_pytorch_model(
                        train_file_path=train_path,
                        train_data=train_data,
                        test_data=test_data,
                        corpus_name=corpus,
                        corpus_kind=kind,
                        representation=current_representation,
                        replace_old=False,
                        model_params=classifier_params["model_params"],
                        path_to_models=path_to_best_models,
                        random_seed=rnd_seed,
                    )

                elif classifier_type == "BERT":
                    # Since each training stage takes too long, we retrain less times.
                    if i > 14:
                        break

                    train_eval_bert_model(
                        corpus_name=corpus,
                        corpus_kind=kind,
                        replace_old=False,
                        model_params=classifier_params["model_params"],
                        path_to_models=path_to_best_models,
                        random_seed=rnd_seed,
                    )

                else:
                    with open(train_path, "rb") as fp:
                        x_train, y_train = pickle.load(fp)

                    with open(test_path, "rb") as fp:
                        x_test, y_test = pickle.load(fp)

                    new_params, has_parameter_seed = replace_seed_sklearn_model(
                        classifier_params, rnd_seed
                    )
                    classifier = get_classifier(classifier_type, new_params)

                    if not has_parameter_seed:
                        x_train, y_train = shuffle(
                            x_train, y_train, random_state=rnd_seed
                        )
                        x_test, y_test = shuffle(x_test, y_test, random_state=rnd_seed)

                    train_eval_model(
                        classifier=classifier,
                        train_file_path=train_path,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        corpus_name=corpus,
                        corpus_kind=kind,
                        representation=current_representation,
                        replace_old=False,
                        path_to_models=path_to_best_models,
                        random_seed=rnd_seed,
                    )

    print_message("#" * 50)
    print_message("END OF SCRIPT")
