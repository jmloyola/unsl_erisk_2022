"""
Deploy models for the laboratory.
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
import faulthandler
import json
import os

from src.config import PATH_DEPLOY, PATH_INTERIM_CORPUS
from src.models.model import EARLIEST, SS3, EarlyModel
from src.utils.utilities import print_message


def merge_datasets(corpus, model_type):
    """Merge the available datasets."""
    print_message(f"Merging {corpus} datasets for {model_type}")

    xml_train_documents = []
    xml_train_labels = []
    xml_train_path = os.path.join(
        PATH_INTERIM_CORPUS, "xml", corpus, f"{corpus}-train-raw.txt"
    )
    if os.path.exists(xml_train_path):
        with open(xml_train_path, "r") as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                label = 1 if label == "positive" else 0
                xml_train_labels.append(label)
                xml_train_documents.append(posts)
    else:
        print_message(f"The corpus {xml_train_path} does not exists.")

    xml_test_documents = []
    xml_test_labels = []
    xml_test_path = os.path.join(
        PATH_INTERIM_CORPUS, "xml", corpus, f"{corpus}-test-raw.txt"
    )
    if os.path.exists(xml_test_path):
        with open(xml_test_path, "r") as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                label = 1 if label == "positive" else 0
                xml_test_labels.append(label)
                xml_test_documents.append(posts)
    else:
        print_message(f"The corpus {xml_test_path} does not exists.")

    reddit_train_documents = []
    reddit_train_labels = []
    reddit_train_path = os.path.join(
        PATH_INTERIM_CORPUS, "reddit", corpus, f"{corpus}-train-raw.txt"
    )
    if os.path.exists(reddit_train_path):
        with open(reddit_train_path, "r") as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                label = 1 if label == "positive" else 0
                reddit_train_labels.append(label)
                reddit_train_documents.append(posts)
    else:
        print_message(f"The corpus {reddit_train_path} does not exists.")

    reddit_test_documents = []
    reddit_test_labels = []
    reddit_test_path = os.path.join(
        PATH_INTERIM_CORPUS, "reddit", corpus, f"{corpus}-test-raw.txt"
    )
    if os.path.exists(reddit_test_path):
        with open(reddit_test_path, "r") as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                label = 1 if label == "positive" else 0
                reddit_test_labels.append(label)
                reddit_test_documents.append(posts)
    else:
        print_message(f"The corpus {reddit_test_path} does not exists.")

    if model_type == "EARLIEST":
        x_train = xml_train_documents + reddit_train_documents + reddit_test_documents
        y_train = xml_train_labels + reddit_train_labels + reddit_test_labels
        x_test = xml_test_documents
        y_test = xml_test_labels
    else:
        x_train = (
            xml_train_documents
            + xml_test_documents
            + reddit_train_documents
            + reddit_test_documents
        )
        y_train = (
            xml_train_labels
            + xml_test_labels
            + reddit_train_labels
            + reddit_test_labels
        )
        x_test = None
        y_test = None

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to deploy models.")
    parser.add_argument(
        "-c",
        "--corpus",
        help="eRisk task corpus name",
        choices=["depression", "gambling"],
        required=True,
    )
    parser.add_argument(
        "-p",
        "--model_path",
        help="path to selected models",
        action="append",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--model_type",
        help="selected models type",
        action="append",
        choices=["EarlyModel", "SS3", "EARLIEST"],
        required=True,
    )
    parser.add_argument(
        "-i",
        "--model_index",
        help="index for deployed models",
        action="append",
        type=str,
        choices=["0", "1", "2", "3", "4"],
        required=True,
    )
    args = parser.parse_args()

    faulthandler.enable()

    if len(args.model_path) != len(args.model_type) or (
        len(args.model_index) != len(args.model_path)
    ):
        raise Exception(
            f"The number of model paths, model types and model indices must be the same. The "
            f"len(args.model_path) is {len(args.model_path)}, len(args.model_type) is "
            f"{len(args.model_type)} and len(args.model_index) is {len(args.model_index)}."
        )

    at_least_one_earliest = sum([1 if m == "EARLIEST" else 0 for m in args.model_type])
    if at_least_one_earliest > 0:
        print_message(
            "Remember to modify the json file with the missing information of the EARLIEST models. "
            'The two attributes missing are: the optimizer learning rate ("lr") and the value of '
            'lambda ("lam").'
        )

    num_models = len(args.model_path)
    print_message(
        f"Deploying {num_models} models. For that we re-train the models using all the datasets."
    )

    for idx, model_path, model_type in zip(
        args.model_index, args.model_path, args.model_type
    ):
        print_message(f"Deploying {model_type} in {model_path}.")

        if model_type == "EarlyModel":
            model = EarlyModel.load(model_path)
        elif model_type == "SS3":
            # For SS3 we provide the path to the model_information.json
            model_information_path = model_path
            with open(model_information_path, "r") as fp:
                model_information = json.load(fp=fp)

            model_information_folder_path = os.path.dirname(model_information_path)
            model_path = model_information["model_path"]
            state_path = os.path.join(model_information_folder_path, model_path)
            model = SS3.load(
                state_path=state_path,
                model_folder_path=model_information_folder_path,
                model_name=model_information["model_name"],
                policy_value=model_information["policy_value"],
                normalize_score=model_information["normalize_score"],
            )
        elif model_type == "EARLIEST":
            model = EARLIEST.load(model_path, for_competition=False)

        model.clear_model_state()

        x_train, y_train, x_test, y_test = merge_datasets(
            corpus=args.corpus, model_type=model_type
        )

        current_deploy_path = os.path.join(PATH_DEPLOY, args.corpus, idx)
        os.makedirs(current_deploy_path, exist_ok=True)
        model.deploy(current_deploy_path, x_train, y_train, x_test, y_test)
        print_message(
            f"Finished deploying {model_type} ({model_path}) to {current_deploy_path}."
        )

    print_message("#" * 50)
    print_message("END OF SCRIPT")
