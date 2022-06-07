"""
EARLIEST training script.
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
import random
import time

import gensim
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    BATCH_SIZE,
    MAX_SEQ_LENGTH,
    PATH_BEST_MODELS,
    PATH_INTERIM_CORPUS,
    PATH_MODELS,
    PICKLE_PROTOCOL,
)
from src.features.build_features import get_doc2vec_representation
from src.models.model import (
    EARLIEST,
    one_epoch_train_earliest_model,
    test_earliest_model,
    validate_earliest_model,
)
from src.utils.performance_metrics import erde, f_latency, value_p
from src.utils.utilities import get_documents_from_corpus, print_message

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train earliest model.")
    parser.add_argument(
        "--corpus", help="eRisk task corpus name", choices=["depression", "gambling"]
    )
    parser.add_argument(
        "--device", help="device to use", choices=["auto", "cpu", "gpu"]
    )
    args = parser.parse_args()

    corpus = args.corpus
    kind = "reddit"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print_message("CUDA drivers not installed. Device fallback to cpu")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print_message(f"Using the device '{device}'.")

    # Set the random seeds.
    random_seed = 30
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    faulthandler.enable()

    PATH_EARLIEST_MODELS = os.path.join(PATH_MODELS, kind, corpus, "earliest")
    os.makedirs(PATH_EARLIEST_MODELS, exist_ok=True)

    # These are the representations used by the best EarlyModels with doc2vec.
    # TODO: update path
    rep_path = (
        "08_representation_doc2vec.pkl"
        if corpus == "depression"
        else "09_representation_doc2vec.pkl"
    )
    DOC2VEC_REPRESENTATIONS = os.path.join(
        PATH_BEST_MODELS, "positive_f1", kind, corpus, "selected_models", rep_path
    )
    output_train_file_path = os.path.join(PATH_EARLIEST_MODELS, "train_data.pt")
    output_valid_file_path = os.path.join(PATH_EARLIEST_MODELS, "valid_data.pt")
    output_test_file_path = os.path.join(PATH_EARLIEST_MODELS, "test_data.pt")
    test_corpus_information_path = os.path.join(
        PATH_EARLIEST_MODELS, "test_corpus_information.json"
    )
    doc2vec_representation = gensim.models.doc2vec.Doc2Vec.load(DOC2VEC_REPRESENTATIONS)

    if not os.path.exists(output_train_file_path):
        input_train_file_path = os.path.join(
            PATH_INTERIM_CORPUS, kind, corpus, f"{corpus}-train-clean.txt"
        )
        x_train, y_train = get_documents_from_corpus(input_train_file_path)
        x_train = get_doc2vec_representation(
            documents=x_train,
            doc2vec_model=doc2vec_representation,
            sequential=True,
            max_sequence_length=MAX_SEQ_LENGTH,
        )
        train_data = TensorDataset(
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(x_train, dtype=torch.float),
        )
        torch.save(
            obj=train_data, f=output_train_file_path, pickle_protocol=PICKLE_PROTOCOL
        )
    else:
        train_data = torch.load(output_train_file_path)

    if not os.path.exists(output_valid_file_path):
        input_valid_file_path = os.path.join(
            PATH_INTERIM_CORPUS, kind, corpus, f"{corpus}-test-clean.txt"
        )
        x_valid, y_valid = get_documents_from_corpus(input_valid_file_path)
        x_valid = get_doc2vec_representation(
            documents=x_valid,
            doc2vec_model=doc2vec_representation,
            sequential=True,
            max_sequence_length=MAX_SEQ_LENGTH,
        )
        valid_data = TensorDataset(
            torch.tensor(y_valid, dtype=torch.long),
            torch.tensor(x_valid, dtype=torch.float),
        )
        torch.save(
            obj=valid_data, f=output_valid_file_path, pickle_protocol=PICKLE_PROTOCOL
        )
    else:
        valid_data = torch.load(output_valid_file_path)

    if not os.path.exists(output_test_file_path):
        # For testing we used the xml corpus provided by the organizers.
        input_test_file_path = os.path.join(
            PATH_INTERIM_CORPUS, "xml", corpus, f"{corpus}-test-clean.txt"
        )
        x_test, y_test = get_documents_from_corpus(
            input_test_file_path, corpus_info_path=test_corpus_information_path
        )
        x_test = get_doc2vec_representation(
            documents=x_test,
            doc2vec_model=doc2vec_representation,
            sequential=True,
            max_sequence_length=MAX_SEQ_LENGTH,
        )
        test_data = TensorDataset(
            torch.tensor(y_test, dtype=torch.long),
            torch.tensor(x_test, dtype=torch.float),
        )
        torch.save(
            obj=test_data, f=output_test_file_path, pickle_protocol=PICKLE_PROTOCOL
        )
    else:
        test_data = torch.load(output_test_file_path)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate the weights to use to account for the imbalanced corpus.
    num_positives = test_data[:][0].sum()
    num_negatives = len(test_data) - num_positives

    w = torch.tensor([num_negatives, num_positives], dtype=torch.float32)
    w = w / w.sum()
    w = 1.0 / w

    experiment_params = {
        "earliest_params": {
            "n_hidden": 256,
            "n_layers": 1,
            "lam": None,
            "num_epochs": 600,
            "input_type": "doc2vec",
            "drop_p": 0,
            "is_competition": False,
            "n_inputs": doc2vec_representation.vector_size,
        },
        "optimizer_params": {"lr": None},
    }

    for n_classes in [1, 2]:
        for lam in [0.0005, 0.0001, 0.00005, 0.00001]:
            for lr in [0.1, 0.01, 0.001]:
                experiment_params["earliest_params"]["n_classes"] = n_classes
                experiment_params["earliest_params"]["lam"] = lam
                experiment_params["optimizer_params"]["lr"] = lr

                experiment_name = f"cls_{n_classes}_lam_{lam}_lr_{lr}"
                CURR_EXP_PATH = os.path.join(PATH_EARLIEST_MODELS, experiment_name)
                os.makedirs(CURR_EXP_PATH, exist_ok=True)
                print_message(f"Writer path: {CURR_EXP_PATH}")
                writer = SummaryWriter(CURR_EXP_PATH)

                current_weights = w.clone()
                if experiment_params["earliest_params"]["n_classes"] == 1:
                    current_weights = w[1]
                current_weights.to(device)
                experiment_params["earliest_params"].update(
                    {"weights": current_weights}
                )

                model = EARLIEST(**experiment_params["earliest_params"])
                model.representation_path = DOC2VEC_REPRESENTATIONS
                model = model.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), **experiment_params["optimizer_params"]
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.99
                )
                num_epochs = experiment_params["earliest_params"]["num_epochs"]

                experiment_params["earliest_params"]["weights"] = (
                    current_weights.clone().cpu().numpy().tolist()
                )
                model_information_path = os.path.join(CURR_EXP_PATH, "model.json")
                with open(model_information_path, "w") as fp:
                    json.dump(fp=fp, obj=experiment_params, indent="\t")

                training_delays = None

                # Training.
                best_validation_loss = float("inf")
                model_state_dict_path = model_information_path.replace(
                    ".json", "_state_dict.pt"
                )
                for epoch in range(num_epochs):
                    initial_time = time.time()
                    (
                        training_delays,
                        loss_sum,
                        loss_sum_r,
                        loss_sum_b,
                        loss_sum_c,
                        loss_sum_e,
                    ) = one_epoch_train_earliest_model(
                        earliest_model=model,
                        earliest_optimizer=optimizer,
                        earliest_scheduler=scheduler,
                        loader=train_loader,
                        current_epoch=epoch,
                        device=device,
                        num_epochs=num_epochs,
                    )
                    elapsed_time = np.round(time.time() - initial_time, 2)

                    writer.add_scalar(
                        "Train Loss", np.round(loss_sum / len(train_loader), 3), epoch
                    )
                    writer.add_scalar(
                        "Train Loss RL",
                        np.round(loss_sum_r / len(train_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Train Loss Baseline",
                        np.round(loss_sum_b / len(train_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Train Loss Classific",
                        np.round(loss_sum_c / len(train_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Train Loss Earliness",
                        np.round(loss_sum_e / len(train_loader), 3),
                        epoch,
                    )
                    writer.add_scalar("Train Elapsed Time", elapsed_time, epoch)

                    initial_time = time.time()
                    (
                        validation_epoch_loss,
                        val_loss_sum_r,
                        val_loss_sum_b,
                        val_loss_sum_c,
                        val_loss_sum_e,
                    ) = validate_earliest_model(
                        earliest_model=model, loader=valid_loader, device=device
                    )
                    elapsed_time = np.round(time.time() - initial_time, 2)

                    writer.add_scalar(
                        "Valid Loss",
                        np.round(validation_epoch_loss / len(valid_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Valid Loss RL",
                        np.round(val_loss_sum_r / len(valid_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Valid Loss Baseline",
                        np.round(val_loss_sum_b / len(valid_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Valid Loss Classific",
                        np.round(val_loss_sum_c / len(valid_loader), 3),
                        epoch,
                    )
                    writer.add_scalar(
                        "Valid Loss Earliness",
                        np.round(val_loss_sum_e / len(valid_loader), 3),
                        epoch,
                    )
                    writer.add_scalar("Valid Elapsed Time", elapsed_time, epoch)

                    if epoch > 25 and validation_epoch_loss < best_validation_loss:
                        best_validation_loss = validation_epoch_loss
                        torch.save(model.state_dict(), model_state_dict_path)
                        model_save_epoch = epoch

                training_delays = training_delays.cpu().numpy()
                print_message(f"Training_delays: {training_delays}")

                # Testing.
                # Load the model with the smallest validation loss.
                print_message(f"Loading model from epoch {model_save_epoch}.")
                model.load_state_dict(torch.load(model_state_dict_path))

                (
                    testing_predictions,
                    testing_labels,
                    testing_delays,
                ) = test_earliest_model(
                    earliest_model=model, loader=test_loader, device=device
                )

                testing_predictions = torch.cat(testing_predictions).cpu().numpy()
                testing_labels = torch.cat(testing_labels).cpu().numpy()
                testing_delays = torch.cat(testing_delays).cpu().numpy()

                with open(test_corpus_information_path, "r") as fp:
                    test_corpus_information = json.load(fp=fp)
                median_num_post_users_test = test_corpus_information[
                    "median_num_post_users"
                ]

                classification_report_text = classification_report(
                    testing_labels, testing_predictions
                )
                confusion_matrix_text = confusion_matrix(
                    testing_labels, testing_predictions
                )
                accuracy = np.round(
                    accuracy_score(testing_labels, testing_predictions), 3
                )
                (
                    precision_weighted,
                    recall_weighted,
                    f1_weighted,
                    _,
                ) = precision_recall_fscore_support(
                    testing_labels, testing_predictions, average="weighted"
                )
                (
                    precision_positives,
                    recall_positives,
                    f1_positives,
                    _,
                ) = precision_recall_fscore_support(
                    testing_labels, testing_predictions, pos_label=1, average="binary"
                )

                print_message("Accuracy: {}".format(accuracy))
                print_message(classification_report_text)
                print_message(confusion_matrix_text)

                performance_measures = {
                    "classification_report": classification_report_text,
                    "confusion_matrix": confusion_matrix_text.tolist(),
                }

                print_message(f"delays: {testing_delays}")
                median_global_delay = np.median(testing_delays)
                median_positive_delay = np.median(testing_delays[testing_labels == 1])

                c_fp = sum(testing_labels) / len(testing_labels)
                erde_5 = erde(
                    labels_list=testing_predictions,
                    true_labels_list=testing_labels,
                    delay_list=testing_delays,
                    c_fp=c_fp,
                    o=5,
                )
                erde_50 = erde(
                    labels_list=testing_predictions,
                    true_labels_list=testing_labels,
                    delay_list=testing_delays,
                    c_fp=c_fp,
                    o=50,
                )
                print_message(f"erde_5: {erde_5}")
                print_message(f"erde_50: {erde_50}")
                performance_measures["erde_5"] = erde_5
                performance_measures["erde_50"] = erde_50

                print_message(
                    f"median_num_post_users_test: {median_num_post_users_test}"
                )
                p = value_p(k=median_num_post_users_test)
                f_latency_result = f_latency(
                    labels=testing_predictions,
                    true_labels=testing_labels,
                    delays=testing_delays,
                    penalty=p,
                )
                print_message(f"f_latency: {f_latency_result}")
                performance_measures["f_latency"] = f_latency_result

                performance_measures_path = model_information_path.replace(
                    ".json", "_performance.json"
                )
                with open(performance_measures_path, "w") as fp:
                    json.dump(fp=fp, obj=performance_measures, indent="\t")

                hparam_dict = {}
                hparam_dict.update(experiment_params["earliest_params"])
                hparam_dict.update(experiment_params["optimizer_params"])

                if type(hparam_dict["weights"]) == list:
                    hparam_dict["weights"] = str(hparam_dict["weights"])

                metric_dict = {
                    "median_global_delay": median_global_delay,
                    "median_positive_delay": median_positive_delay,
                    "accuracy": accuracy,
                    "precision_weighted": precision_weighted,
                    "recall_weighted": recall_weighted,
                    "f1_weighted": f1_weighted,
                    "precision_positives": precision_positives,
                    "recall_positives": recall_positives,
                    "f1_positives": f1_positives,
                    "erde_5": erde_5,
                    "erde_50": erde_50,
                    "f_latency": f_latency_result,
                }

                writer.add_hparams(hparam_dict, metric_dict)

                trained_model_path = model_information_path.replace(
                    ".json", "_trained.json"
                )
                model.save(trained_model_path)
                writer.close()

    print_message("#" * 50)
    print_message("END OF SCRIPT")
