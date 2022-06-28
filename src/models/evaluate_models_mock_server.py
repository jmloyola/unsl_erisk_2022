"""
Evaluate model using mock server.
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
import asyncio
import datetime
import faulthandler
import json
import os
import sys

import httpx

from src.config import (
    END_OF_POST_TOKEN,
    NUM_POSTS_FOR_BERT_REP,
    PATH_BEST_MODELS,
    PROJECT_BASE_PATH,
)
from src.data.make_clean_corpus import get_cleaned_post
from src.models.model import (
    EARLIEST,
    SS3,
    EarlyModel,
    HistoricStopCriterion,
    LearnedDecisionTreeStopCriterion,
    SimpleStopCriterion,
)
from src.utils.utilities import print_message

# Requests parameters.
GET_TIMEOUT_LIMIT = 920
POST_TIMEOUT_LIMIT = 920
NUMBER_RETRIES = 1
# Variable to limit the number of posts that are read and answered sequentially.
PROCESS_ALL_POST_LIMIT = 100
# Posts that are not from the beginning are only processed if their number is divisible by this variable.
SELECTED_POST_NUMBER = 5
# Default parameters used for SimpleStopCriterion.
SIMPLE_STOP_CRITERION_PARAMS = {
    "threshold": 0.7,
    "min_delay": 10,
    "max_delay": None,
}
# Default parameters used for HistoricStopCriterion.
HISTORIC_STOP_CRITERION_PARAMS = {
    "threshold": 0.7,
    "min_delay": 10,
    "history_length": NUM_POSTS_FOR_BERT_REP,
}
# Default parameters used for SS3.
SS3_PARAMS = {
    "policy_value": 2.5,
}


async def create_new_team(base_url, team_data):
    """Create a new team. If it already exists, exit from the script with error."""
    print_message("Creating a new team.")
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post("/teams/new", json=team_data)
        if response.status_code == 200:
            print_message(f"The new team stored information is: {response.json()}.")
        else:
            print_message(
                f"ERROR: The team ({team_data}) already exists in the database. Either create a new team, or "
                "delete previous database entry."
            )
            sys.exit()


def get_users_nicknames(json_data):
    """Get the users nicknames from the first call to the get writings endpoint."""
    if int(json_data[0]["number"]) != 0:
        raise Exception(
            "ERROR: The function `get_users_nicknames` should have been called the first time you asked "
            "for writings."
        )
    users_nicknames = []
    for user_data in json_data:
        users_nicknames.append(user_data["nick"])
    return users_nicknames


async def get_writings(base_url, corpus, team_token):
    """Get the current users writings."""
    print_message("Getting the current users writings.")
    request_status_code = 400
    response = None
    number_tries = 0
    while request_status_code != 200 and number_tries < NUMBER_RETRIES:
        try:
            async with httpx.AsyncClient(base_url=base_url) as client:
                response = await client.get(
                    f"/{corpus}/getwritings/{team_token}",
                    timeout=GET_TIMEOUT_LIMIT,
                )
            request_status_code = response.status_code
        except httpx.TimeoutException:
            print_message(
                f"WARNING: The request took longer than {GET_TIMEOUT_LIMIT} seconds."
            )
            request_status_code = 408
        except httpx.ConnectError:
            print_message(
                "WARNING: Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded."
            )
            request_status_code = 429

        if request_status_code != 200:
            print_message(
                f"WARNING: The request failed, trying again. Current attempt number: {number_tries + 1}."
            )
            number_tries += 1
            await asyncio.sleep(5)
    return response, request_status_code


def get_historic_data(json_data):
    """Save historic data in a file and return the list of posts in the correct order."""
    current_data = {}
    for user_data in json_data:
        nickname = user_data["nick"]
        title = user_data["title"]
        content = user_data["content"]
        post = title + " " + content
        clean_post = get_cleaned_post(post)
        current_data[nickname] = clean_post

    file_path = os.path.join(HISTORIC_DATA_PATH, f"historic_data_{args.corpus}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as fp:
            historic_data_new = json.load(fp=fp)
        for nickname, current_post in current_data.items():
            historic_data_new[nickname] = (
                historic_data_new[nickname] + END_OF_POST_TOKEN + current_post
            )
        with open(file_path, "w") as fp:
            json.dump(fp=fp, obj=historic_data_new, indent="\t")
        return [historic_data_new[nickname] for nickname in USERS_NICKNAMES]
    else:
        with open(file_path, "w") as fp:
            json.dump(fp=fp, obj=current_data, indent="\t")
        return [current_data[nickname] for nickname in USERS_NICKNAMES]


def response_to_input_data(json_response):
    """Transform json response into input data.

    Concatenate the title and content of each user, store it in a list, and
    return the list in the correct order.
    """
    current_data = {}
    for r in json_response:
        current_data[r["nick"]] = f"{r['title']}\n{r['content']}"

    return [
        current_data[nickname] if nickname in current_data else ""
        for nickname in USERS_NICKNAMES
    ]


def generate_response(decisions, scores):
    """Generate the response for the mock server."""
    response = []
    for j, nickname in enumerate(USERS_NICKNAMES):
        d = {
            "nick": nickname,
            "decision": int(decisions[j]),
            "score": float(scores[j]),
        }
        response.append(d)
    return response


def get_model_response(model, input_data, post_number):
    """Get the model response."""
    if model.__class__ in [EarlyModel, EARLIEST]:
        model.predict(documents_test=input_data, delay=post_number)
        predictions = model.predictions
        delays = model.delays
        decisions = [
            predictions[j] if delays[j] != -1 else 0 for j in range(len(delays))
        ]
        scores = model.probabilities
    elif model.__class__ == SS3:
        decisions, scores, _ = model.predict(
            documents_test=input_data, delay=post_number
        )
    else:
        raise Exception(f'Model "{model.__class__}" not implemented yet.')
    return generate_response(decisions=decisions, scores=scores)


async def post_team_responses(
    base_url,
    corpus,
    team_token,
    team_runs,
    historic_data,
    raw_data,
    current_response_number,
):
    """Post the response for all the team's runs."""
    responses = await asyncio.gather(
        *[
            post_response(
                base_url,
                corpus,
                team_token,
                i,
                historic_data,
                raw_data,
                current_response_number,
            )
            for i in range(team_runs)
        ]
    )
    # Get the status code of all the POSTs.
    responses_status_code = [r[1] for r in responses]
    return responses_status_code


async def post_response(
    base_url,
    corpus,
    team_token,
    run_id,
    historic_data,
    raw_data,
    current_response_number,
):
    """Post the current run response."""
    print_message("Posting the current run response.")
    request_status_code = 400
    response = None
    number_tries = 0
    input_data = raw_data if MODELS_LIST[run_id].__class__ == SS3 else historic_data

    if (
        (LAST_MODELS_RESPONSE[run_id] == [])
        or (current_response_number < PROCESS_ALL_POST_LIMIT)
        or ((current_response_number % SELECTED_POST_NUMBER) == 0)
    ):
        # Process input
        model_response = get_model_response(
            model=MODELS_LIST[run_id],
            input_data=input_data,
            post_number=current_response_number + 1,
        )
        LAST_MODELS_RESPONSE[run_id] = model_response
    else:
        model_response = LAST_MODELS_RESPONSE[run_id]

    while request_status_code != 200 and number_tries < NUMBER_RETRIES:
        try:
            async with httpx.AsyncClient(base_url=base_url) as client:
                response = await client.post(
                    f"/{corpus}/submit/{team_token}/{str(run_id)}",
                    json=model_response,
                    timeout=POST_TIMEOUT_LIMIT,
                )
            request_status_code = response.status_code
        except httpx.TimeoutException:
            print_message(
                f"WARNING: The request took longer than {POST_TIMEOUT_LIMIT} seconds."
            )
            request_status_code = 408
        except httpx.ConnectError:
            print_message(
                "WARNING: Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded."
            )
            request_status_code = 429

        if request_status_code != 200:
            print_message(
                f"WARNING: The request failed, trying again. Current attempt number: {number_tries + 1}."
            )
            number_tries += 1
            await asyncio.sleep(5)
    return response, request_status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to test models on the mock server."
    )
    parser.add_argument(
        "-c",
        "--corpus",
        help="eRisk task corpus name",
        choices=["depression", "gambling"],
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dmc_type",
        help="type of DMC to use",
        choices=[
            "SimpleStopCriterion",
            "LearnedDecisionTreeStopCriterion",
            "HistoricStopCriterion",
            "normalize-score-1",
            "normalize-score-2",
            "none",
        ],
        default="none",
    )
    parser.add_argument(
        "-a", "--address", help="mock server address", default="localhost"
    )
    parser.add_argument("-p", "--port", help="mock server port", type=int, default=8000)
    parser.add_argument(
        "-m",
        "--model_path",
        help="path to base models",
        action="append",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--model_type",
        help="type of models to train",
        choices=["EarlyModel", "SS3", "EARLIEST"],
        required=True,
    )
    parser.add_argument("-n", "--team_name_token", help="team name and token", type=str)
    args = parser.parse_args()

    faulthandler.enable()

    if (
        args.corpus == "depression"
        and args.dmc_type == "LearnedDecisionTreeStopCriterion"
    ):
        path_to_lists = os.path.join(
            PROJECT_BASE_PATH, "notebooks/manual_review_corpus/depression"
        )
        model_path = os.path.join(
            PATH_BEST_MODELS,
            "positive_f1/reddit/depression/selected_models/dmc_decision_tree.pkl",
        )
        with open(
            os.path.join(path_to_lists, "depression_information_gain_words.json"), "r"
        ) as fp:
            information_gain_list = json.load(fp=fp)

        with open(os.path.join(path_to_lists, "depression_chi2_words.json"), "r") as fp:
            chi2_list = json.load(fp=fp)

        # Default parameters used for LearnedDecisionTreeStopCriterion.
        LEARNED_DT_STOP_CRITERION_PARAMS = {
            "model_path": model_path,
            "information_gain_list": information_gain_list,
            "chi2_list": chi2_list,
        }

    if args.team_name_token is None:
        # Time with Year-Month-Day-Hour
        time_format = "%Y-%m-%d-%H"
        current_time = datetime.datetime.now()
        team_name = current_time.strftime(time_format)
    else:
        team_name = args.team_name_token

    HISTORIC_DATA_PATH = os.path.join(
        PROJECT_BASE_PATH, "mock_server_runs", "historic_data", args.corpus, team_name
    )
    os.makedirs(HISTORIC_DATA_PATH, exist_ok=True)

    MODELS_LIST = []
    MODELS_PATH_LIST = []
    for idx, path in enumerate(args.model_path):
        if args.model_type == "EarlyModel":
            if args.dmc_type == "SimpleStopCriterion":
                stop_criterion = SimpleStopCriterion(**SIMPLE_STOP_CRITERION_PARAMS)
            elif args.dmc_type == "LearnedDecisionTreeStopCriterion":
                if args.corpus == "gambling":
                    raise Exception(
                        "LearnedDecisionTreeStopCriterion is not implemented for gambling."
                    )
                stop_criterion = LearnedDecisionTreeStopCriterion(
                    **LEARNED_DT_STOP_CRITERION_PARAMS
                )
            elif args.dmc_type == "HistoricStopCriterion":
                stop_criterion = HistoricStopCriterion(**HISTORIC_STOP_CRITERION_PARAMS)
            else:
                stop_criterion = None
            model = EarlyModel(
                path_to_model_information=path,
                stop_criterion=stop_criterion,
                is_competition=True,
            )
        elif args.model_type == "SS3":
            if args.dmc_type == "normalize-score-1":
                normalize_score = 1
            elif args.dmc_type == "normalize-score-2":
                normalize_score = 2
            else:
                normalize_score = 0
            if not path.endswith(".ss3m"):
                raise Exception(
                    'The file path in `model_path` should have the extension ".ss3m" to load a SS3 model.'
                )
            # The model_folder_path for SS3 points two directories above the state file provided.
            model_information_folder_path = os.path.dirname(os.path.dirname(path))
            model_name = os.path.basename(path)[: -len(".ss3m")]
            model = SS3.load(
                model_folder_path=model_information_folder_path,
                state_path=None,
                model_name=model_name,
                normalize_score=normalize_score,
                **SS3_PARAMS,
            )
            # Generate the model_information.json file for the deployment.
            model_information = {
                "model_class": "SS3",
                "model_name": model.__model__.__name__,
                "model_path": None,
                "policy_value": model.__policy_value__,
                "normalize_score": model.__normalize_score__,
            }
            model_information_path = os.path.join(
                model_information_folder_path, f"model_information_{idx:02d}.json"
            )
            with open(model_information_path, "w") as fp:
                json.dump(fp=fp, obj=model_information, indent="\t")
        elif args.model_type == "EARLIEST":
            model = EARLIEST.load(path, for_competition=True)

        MODELS_LIST.append(model)

        early_model_base_path = os.path.dirname(path)
        early_model_file_name = f"{args.model_type}_{team_name}_{idx:02d}.json"
        early_model_path = os.path.join(early_model_base_path, early_model_file_name)
        MODELS_PATH_LIST.append(early_model_path)

    base_url = f"http://{args.address}:{args.port}"
    print_message(
        f"Connecting to the mock server for the task {args.corpus} at {base_url}."
    )

    team_data = {
        "name": team_name,
        "token": team_name,
        "number_runs": len(MODELS_LIST),
        "extra_info": "".join(
            [f"Model idx: {idx}\n" + m.__repr__() for idx, m in enumerate(MODELS_LIST)]
        ),
    }

    # Create a new team
    asyncio.run(create_new_team(base_url, team_data))

    last_json_response = None
    # Get the user writings and post the classification results after every post.
    # When we get the first post of every users, we have to save a list of the users' nicknames.
    get_response, status_code = asyncio.run(
        get_writings(base_url, args.corpus, team_data["token"])
    )

    if status_code != 200:
        print_message("ERROR: GET request failed. Aborting script.")
        sys.exit()

    new_json_response = get_response.json()

    # Save the last response from each model, since we are not going to process
    # all the input to speed up the process.
    LAST_MODELS_RESPONSE = [[] for _ in range(len(MODELS_LIST))]

    USERS_NICKNAMES = get_users_nicknames(json_data=new_json_response)

    initial_response_number = int(new_json_response[0]["number"])
    current_response_number = initial_response_number

    while new_json_response != last_json_response:
        print_message(f">> Post number being processed: {current_response_number + 1}.")
        historic_data = get_historic_data(json_data=new_json_response)

        responses_status_code = asyncio.run(
            post_team_responses(
                base_url=base_url,
                corpus=args.corpus,
                team_token=team_data["token"],
                team_runs=team_data["number_runs"],
                historic_data=historic_data,
                raw_data=response_to_input_data(new_json_response),
                current_response_number=current_response_number,
            )
        )
        responses_status_are_200 = [r == 200 for r in responses_status_code]
        if not all(responses_status_are_200):
            print_message(
                "ERROR: At least one of the POST requests failed. Aborting script."
            )
            sys.exit()

        last_json_response = new_json_response

        get_response, status_code = asyncio.run(
            get_writings(base_url, args.corpus, team_data["token"])
        )

        if status_code != 200:
            print_message("ERROR: GET request failed. Aborting script.")
            sys.exit()

        new_json_response = get_response.json()

        if not new_json_response:
            print_message("No more posts to process.")
            break

        assert (int(new_json_response[0]["number"]) == current_response_number + 1) or (
            new_json_response == last_json_response
        )
        current_response_number = int(new_json_response[0]["number"])

    # Save each model
    for i, model in enumerate(MODELS_LIST):
        print_message(f"Saving model to {MODELS_PATH_LIST[i]}")
        model.save(MODELS_PATH_LIST[i])

    print_message("#" * 50)
    print_message("END OF SCRIPT")
