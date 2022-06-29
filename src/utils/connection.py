"""
Communicate with the eRisk laboratory.
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
import json
import os
import sys
import time

import httpx

from src.config import END_OF_POST_TOKEN, FP_PRECISION_ELAPSED_TIMES, PATH_COMPETITION
from src.data.make_clean_corpus import get_cleaned_post
from src.models.model import EARLIEST, SS3, EarlyModel
from src.utils.utilities import print_message

# TODO: Add your team name and the number of runs of your team.
number_of_runs_team = {"UNSL": 5}
# TODO: In case any URLs change, you we'll need to change them here.
#       localhost refers to the local mock server.
server_urls_dict = {
    "unofficial_server": "https://erisk.irlab.org/challenge-service/",
    "gambling": "https://erisk.irlab.org/challenge-t1/",
    "depression": "https://erisk.irlab.org/challenge-t2/",
    "localhost": "http://localhost:9090/",
}
# Requests parameters.
GET_TIMEOUT_LIMIT = 920
POST_TIMEOUT_LIMIT = 920
# Variable to limit the number of posts that are read and answered sequentially.
PROCESS_ALL_POST_LIMIT = 100
# Posts that are not from the beginning are only processed if their number is divisible by this variable.
SELECTED_POST_NUMBER = 5


def get_users_nicknames(json_data):
    """Get the users nicknames."""
    file_path = os.path.join(SERVER_DATA_PATH, f"nicknames_{args.server_task}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as fp:
            users_nicknames = json.load(fp=fp)
    else:
        if int(json_data[0]["number"]) != 0:
            raise Exception("The file with nicknames should have been created already.")
        users_nicknames = []
        for user_data in json_data:
            users_nicknames.append(user_data["nick"])
        with open(file_path, "w") as fp:
            json.dump(fp=fp, obj=users_nicknames, indent="\t")
    return users_nicknames


def get_historic_data(json_data):
    """Concatenate all the users previous posts."""
    start_time = time.time()
    current_data = {}
    for user_data in json_data:
        nickname = user_data["nick"]
        title = user_data["title"]
        content = user_data["content"]
        post = title + " " + content
        clean_post = get_cleaned_post(post)
        current_data[nickname] = clean_post

    file_path = os.path.join(
        HISTORIC_DATA_PATH, f"historic_data_{args.server_task}.json"
    )
    if os.path.exists(file_path):
        with open(file_path, "r") as fp:
            historic_data_new = json.load(fp=fp)
        for nickname, current_post in current_data.items():
            historic_data_new[nickname] = (
                historic_data_new[nickname] + END_OF_POST_TOKEN + current_post
            )
        with open(file_path, "w") as fp:
            json.dump(fp=fp, obj=historic_data_new, indent="\t")
        return [
            historic_data_new[nickname] for nickname in USERS_NICKNAMES
        ], time.time() - start_time
    else:
        with open(file_path, "w") as fp:
            json.dump(fp=fp, obj=current_data, indent="\t")
        return [
            current_data[nickname] for nickname in USERS_NICKNAMES
        ], time.time() - start_time


def get_model(model_id):
    """Load model."""
    model_information_path = os.path.join(
        MODELS_PATH, str(model_id), "model_information.json"
    )
    if not os.path.exists(model_information_path):
        raise Exception(
            f"There is no file with the model information in '{model_information_path}'. We can not load it."
        )
    with open(model_information_path, "r") as fp:
        model_information = json.load(fp=fp)
    model_class = model_information["model_class"]
    model_path = model_information["model_path"]
    # Load the model
    if model_class == "EarlyModel":
        model = EarlyModel.load(model_path)
    elif model_class == "EARLIEST":
        model = EARLIEST.load(model_path, for_competition=True)
    elif model_class == "SS3":
        model_information_folder_path = os.path.dirname(model_information_path)
        model_path = os.path.join(model_information_folder_path, model_path)
        model = SS3.load(
            model_folder_path=model_information_folder_path,
            model_name=model_information["model_name"],
            state_path=model_path,
            policy_value=model_information["policy_value"],
            normalize_score=model_information["normalize_score"],
        )
    else:
        raise Exception(f'Function to load model "{model_class}" not implemented.')

    # Check if the model has already been run previously, to see if we have to
    # clean the internal state of the model or not. In case the model has not
    # been run for the competition, it may still have information used to build
    # it, so it is necessary to clean that information.
    model_responses_base_path = os.path.join(
        PATH_COMPETITION, "model_responses", args.server_task, str(model_id)
    )
    responses_file_name = f"{model_id}_model_{args.server_task}_{0:06d}_response.json"
    model_responses_path = os.path.join(model_responses_base_path, responses_file_name)
    if not os.path.exists(model_responses_path):
        print_message(
            "Clearing model internal state since it has the information from its training stage."
        )
        model.clear_model_state()
    return model, model_path


def response_to_input_data(json_response):
    """Format input data for SS3."""
    current_data = {}
    for r in json_response:
        current_data[r["nick"]] = "%s\n%s" % (r["title"], r["content"])

    return [
        current_data[nickname] if nickname in current_data else ""
        for nickname in USERS_NICKNAMES
    ]


def generate_response(decisions, scores):
    """Structure response for the laboratory."""
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
    """Get the model response given input data."""
    elapsed_time_features = 0
    if model.__class__ in [EarlyModel, EARLIEST]:
        _, elapsed_time_features, elapsed_time_pred = model.predict(
            documents_test=input_data, delay=post_number
        )
        predictions = model.predictions
        delays = model.delays
        decisions = [
            predictions[j] if delays[j] != -1 else 0 for j in range(len(delays))
        ]
        scores = model.probabilities
    elif model.__class__ == SS3:
        decisions, scores, elapsed_time_pred = model.predict(
            documents_test=input_data, delay=post_number
        )
    else:
        raise Exception(f'Model "{model.__class__}" not implemented yet.')
    return (
        generate_response(decisions=decisions, scores=scores),
        elapsed_time_features,
        elapsed_time_pred,
    )


async def get_writings(base_url, team_token):
    """Get the current users writings."""
    print_message("Getting the current users writings.")
    request_status_code = 400
    response = None
    # We don't do multiple retries. That generated problems in the server side.
    try:
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.get(
                f"getwritings/{team_token}",
                timeout=GET_TIMEOUT_LIMIT,
            )
        request_status_code = response.status_code
    except httpx.TimeoutException:
        print_message(
            f"WARNING: The request took longer than {GET_TIMEOUT_LIMIT} seconds."
        )
        request_status_code = 408
    except httpx.ConnectError:
        print_message("WARNING: Connection Error.")
        request_status_code = 429

    if request_status_code != 200:
        print_message("WARNING: The request failed.")
    return response, request_status_code


async def post_team_responses(
    base_url, team_token, team_runs, historic_data, raw_data, current_response_number
):
    """Post the response for all the team's runs."""
    responses = await asyncio.gather(
        *[
            post_response(
                base_url,
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
    base_url, team_token, run_id, historic_data, raw_data, current_response_number
):
    """Post the current run response."""
    print_message(f"Posting the response from run {run_id}.")
    request_status_code = 400
    response = None
    elapsed_time_features = 0
    elapsed_time_pred = 0
    input_data = raw_data if MODELS[run_id].__class__ == SS3 else historic_data

    if (
        (LAST_MODELS_RESPONSE[run_id] == [])
        or (current_response_number < PROCESS_ALL_POST_LIMIT)
        or ((current_response_number % SELECTED_POST_NUMBER) == 0)
    ):
        # Process input
        model_response, elapsed_time_features, elapsed_time_pred = get_model_response(
            model=MODELS[run_id],
            input_data=input_data,
            post_number=current_response_number + 1,
        )
        FEATURES_DELAYS[run_id].append(
            round(elapsed_time_features, FP_PRECISION_ELAPSED_TIMES)
        )
        PREDICTION_DELAYS[run_id].append(
            round(elapsed_time_pred, FP_PRECISION_ELAPSED_TIMES)
        )
        LAST_MODELS_RESPONSE[run_id] = model_response
    else:
        FEATURES_DELAYS[run_id].append(
            round(elapsed_time_features, FP_PRECISION_ELAPSED_TIMES)
        )
        PREDICTION_DELAYS[run_id].append(
            round(elapsed_time_pred, FP_PRECISION_ELAPSED_TIMES)
        )
        model_response = LAST_MODELS_RESPONSE[run_id]

    try:
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.post(
                f"submit/{team_token}/{str(run_id)}",
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
        print_message("WARNING: Connection Error.")
        request_status_code = 429

    if request_status_code != 200:
        print_message("WARNING: The request failed.")

    # If the POST was successful, save the model response
    responses_path = os.path.join(
        PATH_COMPETITION,
        "model_responses",
        args.server_task,
        str(run_id),
        f"{run_id}_model_{args.server_task}_{current_response_number:06d}_response.json",
    )
    with open(responses_path, "w") as f:
        json.dump(fp=f, obj=model_response, indent="\t")

    MODELS[run_id].save(PATH_TO_CURRENT_MODELS[run_id])

    return response, request_status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to connect to the eRisk server to get\
        writings and to send the responses."
    )
    parser.add_argument("-n", "--team_name", help="Team name as register in CLEF")
    parser.add_argument(
        "-t", "--team_token", help="Team token provided by the eRisk's organizers"
    )
    parser.add_argument(
        "-s",
        "--server_task",
        help="Server name for the task to solve",
        choices=["unofficial_server", "gambling", "depression", "localhost"],
        default="localhost",
    )
    parser.add_argument(
        "-p",
        "--number_posts",
        help="Number of post you want to process before stopping the script",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    if args.team_token is None or args.team_name is None:
        print_message("You should specify all the options to run the script")
        sys.exit()

    if args.team_name not in number_of_runs_team:
        print_message(
            f'The team "{args.team_name}" is not included in the number_of_runs_team dictionary.'
            f" Add it manually to the script."
        )

    print_message(
        f"Connecting to the server {args.server_task} with the token for the team {args.team_name}"
    )

    last_json_response = None
    # Get the user writings and post the classification results after every post.
    # When we get the first post of every users, we have to save a list of the users.
    get_response, status_code = asyncio.run(
        get_writings(
            base_url=server_urls_dict[args.server_task], team_token=args.team_token
        )
    )

    if status_code != 200:
        print_message("GET request failed. Aborting script...")
        sys.exit()

    new_json_response = get_response.json()

    SERVER_DATA_PATH = os.path.join(
        PATH_COMPETITION, "data", "server_data", args.server_task
    )
    HISTORIC_DATA_PATH = os.path.join(
        PATH_COMPETITION, "data", "historic_data", args.server_task
    )
    MODELS_PATH = os.path.join(PATH_COMPETITION, "models", args.server_task)
    ELAPSED_TIMES_FILE_PATH = os.path.join(
        PATH_COMPETITION, "elapsed_times", args.server_task, "elapsed_times.json"
    )
    os.makedirs(SERVER_DATA_PATH, exist_ok=True)
    os.makedirs(HISTORIC_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(ELAPSED_TIMES_FILE_PATH), exist_ok=True)

    LAST_MODELS_RESPONSE = [[] for _ in range(number_of_runs_team[args.team_name])]

    # Check if there are delays already saved, if that is not the case
    # initialize the lists.
    if os.path.exists(ELAPSED_TIMES_FILE_PATH):
        with open(ELAPSED_TIMES_FILE_PATH, "r") as fp:
            elapsed_time_dict = json.load(fp=fp)
            HISTORIC_DATA_DELAYS = elapsed_time_dict["historic_data_delays"]
            FEATURES_DELAYS = elapsed_time_dict["features_delays"]
            PREDICTION_DELAYS = elapsed_time_dict["predictions_delays"]
    else:
        HISTORIC_DATA_DELAYS = []
        FEATURES_DELAYS = [[] for i in range(number_of_runs_team[args.team_name])]
        PREDICTION_DELAYS = [[] for i in range(number_of_runs_team[args.team_name])]

    USERS_NICKNAMES = get_users_nicknames(json_data=new_json_response)
    MODELS = [None for i in range(number_of_runs_team[args.team_name])]
    PATH_TO_CURRENT_MODELS = [None for i in range(number_of_runs_team[args.team_name])]
    for i in range(number_of_runs_team[args.team_name]):
        responses_base_path = os.path.join(
            PATH_COMPETITION, "model_responses", args.server_task, str(i)
        )
        os.makedirs(responses_base_path, exist_ok=True)

        MODELS[i], PATH_TO_CURRENT_MODELS[i] = get_model(model_id=i)

    initial_response_number = int(new_json_response[0]["number"])
    current_response_number = initial_response_number
    while (
        new_json_response != last_json_response
        and (current_response_number - initial_response_number) < args.number_posts
    ):
        # We save the json data
        json_path = os.path.join(
            SERVER_DATA_PATH,
            f"posts_{current_response_number:06d}_{args.server_task}.json",
        )
        with open(json_path, "w") as f:
            json.dump(fp=f, obj=new_json_response, indent="\t")

        print_message(f">> Post number being processed: {current_response_number}")

        historic_data, historic_data_time_spend = get_historic_data(
            json_data=new_json_response
        )
        HISTORIC_DATA_DELAYS.append(
            round(historic_data_time_spend, FP_PRECISION_ELAPSED_TIMES)
        )
        responses_status_code = asyncio.run(
            post_team_responses(
                base_url=server_urls_dict[args.server_task],
                team_token=args.team_token,
                team_runs=number_of_runs_team[args.team_name],
                historic_data=historic_data,
                raw_data=response_to_input_data(new_json_response),
                current_response_number=current_response_number,
            )
        )
        responses_status_are_200 = [r == 200 for r in responses_status_code]
        if not all(responses_status_are_200):
            print_message(
                "ERROR: At least one of the POSTs requests failed. Aborting script."
            )
            sys.exit()

        # If all the POSTs were successful, save the delays.
        with open(ELAPSED_TIMES_FILE_PATH, "w") as fp:
            elapsed_time_dict = {
                "historic_data_delays": HISTORIC_DATA_DELAYS,
                "features_delays": FEATURES_DELAYS,
                "predictions_delays": PREDICTION_DELAYS,
            }
            json.dump(fp=fp, obj=elapsed_time_dict, indent="\t")

        last_json_response = new_json_response

        get_response, status_code = asyncio.run(
            get_writings(
                base_url=server_urls_dict[args.server_task], team_token=args.team_token
            )
        )

        if status_code != 200:
            print_message("GET request failed. Aborting script...")
            sys.exit()

        new_json_response = get_response.json()

        if not new_json_response:
            print_message("No more posts to process")
            break

        assert (int(new_json_response[0]["number"]) == current_response_number + 1) or (
            new_json_response == last_json_response
        )
        current_response_number = int(new_json_response[0]["number"])

    if (
        new_json_response
        and new_json_response != last_json_response
        and (current_response_number - initial_response_number) == args.number_posts
    ):
        print_message(f"Reached the number of posts limit ({args.number_posts} posts)")

    print_message("#" * 50)
    print_message("END OF SCRIPT")
