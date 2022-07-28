[![DOI](https://zenodo.org/badge/480838000.svg)](https://zenodo.org/badge/latestdoi/480838000)

# unsl_erisk_2022
Repository accompanying the CLEF 2022 eRisk Workshop participation for the UNSL team (Universidad Nacional de San Luis).

## Environment set up
To set up the environment we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Once you have miniconda installed, run:
```bash
conda env create -f environment.yml
```

## Contributing
If you want to contribute to this repository, we recommend you set up `pre-commit`.
Note that the `environment.yml` already installs it, but we need to set it up in the repository:
```bash
pre-commit install
```

Now, for every new commit you make, pre-commit will fix some errors and will notify you of others that you need to resolve.

## Citation
If you use this code in a scientific publication, we would appreciate citations to the following paper:

> J. M. Loyola, H. Thompson, S. Burdisso, M. Errecalde, UNSL at eRisk 2022: Decision policies with history for early classification,  in: Working Notes of CLEF 2022 - Conference and Labs of the Evaluation Forum, Bologna, Italy, September 5-8, 2022.

## Commands to replicate experiments
First create the directories where the logs will be placed.
```bash
mkdir -p .logs/{data,features,models,competition}
```

### Create raw datasets
If you have access to the `xml` files provided by the organizers, you can create a corpus based on these.
For that, first, you'll need to decompress the `zip` file in the directory `data/raw/xml/TASK`.
Replace `TASK` with the name of the task ("depression" or "gambling" for the 2022 edition).

Once that is done, we can create the "xml" corpus from those files with:
```bash
nohup python -m src.data.make_xml_corpus --corpus gambling > .logs/data/make_gambling_xml_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_gambling_xml_corpus_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.data.make_xml_corpus --corpus depression > .logs/data/make_depression_xml_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_depression_xml_corpus_`date +"%Y_%m_%d_%H"`.err &
```

To create the raw reddit corpus, run:
```bash
nohup python -m src.data.make_reddit_corpus --corpus gambling --mode append > .logs/data/make_gambling_reddit_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_gambling_reddit_corpus_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.data.make_reddit_corpus --corpus depression --mode append > .logs/data/make_depression_reddit_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_depression_reddit_corpus_`date +"%Y_%m_%d_%H"`.err &
```

### Create clean datasets
To clean (pre-process) the raw datasets, run the following commands:
```bash
# xml corpus
nohup python -m src.data.make_clean_corpus --corpus gambling --kind xml > .logs/data/make_gambling_xml_clean_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_gambling_xml_clean_corpus_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.data.make_clean_corpus --corpus depression --kind xml > .logs/data/make_depression_xml_clean_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_depression_xml_clean_corpus_`date +"%Y_%m_%d_%H"`.err &

# reddit corpus
nohup python -m src.data.make_clean_corpus --corpus gambling --kind reddit > .logs/data/make_gambling_reddit_clean_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_gambling_reddit_clean_corpus_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.data.make_clean_corpus --corpus depression --kind reddit > .logs/data/make_depression_reddit_clean_corpus_`date +"%Y_%m_%d_%H"`.out 2> .logs/data/make_depression_reddit_clean_corpus_`date +"%Y_%m_%d_%H"`.err &
```

### Generate the documents representations
To generate the different representations (doc2vec, lda, lsa, padded_sequential, bow), you need to run the following commands:
```bash
nohup python -m src.features.build_features --corpus gambling --kind reddit --replace_old False > .logs/features/features_gambling_reddit_`date +"%Y_%m_%d_%H"`.out 2> .logs/features/features_gambling_reddit_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.features.build_features --corpus depression --kind reddit --replace_old False > .logs/features/features_depression_reddit_`date +"%Y_%m_%d_%H"`.out 2> .logs/features/features_depression_reddit_`date +"%Y_%m_%d_%H"`.err &
```

### Train base models
To train the base models, run:
```bash
nohup python -m src.models.train_model --corpus gambling > .logs/models/models_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/models_gambling_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.models.train_model --corpus depression > .logs/models/models_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/models_depression_`date +"%Y_%m_%d_%H"`.err &
```

### Re-train base models
Once the base models are trained, we can re-train the best pair `<model_type, representation_type>` changing the seed used.
For that we first need to run the notebook `02_atemporal_models_comparison_reddit_CORPUS_NAME` where `CORPUS_NAME` is the name of the corpus.
This notebook will generate a pickle file with a DataFrame containing the best models.

```bash
nohup python -m src.models.retrain_best_models --corpus gambling > .logs/models/best_models_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/best_models_gambling_`date +"%Y_%m_%d_%H"`.err &
nohup python -m src.models.retrain_best_models --corpus depression > .logs/models/best_models_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/best_models_depression_`date +"%Y_%m_%d_%H"`.err &
```

### Compare and select the best best models
Once re-trained, choose the best pair of `<model, representation>` using the notebook `03_atemporal_best_models_comparison_reddit_CORPUS_NAME` where `CORPUS_NAME` is the name of the corpus.
Then, use the notebook `04_copy_best_models_reddit_CORPUS_NAME`, where `CORPUS_NAME` is the name of the corpus, to copy the best models to the directory `selected_models`.

### Train EARLIEST models
To train the EARLIEST models, first, select the best `doc2vec` representations obtained and edit the file `config.py` in order to use them.
Then, run this commands to train the models:
```bash
nohup python -m src.models.train_earliest --corpus gambling --device auto > .logs/models/train_earliest_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/train_earliest_gambling_`date +"%Y_%m_%d_%H"`.err &

nohup python -m src.models.train_earliest --corpus depression --device auto > .logs/models/train_earliest_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/train_earliest_depression_`date +"%Y_%m_%d_%H"`.err &
```

You can use [TensorBoard](https://www.tensorflow.org/tensorboard) to track and visualize the loss of the models.

### Train SS3 models
To train the SS3 models, run the notebooks called `05_ss3_training` on the notebooks directory.
The final cell of each notebook starts a [Live Test](https://pyss3.readthedocs.io/en/latest/user_guide/visualizations.html#live-test) that allows to actively test the models.

### Train LearnedDecisionTreeStopCriterion decision policy
In order to train the model for the decision policy `LearnedDecisionTreeStopCriterion` you have to run the notebooks on the directory `notebooks/manual_review/depression`.
Note that this policy was trained only for the depression corpus since it had more training data available.
We reviewed some positive users for the training corpus provided by the organizers of eRisk.
Thus, if you want to train the policy, you'll need to flag the point (post number) in which each user starts showing depression.

### Evaluate models in the environment of the competition using the mock server
The script `src.models.evaluate_models_mock_server` allows us to evaluate the models (EarlyModel, SS3, and EARLIEST)
in an environment similar to the eRisk laboratory.
First, you'll need to start the [mock server](https://github.com/jmloyola/erisk_mock_server/), then you can run the
following commands:

```bash
# EarlyModel
nohup python -m src.models.evaluate_models_mock_server --corpus gambling --dmc_type SimpleStopCriterion --address localhost --port 9090 --model_path path/to/model --model_type EarlyModel --team_name_token earlymodel_gambling > .logs/models/earlymodel_mock_server_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/earlymodel_mock_server_gambling_`date +"%Y_%m_%d_%H"`.err &

nohup python -m src.models.evaluate_models_mock_server --corpus depression --dmc_type SimpleStopCriterion --address localhost --port 9090 --model_path path/to/model --model_type EarlyModel --team_name_token earlymodel_depression > .logs/models/earlymodel_mock_server_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/earlymodel_mock_server_depression_`date +"%Y_%m_%d_%H"`.err &

# SS3
nohup python -m src.models.evaluate_models_mock_server --corpus gambling --address localhost --port 9090 --model_path path/to/model --model_type SS3 --dmc_type normalize-score-1 --team_name_token ss3_gambling > .logs/models/ss3_normalize_score_mock_server_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/ss3_normalize_score_mock_server_gambling_`date +"%Y_%m_%d_%H"`.err &

nohup python -m src.models.evaluate_models_mock_server --corpus depression --address localhost --port 9090 --model_path path/to/model --model_type SS3 --dmc_type normalize-score-1 --team_name_token ss3_depression > .logs/models/ss3_normalize_score_mock_server_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/ss3_normalize_score_mock_server_depression_`date +"%Y_%m_%d_%H"`.err &

# EARLIEST
nohup python -m src.models.evaluate_models_mock_server --corpus gambling --address localhost --port 9090 --model_path path/to/model --model_type EARLIEST --team_name_token earliest_gambling > .logs/models/earliest_mock_server_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/earliest_mock_server_gambling_`date +"%Y_%m_%d_%H"`.err &

nohup python -m src.models.evaluate_models_mock_server --corpus depression --address localhost --port 9090 --model_path path/to/model --model_type EARLIEST --team_name_token earliest_depression > .logs/models/earliest_mock_server_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/earliest_mock_server_depression_`date +"%Y_%m_%d_%H"`.err &
```

If you want to run multiple models in the same run, you can use the `--model_path` parameter multiple times.
Note that all the models included must be of the same type.

### Deploy models
The script `src/models/deploy_models.py` can be used to re-train the selected models with all the available datasets.
This also generates a directory to store the models and the files they need to run during the laboratory.

```bash
nohup python -m src.models.deploy_models --corpus gambling --model_path path/to/model.json --model_type EarlyModel --model_index 0 --model_path path/to/model_2.json --model_type SS3 --model_index 1 --model_path path/to/model_3.json --model_type EARLIEST --model_index 2 > .logs/models/deploy_models_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/deploy_models_gambling_`date +"%Y_%m_%d_%H"`.err &

nohup python -m src.models.deploy_models --corpus depression --model_path path/to/model.json --model_type EarlyModel --model_index 0 --model_path path/to/model_2.json --model_type SS3 --model_index 1 --model_path path/to/model_3.json --model_type EARLIEST --model_index 2 > .logs/models/deploy_models_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/models/deploy_models_depression_`date +"%Y_%m_%d_%H"`.err &
```

### Communicate with the eRisk laboratory
To connect to the eRisk laboratory server and participate in the laboratory using the deployed models, run the `connection` script:
```bash
nohup python -m src.utils.connection --team_name UNSL --team_token 777 --server_task gambling --number_posts 20 > .logs/competition/erisk2022_gambling_`date +"%Y_%m_%d_%H"`.out 2> .logs/competition/erisk2022_gambling_`date +"%Y_%m_%d_%H"`.err &

nohup python -m src.utils.connection --team_name UNSL --team_token 777 --server_task depression --number_posts 20 > .logs/competition/erisk2022_depression_`date +"%Y_%m_%d_%H"`.out 2> .logs/competition/erisk2022_depression_`date +"%Y_%m_%d_%H"`.err &
```
Replace the team information with the provided to you by the organizers of eRisk.
The `number_posts` parameter allows you to run the script for a limited number of posts.
If you don't want this, you can set this to a high value.
In general, there has never been a task with more than 2500 posts, thus you can use values greater than that to process all the input.


If you want to check that everything works correctly before sending responses to the eRisk laboratory, you can use the same script but point the `server_task` to a local instance of the mock server.
Note that you will need to edit the URL used for the GET and POST requests in the `src.utils.connection` script.
You need to add the name of the task.
For example, for the gambling task instead of `f"getwritings/{team_token}"` you will need `f"gambling/getwritings/{team_token}"` and instead of `f"submit/{team_token}/{str(run_id)}"` you'll need `f"gambling/submit/{team_token}/{str(run_id)}"`.
You will also need to copy the deployed folder into `competition/models/localhost` (`cp -vr competition/models/gambling/ competition/models/localhost/`).
```bash
nohup python -m src.utils.connection --team_name UNSL --team_token 777 --server_task localhost --number_posts 20 > .logs/competition/testing_connection_script_`date +"%Y_%m_%d_%H"`.out 2> .logs/competition/testing_connection_script_`date +"%Y_%m_%d_%H"`.err &
```

You can also use the test server provided by the organizers:
```bash
nohup python -m src.utils.connection --team_name UNSL --team_token 777 --server_task unofficial_server --number_posts 20 > .logs/competition/testing_connection_script_`date +"%Y_%m_%d_%H"`.out 2> .logs/competition/testing_connection_script_`date +"%Y_%m_%d_%H"`.err &
```
