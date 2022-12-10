# cs230-political-extremism

## Download the QAnon dataset

Create a `data` directory and download `non-q-posts-v2.csv.gz`, `q-posts-v2.csv.gz`, and `bert.csv.gz` from https://cs230-reddit.s3.us-west-1.amazonaws.com/non-q-posts-v2.csv.gz, https://cs230-reddit.s3.us-west-1.amazonaws.com/q-posts-v2.csv.gz, https://cs230-reddit.s3.us-west-1.amazonaws.com/bert.csv.gz

## Quickstart

A `base_model` directory under the `experiments` directory contains a file `params.json` which sets the parameters for the run. It looks like

```json
{
  "learning_rate": 1e-3,
  "batch_size": 5,
  "num_epochs": 2
}
```

To **Train** a model, simply run

```
python train.py
```

Optionally `--model_dir` and `--data_dir` can be used to specify a different model (besides baseline) and data dir, respectively.

For tuning the hyperparameters, here are the supported arguments that would override the param json:

```angular2html
--h1_units, --h2_units, --l2_reg_lambda, --learning_rate, --batch_size, --num_epochs, --dropout_rate, --early_stopping_patience, --sample_rate, --sentences_length, --max_features, --embedding_size --word_embeddings
```
