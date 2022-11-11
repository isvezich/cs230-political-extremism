# cs230-political-extremism

## Download the QAnon dataset

Create a `data/QAnon` directory and download both `Hashed_allAuthorStatus.csv` and `Hashed_Q_Submissions_Raw_Combined.csv` from https://figshare.com/articles/dataset/Datasets_for_QAnon_on_Reddit_research_project_/19251581

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