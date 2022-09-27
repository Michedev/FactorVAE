# FactorVAE

FactorVAE implementation using _Pytorch, Pytorch Lightning, Poetry and Hydra_.
Currently, this implementation supports only the _DSprites_ dataset.


## Project structure

    ├── data  # Data folder
    ├── deep_learning_template  # source code
    │   ├── config
    │   │   ├── dataset  # Dataset config
    │   │   ├── model  # Model config
    │   │   ├── model_dataset  # model and dataset specific config
    │   │   ├── test.yaml   # testing configuration
    │   │   └── train.yaml  # training configuration
    │   ├── dataset  # Dataset definition
    │   ├── model  # Model definition
    │   ├── utils
    │   │   ├── experiment_tools.py # Iterate over experiments
    │   │   └── paths.py  # common paths
    │   ├── train.py  # Entrypoint point for training
    │   └── test.py  # Entrypoint point for testing
    ├── pyproject.toml  # Project configuration
    ├── saved_models  # where models are saved
    └── readme.md  # This file


## TODO

- Add disentanglement evaluation
- Add generation procedure
