# FactorVAE

FactorVAE implementation using _Pytorch, Pytorch Lightning, Pipenv and Hydra_.
Currently, this implementation supports only the _DSprites_ dataset.
This implementation follows as much as possible the specifications contained in Disentangling by Factorising (Kim & Mnih, 2018) https://arxiv.org/pdf/1802.05983.pdf

![https://seqamlab.com/wp-content/uploads/2019/09/imgonline-com-ua-twotoone-CrWAJ4mw43b9N4-600x326.jpg](https://seqamlab.com/wp-content/uploads/2019/09/imgonline-com-ua-twotoone-CrWAJ4mw43b9N4-600x326.jpg)

## Install

1. Clone the repository

```bash
git clone https://github.com/Michedev/FactorVAE.git
```

2. Install [pipenv](https://pipenv.pypa.io/en/latest/)

```bash
pip install pipenv
```


3. Install the dependencies in a isolated virtualenv

```bash
pipenv install
```

## Train

To train the model, run the following command:

```bash
pipenv run python deep_learning_template/train.py
```

or alternatively, to train single GPU:


```bash
pipenv run train-gpu
 ```

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

## Generate batch of images

Once trained a model, generate the image through the script generate.py via the following command

```bash
pipenv run python factor_vae/generate.py checkpoint_path=saved_models/{model_folder}
```

Then, inside _{model_folder}_ there will be the file _generated.png_ containing the batch of generated images.

![generated](https://user-images.githubusercontent.com/12683228/193795520-e162eace-62ca-47f7-b9e2-428dbe88203e.png)


## TODO

- Add disentanglement evaluation
- ~~Add generation procedure~~
