import torch
import pytorch_lightning as pl
import hydra
from disentanglement.factor_vae import disentangle
from utils.experiment_tools import load_checkpoint_model_eval
from utils.paths import ROOT
import yaml


def save_results(checkpoint_path, factor_vae_score):
    """
    Update the results of the experiment in a yaml file 
    containing the disentanglement scores, under factor_vae key
    """
    results = {'factor_vae': factor_vae_score}
    # load the yaml file
    result_file = checkpoint_path / 'disentanglement.yaml'
    if result_file.exists():
        with open(result_file, 'r') as f:
            results_dict = yaml.load(f)
    else:
        results_dict = {}
    # update the results
    results_dict.update(results)
    # save the updated results
    with open(result_file, 'w') as f:
        yaml.dump(results_dict, f)

@hydra.main(config_path="config", config_name="disentangle.yaml")
def main(config):
    model = load_checkpoint_model_eval(ROOT / config.checkpoint_path, config.seed, config.device)['model']
    model.eval()
    model.freeze()
    factor_vae_score = disentangle(model, config.rounds, config.dataset_size)

    print('=' * 150)
    print('Factor VAE disentanglement score:', factor_vae_score)
    print('=' * 150)
    save_results(ROOT / config.checkpoint_path, factor_vae_score)


if __name__ == '__main__':
    main()
