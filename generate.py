import hydra
from omegaconf import DictConfig
import torchvision

from utils.paths import CONFIG, ROOT
from utils.experiment_tools import load_checkpoint_model_eval


@hydra.main(CONFIG, 'generate.yaml')
def main(config: DictConfig):
    ckpt = load_checkpoint_model_eval(ROOT / config.checkpoint_path, config.seed, config.device)
    model = ckpt['model']
    ckpt_folder = ckpt['ckpt_folder']
    generated_batch = model.generate(config.batch_size)
    print('generated batch', generated_batch.shape)
    img_grid = torchvision.utils.make_grid(generated_batch, nrow=config.grid_rows)
    torchvision.utils.save_image(img_grid, ckpt_folder / 'generated.png')
    print('saved', ckpt_folder / 'generated.png')


if __name__ == '__main__':
    main()
