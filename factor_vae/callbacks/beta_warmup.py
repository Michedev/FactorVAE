from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class BetaWarmUp(Callback):

    def __init__(self, beta_start: float, beta_end: float, steps: int, delta: float):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps
        self.delta = delta

    def on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_step <= 1:
            print('=====================================')
            print('Set beta to', self.beta_start)
            print('=====================================')
            pl_module.beta = self.beta_start
        elif trainer.global_step % self.steps == 0:
            pl_module.beta = min(self.beta_end, pl_module.beta + self.delta)
            print('=====================================')
            print('Set beta to', pl_module.beta)
            print('=====================================')
