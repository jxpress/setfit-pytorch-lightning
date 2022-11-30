import logging
import os
import pickle
from glob import glob
from weakref import proxy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import WarningCache

log = logging.getLogger(__name__)
warning_cache = WarningCache()


class SetFitModelCheckpoint(ModelCheckpoint):
    """This class is created to save the model head, if model_head is sklearn class."""

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        trainer.save_checkpoint(filepath, self.save_weights_only)
        if not trainer.model.is_torch_model_head:
            with open(os.path.splitext(filepath)[0] + "_head.pickle", "wb") as p:
                pickle.dump(trainer.model.model_head, p)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.model.is_torch_model_head:
            checkpoint_file_names = [
                os.path.splitext(filepath)[0]
                for filepath in glob(os.path.join(self.dirpath, "*.ckpt"))
            ]
            model_head_paths = glob(os.path.join(self.dirpath, "*.pickle"))
            for model_head_path in model_head_paths:
                if not model_head_path.replace("_head.pickle", "") in checkpoint_file_names:
                    os.remove(model_head_path)
