import copy
import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import (
    BatchHardTripletLossDistanceFunction,
)
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MaxMetric

from logging import getLogger
from setfit import SetFitModel, SetFitTrainer
from setfit.modeling import (
    SupConLoss,
    sentence_pairs_generation,
    sentence_pairs_generation_multilabel,
)

if TYPE_CHECKING:
    from datasets import Dataset

    from setfit import SetFitModel


logger = getLogger(__name__)
logger.info('message')


class SetfitPLModule(LightningModule, SetFitTrainer):
    """Example of LightningModule for LiveDoor text classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_id: str,
        model_init: Callable[[], "SetFitModel"] = None,
        metric: Union[
            str, Callable[["Dataset", "Dataset"], Dict[str, float]]
        ] = "accuracy",
        loss_class=losses.CosineSimilarityLoss,
        num_iterations: int = 20,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        seed: int = 42,
        column_mapping: Dict[str, str] = None,
        use_amp: bool = False,
        warmup_proportion: float = 0.1,
        distance_metric: Callable = BatchHardTripletLossDistanceFunction.cosine_distance,
        margin: float = 0.25,
        body_learning_rate: Optional[float] = None,
        l2_weight: float = 1e-2,
        show_progress_bar: bool = True,
        train_sentence_transformers_once: bool = False,
        train_only_model_head: bool = True,
        **kwargs,
    ):
        # LightningModule.__init__(self)
        super().__init__()
        self.model = SetFitModel.from_pretrained(model_id, **kwargs)
        SetFitTrainer.__init__(  # TODO If **kwargs are introduced to SetFitTrainer, omit these inputs
            self,
            model=self.model,
            model_init=model_init,
            metric=metric,
            loss_class=loss_class,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            column_mapping=column_mapping,
            use_amp=use_amp,
            warmup_proportion=warmup_proportion,
            distance_metric=distance_metric,
            margin=margin,
            num_epochs=1,
        )

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="local_data_path")

        self.model_body = self.model.model_body
        self.model_head = self.model.model_head

        self.is_torch_model_head = isinstance(self.model_head, nn.Module)

        self.model_body_prev_state = copy.deepcopy(self.model_body.state_dict())
        self.model_head_original = (
            copy.deepcopy(self.model_head)
            if not self.is_torch_model_head
            else copy.deepcopy(self.model_head.state_dict())
        )

        self.criterion = (
            self.model.model_head.get_loss_fn()
            if self.is_torch_model_head
            else nn.CrossEntropyLoss()
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.test_acc = Accuracy()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(
        self,
        x_test: Union[List[str], torch.Tensor],
    ):
        return self.model.predict_proba(x_test)

    def predict(
        self,
        x_test: Union[List[str], torch.Tensor],
    ):
        return self.model.predict(x_test)

    def predict_proba(
        self,
        x_test: Union[List[str], torch.Tensor],
    ):
        return self.model.predict_proba(x_test)

    def on_train_start(self):
        self.val_acc_best.reset()

    def on_train_epoch_start(self):
        """
        train only sentence_transformers
        """
        if self.is_torch_model_head:
            self.model.unfreeze()

            # reset parameters of model
            self.model_body.load_state_dict(self.model_body_prev_state)
            self.model_head.load_state_dict(self.model_head_original)
        else:
            self.model_head = copy.deepcopy(self.model_head_original)

        # Initialize data store for training model_head at on_validation_start
        self.train_embeddings = []
        self.train_targets = []

        if not self.hparams.train_sentence_transformers_once or (
            self.hparams.train_sentence_transformers_once and self.current_epoch == 0
        ):
            dataset = self.trainer.train_dataloader.dataset.datasets
            if dataset is None:
                raise ValueError("SetFitTrainer: training requires a train_dataset.")

            x_train = dataset.x
            y_train = dataset.y
            if self.loss_class is None:
                return
            batch_size = self.batch_size
            learning_rate = self.learning_rate
            is_differentiable_head = isinstance(
                self.model.model_head, torch.nn.Module
            )  # If False, assume using sklearn

            if not is_differentiable_head or self._freeze:
                # sentence-transformers adaptation
                if self.loss_class in [
                    losses.BatchAllTripletLoss,
                    losses.BatchHardTripletLoss,
                    losses.BatchSemiHardTripletLoss,
                    losses.BatchHardSoftMarginTripletLoss,
                    SupConLoss,
                ]:
                    train_examples = [
                        InputExample(texts=[text], label=label)
                        for text, label in zip(x_train, y_train)
                    ]
                    train_data_sampler = SentenceLabelDataset(train_examples)

                    batch_size = min(batch_size, len(train_data_sampler))
                    train_dataloader = DataLoader(
                        train_data_sampler, batch_size=batch_size, drop_last=True
                    )

                    if self.loss_class is losses.BatchHardSoftMarginTripletLoss:
                        train_loss = self.loss_class(
                            model=self.model.model_body,
                            distance_metric=self.distance_metric,
                        )
                    elif self.loss_class is SupConLoss:
                        train_loss = self.loss_class(model=self.model.model_body)
                    else:
                        train_loss = self.loss_class(
                            model=self.model.model_body,
                            distance_metric=self.distance_metric,
                            margin=self.margin,
                        )

                    train_steps = len(train_dataloader) * self.num_epochs
                else:
                    train_examples = []

                    for _ in range(self.num_iterations):
                        if self.model.multi_target_strategy is not None:
                            train_examples = sentence_pairs_generation_multilabel(
                                np.array(x_train), np.array(y_train), train_examples
                            )
                        else:
                            train_examples = sentence_pairs_generation(
                                np.array(x_train), np.array(y_train), train_examples
                            )

                    train_dataloader = DataLoader(
                        train_examples, shuffle=True, batch_size=batch_size
                    )
                    train_loss = self.loss_class(self.model_body)
                    train_steps = len(train_dataloader) * self.num_epochs

                logger.info("***** Running training *****")
                logger.info(f"  Num examples = {len(train_examples)}")
                logger.info(f"  Num epochs = {self.num_epochs}")
                logger.info(f"  Total optimization steps = {train_steps}")
                logger.info(f"  Total train batch size = {batch_size}")

                warmup_steps = math.ceil(train_steps * self.warmup_proportion)
                self.model_body.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,
                    steps_per_epoch=train_steps,
                    optimizer_params={"lr": learning_rate},
                    warmup_steps=warmup_steps,
                    show_progress_bar=True,
                    use_amp=self.use_amp,
                )

        self.model_body_prev_state = copy.deepcopy(self.model_body.state_dict())

    def shered_step(self, batch: Any, is_training=False):
        """
        Common code for training step, validation_step, and test_step

        If model_head is torch module, backward
        if model_head is sklearn module, only input data for model_head is stored in self.embeddings and self.targets
        Model_head is trained by using thses data in on_validation_start, since on_validation_start function is executed just after last training epoch
        """
        if self.is_torch_model_head:
            features, labels = batch
            outputs = self.model_body(features)
            outputs = self.model_head(outputs)
            prediction = outputs["prediction"]

            preds = torch.argmax(prediction, dim=1)
            scores = prediction.softmax(dim=1)

            loss = self.criterion(prediction, labels)
            if is_training:
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
            # log train metrics
            return {
                "loss": loss,
                "preds": preds,
                "scores": scores,
                "targets": labels,
                "embeddings": None,
            }
        else:
            features, labels = batch
            embeddings = self.model_body(features)["sentence_embedding"]
            if is_training:
                # for adding global_step count. If model_head is nn.Module, opt.step increases the number of global_step count
                self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.total.completed += (
                    1
                )

                self.train_embeddings.append(embeddings)
                self.train_targets.append(labels.detach().cpu())
                preds = None
                scores = None
            else:
                scores = self.model_head.predict_proba(embeddings.detach().cpu())
                preds = self.model_head.predict(embeddings.detach().cpu())
            return {
                "loss": None,
                "preds": preds,
                "scores": scores,
                "targets": labels,
                "embeddings": embeddings,
            }

    def _convert_data_step_end(self, outputs: List[Any]):
        res = {}
        for key in outputs[0].keys():
            res[key] = torch.cat([output[key] for output in outputs])

        return res

    def training_step(self, batch: Any, batch_idx: int):
        res = self.shered_step(batch, is_training=True)
        if self.is_torch_model_head:
            acc = self.train_acc(res["preds"], res["targets"])
            self.log(
                "train/loss", res["loss"], on_step=False, on_epoch=True, prog_bar=True
            )
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "embeddings": res["embeddings"],
            "targets": res["targets"].detach().cpu(),
        }

    def on_validation_start(self):
        """
        train the model_head if model_head is sklearn module.
        this function is called just after last training step
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        if not self.is_torch_model_head:
            embeddings = torch.cat([output for output in self.train_embeddings])
            targets = torch.cat([output for output in self.train_targets])
            self.model_head.fit(embeddings.detach().cpu(), targets.detach().cpu())

    def validation_step(self, batch: Any, batch_idx: int):
        res = self.shered_step(batch)
        if self.is_torch_model_head:
            acc = self.val_acc(res["preds"], res["targets"])
            self.log(
                "val/loss", res["loss"], on_step=False, on_epoch=True, prog_bar=True
            )
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            acc = self.val_acc(
                torch.tensor(res["preds"]), res["targets"].detach().cpu()
            )
            loss = self.criterion(
                torch.tensor(res["scores"]),
                res["targets"].detach().cpu().to(torch.long),
            )

            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "embeddings": res["embeddings"],
            "targets": res["targets"].detach().cpu(),
        }

    def validation_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        # reset metrics at the end of every epoch
        self.val_acc.reset()
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):

        res = self.shered_step(batch)
        if self.is_torch_model_head:
            acc = self.test_acc(res["preds"], res["targets"])
            self.log(
                "test/loss", res["loss"], on_step=False, on_epoch=True, prog_bar=True
            )
            self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            targets = res["targets"].detach().cpu()
            preds = res["preds"].detach().cpu()
        else:
            acc = self.test_acc(
                torch.tensor(res["preds"]), res["targets"].detach().cpu()
            )
            loss = self.criterion(
                torch.tensor(res["scores"]),
                res["targets"].detach().cpu().to(torch.long),
            )
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            targets = res["targets"].detach().cpu()
            preds = res["preds"]
        return {"embeddings": res["embeddings"], "targets": targets, "preds": preds}

    def test_epoch_end(self, outputs: List[Any]):

        if self.is_torch_model_head:
            prediction = torch.cat([output["preds"] for output in outputs]).numpy()
        else:
            prediction = np.concatenate([output["preds"] for output in outputs])
        texts = self.trainer.test_dataloaders[0].dataset.x
        target = self.trainer.test_dataloaders[0].dataset.y

        res_df = pd.DataFrame(
            {
                "texts": texts,
                "target": target,
                "prediction": prediction,
            }
        )

        if hasattr(self.logger, "_save_dir"):
            res_df.to_csv(
                os.path.join(self.logger._save_dir, "result.csv"), index=False
            )

    def predict_step(self, batch: Any, batch_idx: int):
        pass

    def _prepare_optimizer(
        self,
        learning_rate: float,
        body_learning_rate: Optional[float],
        l2_weight: float,
    ) -> torch.optim.Optimizer:
        body_learning_rate = body_learning_rate or learning_rate
        l2_weight = l2_weight or self.l2_weight

        params = [
            {
                "params": self.model.model_body.parameters(),
                "lr": body_learning_rate,
                "weight_decay": l2_weight,
            }
        ]

        if self.is_torch_model_head:
            params.append(
                {
                    "params": self.model.model_head.parameters(),
                    "lr": learning_rate,
                    "weight_decay": l2_weight,
                }
            )
        optimizer = torch.optim.AdamW(params)

        return optimizer

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = self._prepare_optimizer(
            self.hparams.learning_rate,
            self.hparams.body_learning_rate,
            self.hparams.l2_weight,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
