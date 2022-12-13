import os
from typing import Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
from datasets import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from setfit import sample_dataset
from setfit.data import SetFitDataset
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_id: str = "sst2",
        max_input_length: Optional[int] = None,
        num_samples: int = 16,
        batch_size: int = 16,
        pin_memory: bool = True,
        num_workers: int = os.cpu_count(),
        column_mapping: Optional[Dict] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False, ignore="local_data_path")

        dataset = load_dataset(self.hparams.dataset_id)
        if self.hparams.column_mapping:
            for key in dataset:
                dataset[key] = self._apply_column_mapping(
                    dataset[key], self.hparams.column_mapping
                )
        self.train_dataset = sample_dataset(
            dataset["train"], label_column="label", num_samples=self.hparams.num_samples
        )
        # valid and test data is sepalated from original validation data
        self.valid_dataset, self.test_dataset = self._separate_dataset(dataset["validation"])

        # for error of tokenizer (https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _apply_column_mapping(
        self, dataset: "Dataset", column_mapping: Dict[str, str]
    ) -> "Dataset":
        """Applies the provided column mapping to the dataset, renaming columns accordingly.

        Extra features not in the column mapping are prefixed with `"feat_"`.
        """
        dataset = dataset.rename_columns(
            {
                **column_mapping,
                **{
                    col: f"feat_{col}" for col in dataset.column_names if col not in column_mapping
                },
            }
        )
        dset_format = dataset.format
        dataset = dataset.with_format(
            type=dset_format["type"],
            columns=dataset.column_names,
            output_all_columns=dset_format["output_all_columns"],
            **dset_format["format_kwargs"],
        )
        return dataset

    def _separate_dataset(self, dataset: "Dataset") -> Tuple["Dataset", "Dataset"]:
        df = pd.DataFrame(dataset)
        val_df = df.iloc[: len(df) // 2].reset_index(drop=True)
        test_df = df.iloc[len(df) // 2 :].reset_index(drop=True)

        val_dataset = Dataset(pa.Table.from_pandas(val_df))
        test_dataset = Dataset(pa.Table.from_pandas(test_df))
        return val_dataset, test_dataset

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.setfit_train_dataset: Dataset = SetFitDataset(
                self.train_dataset["text"],
                self.train_dataset["label"],
                tokenizer=self.trainer.model.model_body.tokenizer,
                max_length=self.hparams.max_input_length
                if self.hparams.max_input_length
                else self.trainer.model.model_body.get_max_seq_length(),
            )
            self.setfit_valid_dataset: Dataset = SetFitDataset(
                self.valid_dataset["text"],
                self.valid_dataset["label"],
                tokenizer=self.trainer.model.model_body.tokenizer,
                max_length=self.hparams.max_input_length
                if self.hparams.max_input_length
                else self.trainer.model.model_body.get_max_seq_length(),
            )

        if stage == "test" or stage is None:
            # test_dataset is composed of valid_data since test data of sst2 contains unlabel data
            self.setfit_test_dataset: Dataset = SetFitDataset(
                self.test_dataset["text"],
                self.test_dataset["label"],
                tokenizer=self.trainer.model.model_body.tokenizer,
                max_length=self.hparams.max_input_length
                if self.hparams.max_input_length
                else self.trainer.model.model_body.get_max_seq_length(),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.setfit_train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=SetFitDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.setfit_valid_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=SetFitDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.setfit_test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=SetFitDataset.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
