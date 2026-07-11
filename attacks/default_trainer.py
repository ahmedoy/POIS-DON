import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from schema.attack_experiment import AttackExperiment, AttackDataLoaders
from schema.artifacts import TrainedModelStats
from attacks.model_saver import ModelSaver
from attacks.attack_dataloading import DefaultImageTorchDataset
from attacks.base_transform import TransformMode


@dataclass(frozen=True)
class SplitConfig:
    name: str
    num_models: int


def create_and_save_models(
    model_factory: Callable[[], torch.nn.Module],
    dataloaders: AttackDataLoaders,
    experiment: AttackExperiment,
    is_trojan: bool,
) -> None:
    """Entrypoint to DefaultTrainer."""
    DefaultTrainer(model_factory, dataloaders, experiment, is_trojan).run()


class DefaultTrainer:
    """Trains (and, eventually, saves) a set of models across train/val/test splits."""

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        dataloaders: AttackDataLoaders,
        experiment: AttackExperiment,
        is_trojan: bool,
    ) -> None:
        self.model_factory = model_factory
        self.dataloaders = dataloaders
        self.experiment = experiment
        self.is_trojan = is_trojan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_saver = ModelSaver()

    def run(self) -> None:
        print(f"Using device: {self.device}")

        splits = [
            SplitConfig("train", self.experiment.number_of_models_train),
            SplitConfig("val", self.experiment.number_of_models_val),
            SplitConfig("test", self.experiment.number_of_models_test),
        ]
        for split in splits:
            self._create_split_models(split)

    # ------------------------------------------------------------------
    # Internals — not intended to be overridden
    # ------------------------------------------------------------------

    def _create_split_models(self, split: SplitConfig) -> None:
        for i in range(split.num_models):
            model, stats = self._train_one_model(
                desc=f"{split.name} model {i + 1}/{split.num_models}"
            )

            # AttackTransform at Eval Time
            test_dataset = cast(
                DefaultImageTorchDataset, self.dataloaders.test_loader.dataset
            )
            train_dataset = cast(
                DefaultImageTorchDataset, self.dataloaders.train_loader.dataset
            )

            # Obtain poison rate and clean eval transform which are required metadata when saving the model
            clean_eval_transform = test_dataset.transform.clean_image_transform
            poison_rate = train_dataset.transform.poison_rate

            self.model_saver.save_model(
                model=model,
                trained_model_stats=stats,
                attack_experiment=self.experiment,
                clean_eval_transform=clean_eval_transform,
                split=split.name,
                is_trojan=self.is_trojan,
                poison_rate=poison_rate,
                metadata_dict={},
            )

    def _train_one_model(
        self, desc: str = "Training"
    ) -> tuple[torch.nn.Module, TrainedModelStats]:
        model = self.model_factory().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.experiment.learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()

        model = self._run_training_loop(model, optimizer, criterion, desc)
        trained_model_stats = self.get_model_stats(model)
        print(trained_model_stats)
        return model, trained_model_stats

    def _run_training_loop(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        desc: str,
    ) -> torch.nn.Module:

        # Will be Validation Accuracy or HarmonicMean(Validation Accuracy, Attack Success Rate) depending on is_trojan
        best_metric = -1.0

        with tempfile.TemporaryDirectory(prefix="trainer_tmp_") as tmp_dir:
            ckpt = Path(tmp_dir) / "best.pt"

            for epoch in tqdm(range(self.experiment.num_epochs), desc=desc):
                _ = self._train_epoch(
                    model, self.dataloaders.train_loader, optimizer, criterion
                )
                metric = self._checkpoint_selection_metric(
                    model, self.dataloaders.val_loader
                )

                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), ckpt)

            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )

        return model

    def _train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        model.train()
        dataset = cast(DefaultImageTorchDataset, train_loader.dataset)
        dataset.set_transform_mode(TransformMode.DEFAULT)
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def _eval_accuracy(
        self, model: torch.nn.Module, loader: DataLoader, mode: TransformMode
    ) -> float:
        """Evaluate accuracy on `loader` under a given transform mode.

        Temporarily switches the loader's dataset into `mode` (e.g. CLEAN to
        measure clean accuracy, POISON to measure attack success rate) and
        restores the dataset's original mode afterward, regardless of how
        this function exits.
        """
        model.eval()
        dataset = cast(DefaultImageTorchDataset, loader.dataset)
        original_mode = dataset.transform.transform_mode
        dataset.set_transform_mode(mode)
        try:
            correct = total = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
            return correct / total if total > 0 else 0.0
        finally:
            dataset.set_transform_mode(original_mode)

    def _checkpoint_selection_metric(
        self, model: torch.nn.Module, val_loader: DataLoader
    ) -> float:
        """Metric used to decide whether an epoch's weights become the new checkpoint.

        For benign models this is just clean validation accuracy. For trojaned
        models, selecting on clean accuracy alone can pick an epoch where ASR
        collapses (or vice versa), so we use the harmonic mean of validation
        clean accuracy and validation ASR — it only scores high when both are
        high, which keeps a checkpoint from "winning" by sacrificing one for
        the other.
        """
        val_acc = self._eval_accuracy(model, val_loader, TransformMode.CLEAN)

        if not self.is_trojan:
            return val_acc

        val_asr = self._eval_accuracy(model, val_loader, TransformMode.POISON)
        return self._harmonic_mean(val_acc, val_asr)

    @staticmethod
    def _harmonic_mean(a: float, b: float) -> float:
        if a + b == 0:
            return 0.0
        return 2 * a * b / (a + b)

    @torch.no_grad()
    def get_model_stats(self, model: torch.nn.Module) -> TrainedModelStats:
        model.eval()

        def eval_split(loader: DataLoader) -> tuple[float, float]:
            clean_acc = self._eval_accuracy(model, loader, TransformMode.CLEAN)
            # ASR only makes sense for trojaned models (poison transform/label
            # transform are None otherwise, and AttackTransform will raise).
            attack_success_rate = (
                self._eval_accuracy(model, loader, TransformMode.POISON)
                if self.is_trojan
                else 0.0
            )
            return clean_acc, attack_success_rate

        train_clean_acc, train_asr = eval_split(self.dataloaders.train_loader)
        val_clean_acc, val_asr = eval_split(self.dataloaders.val_loader)
        test_clean_acc, test_asr = eval_split(self.dataloaders.test_loader)

        return TrainedModelStats(
            train_clean_accuracy=train_clean_acc,
            train_attack_success_rate=train_asr,
            validation_clean_accuracy=val_clean_acc,
            validation_attack_success_rate=val_asr,
            test_clean_accuracy=test_clean_acc,
            test_attack_success_rate=test_asr,
        )
