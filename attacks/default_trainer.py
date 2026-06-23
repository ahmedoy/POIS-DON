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
        return model, trained_model_stats

    def _run_training_loop(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        desc: str,
    ) -> torch.nn.Module:
        best_val_acc = -1.0

        with tempfile.TemporaryDirectory(prefix="trainer_tmp_") as tmp_dir:
            ckpt = Path(tmp_dir) / "best.pt"

            for epoch in tqdm(range(self.experiment.num_epochs), desc=desc):
                _ = self._train_epoch(
                    model, self.dataloaders.train_loader, optimizer, criterion
                )
                val_acc = self._eval_epoch(model, self.dataloaders.val_loader)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), ckpt)

            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )

        return model

    def _train_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        model.train()
        correct = total = 0

        for images, labels in loader:
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
    def _eval_epoch(self, model: torch.nn.Module, loader: DataLoader) -> float:
        """Clean accuracy — no batch transform applied."""
        model.eval()
        correct = total = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def get_model_stats(self, model: torch.nn.Module) -> TrainedModelStats:
        model.eval()

        def eval_accuracy(loader: DataLoader, mode: TransformMode) -> float:
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

        def eval_split(loader: DataLoader) -> tuple[float, float]:
            clean_acc = eval_accuracy(loader, TransformMode.CLEAN)
            # ASR only makes sense for trojaned models (poison transform/label
            # transform are None otherwise, and AttackTransform will raise).
            attack_success_rate = (
                eval_accuracy(loader, TransformMode.POISON) if self.is_trojan else 0.0
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
