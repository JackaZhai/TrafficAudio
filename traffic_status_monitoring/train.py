"""Training pipeline for multi-class traffic status monitoring.

This module implements an Audio Spectrogram Transformer (AST) based training
pipeline that turns raw audio recordings into log-Mel spectrograms and
optimises a Transformer encoder to recognise the road status classes (例如
畅通、轻度拥堵、严重拥堵)。

The script only depends on PyTorch/Torchaudio together with the AST reference
implementation that ships with this repository. Two dataset layouts are
supported:

``ImageFolder`` style
    ``train``/``val``/``test`` folders with one sub-directory per class.  This
    matches the layout produced by manual curation or smaller custom datasets.

`MELAUDIS <https://github.com/parineh/MELAUDIS>`_ extracts
    A single root directory that contains all WAV files (optionally nested in
    site sub-folders).  File names encode the traffic status tokens such as
    ``_FF_`` (Free Flow), ``_SF_`` (Slow Flow / light congestion) or ``_TJ_``
    (Traffic Jam / heavy congestion).  When this layout is detected the script
    automatically builds stratified train/validation/test splits.

The pipeline performs the following steps:

* scan the dataset, build the label map and (if required) create splits;
* compute log-Mel spectrograms on the fly with configurable parameters;
* augment the training data with random time shifts and Gaussian noise;
* fine-tune an AST classifier with cross entropy loss;
* evaluate on the validation set after each epoch and keep the best model;
* optionally, run a final evaluation on the test split and store aggregated
  metrics.

Example usage with ``ImageFolder`` data::

    python -m traffic_status_monitoring.train \
        --data-root /path/to/dataset \
        --output-dir experiments/traffic_status_ast \
        --epochs 30 --batch-size 16

Example usage with the MELAUDIS dataset (point ``--data-root`` to the folder
containing the WAV files)::

    python -m traffic_status_monitoring.train \
        --data-root data/MELAUDIS_Vehicles/Final_Veh \
        --output-dir experiments/melaudis_ast \
        --val-ratio 0.15 --test-ratio 0.15 --audioset-pretrain

The experiment folder will contain the trained ``best_model.pt`` checkpoint,
``label_mapping.json`` and ``metrics.json`` files documenting the training
results.  These artefacts can be reused by the GUI application when integrating
all sub-systems of the intelligent traffic monitoring platform.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchaudio
from torchaudio import transforms as T

try:
    # Prefer the soundfile backend for compatibility across platforms.
    torchaudio.set_audio_backend("soundfile")
except Exception:  # pragma: no cover - backend selection is best-effort.
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]
AST_SRC = REPO_ROOT / "ast-master" / "src"
if AST_SRC.exists() and str(AST_SRC) not in sys.path:
    sys.path.append(str(AST_SRC))

AST_PRETRAINED_DIR = (REPO_ROOT / "ast-master" / "pretrained_models").resolve()
os.environ.setdefault("TORCH_HOME", str(AST_PRETRAINED_DIR))

try:
    from models.ast_models import ASTModel  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import error surfaced to user.
    raise ModuleNotFoundError(
        "Unable to import ASTModel from the bundled ast-master sources. "
        "Please ensure dependencies (including timm==0.4.5) are installed."
    ) from exc

# Ensure the Torch cache directory always points to the local pretrained storage.
os.environ["TORCH_HOME"] = str(AST_PRETRAINED_DIR)

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Configuration options required to build the datasets."""

    sample_rate: int = 16_000
    target_duration: float = 5.0
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 0.0
    f_max: Optional[float] = None
    augment: bool = True

    @property
    def target_num_samples(self) -> int:
        return int(self.sample_rate * self.target_duration)


class TrafficStatusDataset(Dataset[Tuple[Tensor, int]]):
    """Dataset that loads audio files and converts them into log-Mel features."""

    def __init__(
        self,
        items: Sequence[Tuple[Path, int]],
        class_names: Sequence[str],
        config: DataConfig,
        is_train: bool,
    ) -> None:
        self.items = list(items)
        self.class_names = list(class_names)
        self.config = config
        self.is_train = is_train

        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)
        self.resampler: Optional[T.Resample] = None

    def __len__(self) -> int:
        return len(self.items)

    def _load_waveform(self, file_path: Path) -> Tensor:
        waveform, sample_rate = torchaudio.load(file_path)
        # Convert to mono by averaging channels.
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sample_rate:
                self.resampler = T.Resample(sample_rate, self.config.sample_rate)
            waveform = self.resampler(waveform)
        return waveform.squeeze(0)

    def _pad_or_trim(self, waveform: Tensor) -> Tensor:
        num_samples = waveform.size(0)
        target = self.config.target_num_samples
        if num_samples > target:
            start = 0
            if self.is_train and self.config.augment:
                start = random.randint(0, num_samples - target)
            waveform = waveform[start : start + target]
        elif num_samples < target:
            padding = target - num_samples
            waveform = F.pad(waveform, (0, padding))
        return waveform

    def _augment(self, waveform: Tensor) -> Tensor:
        if not self.is_train or not self.config.augment:
            return waveform
        # Random time shift up to +/- 10% of the clip length.
        shift_range = int(0.1 * self.config.target_num_samples)
        if shift_range > 0:
            shift = random.randint(-shift_range, shift_range)
            waveform = torch.roll(waveform, shifts=shift)
        # Add light Gaussian noise (SNR ~ 20 dB).
        noise_std = waveform.std() * 0.1
        if noise_std > 0:
            noise = torch.randn_like(waveform) * noise_std
            waveform = waveform + noise
        return waveform

    def _to_log_mel(self, waveform: Tensor) -> Tensor:
        mel = self.mel_transform(waveform.unsqueeze(0))
        mel = self.amplitude_to_db(mel)
        mel = mel.squeeze(0).transpose(0, 1).contiguous()
        # Normalise each sample to zero mean / unit variance to stabilise training.
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.to(torch.float32)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        file_path, label = self.items[index]
        waveform = self._load_waveform(file_path)
        waveform = self._pad_or_trim(waveform)
        waveform = self._augment(waveform)
        features = self._to_log_mel(waveform)
        return features, label


def infer_feature_shape(dataset: "TrafficStatusDataset") -> Tuple[int, int]:
    if len(dataset) == 0:
        raise RuntimeError("The training dataset is empty; cannot infer feature dimensions.")
    features, _ = dataset[0]
    if features.ndim != 2:
        raise ValueError("Expected spectrogram features with shape (time, freq).")
    return features.shape  # (time, freq)


# ---------------------------------------------------------------------------
# Dataset discovery utilities
# ---------------------------------------------------------------------------


MELAUDIS_LABEL_TOKENS: Dict[str, Tuple[str, ...]] = {
    "FreeFlow": ("_FF_",),
    "LightCongestion": ("_SF_", "_SL_", "_LF_"),
    "HeavyCongestion": ("_HF_", "_HC_", "_TJ_", "_TJF_", "_TJN_"),
}


def _normalise_name(path: Path) -> str:
    return path.stem.upper()


def infer_melaudis_label(file_path: Path) -> Optional[str]:
    """Infer the traffic status label from a MELAUDIS file name."""

    upper_name = _normalise_name(file_path)
    for label, tokens in MELAUDIS_LABEL_TOKENS.items():
        if any(token in upper_name for token in tokens):
            return label
    return None


def collect_melaudis_items(data_root: Path) -> Dict[str, List[Path]]:
    """Collect MELAUDIS wav files grouped by inferred traffic status."""

    grouped: Dict[str, List[Path]] = defaultdict(list)
    wav_files: Iterable[Path] = data_root.rglob("*.wav")
    for wav_file in wav_files:
        label = infer_melaudis_label(wav_file)
        if label is None:
            continue
        grouped[label].append(wav_file)
    return grouped


def stratified_split(
    grouped_items: Dict[str, List[Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    rng = random.Random(seed)
    train: List[Tuple[Path, str]] = []
    val: List[Tuple[Path, str]] = []
    test: List[Tuple[Path, str]] = []

    for label, files in grouped_items.items():
        files_copy = list(files)
        if not files_copy:
            continue
        rng.shuffle(files_copy)
        total = len(files_copy)
        n_val = int(round(total * val_ratio)) if val_ratio > 0 else 0
        n_test = int(round(total * test_ratio)) if test_ratio > 0 else 0
        n_val = min(n_val, total - 1)
        n_test = min(n_test, total - 1 - n_val)

        val_split = files_copy[:n_val]
        test_split = files_copy[n_val : n_val + n_test]
        train_split = files_copy[n_val + n_test :]

        if not train_split and val_split:
            train_split.append(val_split.pop())
        if not train_split and test_split:
            train_split.append(test_split.pop())

        train.extend((path, label) for path in train_split)
        val.extend((path, label) for path in val_split)
        test.extend((path, label) for path in test_split)

    return train, val, test


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(predictions: Tensor, targets: Tensor) -> float:
    preds = predictions.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def build_datasets(config: TrainConfig, data_config: DataConfig) -> Tuple[
    TrafficStatusDataset,
    TrafficStatusDataset,
    Optional[TrafficStatusDataset],
    Dict[int, str],
]:
    def build_from_folder_structure() -> Tuple[
        TrafficStatusDataset,
        TrafficStatusDataset,
        Optional[TrafficStatusDataset],
        Dict[int, str],
    ]:
        def scan_split(split: str) -> List[Tuple[Path, int]]:
            split_dir = config.data_root / split
            if not split_dir.exists():
                return []
            items: List[Tuple[Path, int]] = []
            for label, class_name in enumerate(class_names):
                for wav_file in sorted((split_dir / class_name).glob("*.wav")):
                    items.append((wav_file, label))
            return items

        class_names_local = sorted(
            [d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
        if len(class_names_local) < 2:
            raise RuntimeError(
                "At least two class folders are required under the training split."
            )

        index_to_name_local = {idx: name for idx, name in enumerate(class_names_local)}

        train_items_local = scan_split("train")
        val_items_local = scan_split("val")
        test_items_local = scan_split("test")

        if not val_items_local:
            raise FileNotFoundError(
                "Validation split is empty. Please provide a 'val' directory with audio files."
            )

        train_dataset_local = TrafficStatusDataset(
            train_items_local, class_names_local, data_config, is_train=True
        )
        val_dataset_local = TrafficStatusDataset(
            val_items_local, class_names_local, data_config, is_train=False
        )
        test_dataset_local = (
            TrafficStatusDataset(test_items_local, class_names_local, data_config, is_train=False)
            if test_items_local
            else None
        )

        return (
            train_dataset_local,
            val_dataset_local,
            test_dataset_local,
            index_to_name_local,
        )

    def build_from_melaudis() -> Tuple[
        TrafficStatusDataset,
        TrafficStatusDataset,
        Optional[TrafficStatusDataset],
        Dict[int, str],
    ]:
        grouped = collect_melaudis_items(config.data_root)
        if not grouped:
            raise FileNotFoundError(
                "No MELAUDIS audio files with recognised traffic-status tokens were found."
            )

        class_names_local = sorted(grouped.keys())
        if len(class_names_local) < 2:
            raise RuntimeError("At least two traffic status categories are required.")

        train_items_raw, val_items_raw, test_items_raw = stratified_split(
            grouped, config.val_ratio, config.test_ratio, config.seed
        )

        if not val_items_raw:
            raise RuntimeError(
                "Validation split is empty. Consider increasing the dataset size or adjusting the val ratio."
            )

        index_to_name_local = {idx: name for idx, name in enumerate(class_names_local)}
        name_to_index = {name: idx for idx, name in index_to_name_local.items()}

        def attach_labels(items: List[Tuple[Path, str]]) -> List[Tuple[Path, int]]:
            return [(path, name_to_index[label]) for path, label in items]

        train_dataset_local = TrafficStatusDataset(
            attach_labels(train_items_raw), class_names_local, data_config, is_train=True
        )
        val_dataset_local = TrafficStatusDataset(
            attach_labels(val_items_raw), class_names_local, data_config, is_train=False
        )
        test_dataset_local = (
            TrafficStatusDataset(
                attach_labels(test_items_raw), class_names_local, data_config, is_train=False
            )
            if test_items_raw
            else None
        )

        def describe_split(name: str, items: List[Tuple[Path, str]]) -> None:
            counts = defaultdict(int)
            for _, label in items:
                counts[label] += 1
            pretty = ", ".join(f"{label}: {counts[label]}" for label in class_names_local)
            print(f"{name} split -> {pretty}")

        describe_split("Train", train_items_raw)
        describe_split("Validation", val_items_raw)
        if test_items_raw:
            describe_split("Test", test_items_raw)

        return (
            train_dataset_local,
            val_dataset_local,
            test_dataset_local,
            index_to_name_local,
        )

    train_dir = config.data_root / "train"
    if train_dir.exists():
        return build_from_folder_structure()
    return build_from_melaudis()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> float:
    model.train()
    running_loss = 0.0
    use_amp = scaler is not None and scaler.is_enabled()
    for inputs, targets in data_loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if use_amp:
            assert scaler is not None  # for type checkers
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(data_loader.dataset)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_outputs: List[Tensor] = []
    all_targets: List[Tensor] = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    outputs_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    eval_loss = running_loss / len(data_loader.dataset)
    eval_acc = accuracy(outputs_tensor, targets_tensor)
    return eval_loss, eval_acc


def save_metrics(
    output_dir: Path,
    history: List[Dict[str, float]],
    best_epoch: int,
    label_map: Dict[int, str],
) -> None:
    metrics = {
        "history": history,
        "best_epoch": best_epoch,
        "label_map": label_map,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)


def export_label_mapping(output_dir: Path, label_map: Dict[int, str]) -> None:
    with (output_dir / "label_mapping.json").open("w", encoding="utf8") as f:
        json.dump(label_map, f, indent=2)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root folder.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store models and logs.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="L2 regularisation weight.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Target audio sample rate.")
    parser.add_argument("--duration", type=float, default=5.0, help="Clip duration (seconds) after trimming/padding.")
    parser.add_argument("--n-mels", type=int, default=128, help="Number of Mel filter banks.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size.")
    parser.add_argument("--hop-length", type=int, default=320, help="Hop length for the STFT.")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation during training.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of worker processes for the DataLoader."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio when auto-splitting the MELAUDIS dataset.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio when auto-splitting the MELAUDIS dataset (0 to disable).",
    )
    parser.add_argument("--fstride", type=int, default=10, help="AST patch stride on the frequency axis.")
    parser.add_argument("--tstride", type=int, default=10, help="AST patch stride on the time axis.")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base384",
        choices=["tiny224", "small224", "base224", "base384"],
        help="AST backbone size to instantiate.",
    )
    parser.add_argument(
        "--no-imagenet-pretrain",
        action="store_true",
        help="Disable ImageNet pretraining when initialising AST.",
    )
    parser.add_argument(
        "--audioset-pretrain",
        action="store_true",
        help="Load AudioSet+ImageNet pretrained weights for AST (requires download).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("Validation and test ratios must be non-negative.")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("The sum of validation and test ratios must be less than 1.0.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        data_root=args.data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    data_config = DataConfig(
        sample_rate=args.sample_rate,
        target_duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        augment=not args.no_augment,
    )

    train_dataset, val_dataset, test_dataset, index_to_name = build_datasets(train_config, data_config)
    input_tdim, input_fdim = infer_feature_shape(train_dataset)

    device = torch.device(train_config.device)
    if args.audioset_pretrain and device.type != "cuda":
        print(
            "Warning: AudioSet pretraining is enabled but CUDA is not available. "
            "Loading and fine-tuning the large AST backbone on CPU can be very slow."
        )
    model = ASTModel(
        label_dim=len(index_to_name),
        fstride=args.fstride,
        tstride=args.tstride,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        imagenet_pretrain=not args.no_imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain,
        model_size=args.model_size,
        verbose=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(train_config.patience // 2, 2), factor=0.5, verbose=True
    )

    scaler = GradScaler(enabled=device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=train_config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=train_config.num_workers > 0,
    )

    history: List[Dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(train_config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{train_config.epochs} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if epochs_without_improvement >= train_config.patience:
            print("Early stopping triggered.")
            break

    export_label_mapping(output_dir, index_to_name)
    save_metrics(output_dir, history, best_epoch + 1, index_to_name)

    if test_dataset is not None and (output_dir / "best_model.pt").exists():
        print("Evaluating the best checkpoint on the test set...")
        model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=train_config.num_workers > 0,
        )
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        history.append({
            "epoch": best_epoch + 1,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        })
        save_metrics(output_dir, history, best_epoch + 1, index_to_name)
        print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
