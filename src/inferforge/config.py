"""Validated configuration for the initial single-device training workload.

Configuration files use TOML, which Python can parse without optional packages.
Paths in files are resolved relative to the configuration file's directory.
Loading settings never creates directories, downloads data, or detects hardware.
Serving and benchmark settings will be added with their implementations.
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from inferforge.errors import ConfigurationError


class Device(StrEnum):
    """Requested device; auto selection is resolved by the runtime layer."""

    AUTO = "auto"
    CPU = "cpu"
    XPU = "xpu"
    CUDA = "cuda"


class Precision(StrEnum):
    """Initial training precisions; support is checked by the runtime layer."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as cause:
        raise ConfigurationError(f"{name} must be a finite number") from cause
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        limit = "positive" if positive else "nonnegative"
        raise ConfigurationError(f"{name} must be finite and {limit}")
    return number


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty path string")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution preferences, independent of actual device availability."""

    device: Device = Device.AUTO
    precision: Precision = Precision.FP32

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device) or not isinstance(self.precision, Precision):
            raise ConfigurationError("runtime settings require Device and Precision enum values")


@dataclass(frozen=True)
class DataConfig:
    """Food-101 input settings; subset selection is explicit and deterministic."""

    root: Path = Path("datasets/food-101")
    image_size: int = 224
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 42
    subset_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise ConfigurationError("data.root must be a Path")
        for name in ("image_size", "batch_size"):
            _integer(getattr(self, name), f"data.{name}", 1)
        for name in ("num_workers", "seed"):
            _integer(getattr(self, name), f"data.{name}", 0)
        if self.subset_size is not None:
            _integer(self.subset_size, "data.subset_size", 1)


@dataclass(frozen=True)
class ModelConfig:
    """The first model path is ConvNeXt-Tiny with a Food-101 classifier."""

    name: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 101

    def __post_init__(self) -> None:
        if self.name != "convnext_tiny":
            raise ConfigurationError("model.name must be 'convnext_tiny'")
        _boolean(self.pretrained, "model.pretrained")
        if _integer(self.num_classes, "model.num_classes", 1) != 101:
            raise ConfigurationError("the initial Food-101 workload requires 101 classes")


@dataclass(frozen=True)
class TrainingConfig:
    """Initial training controls; defaults do not claim tuned hyperparameters."""

    epochs: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    checkpoint_dir: Path = Path("artifacts/checkpoints")

    def __post_init__(self) -> None:
        _integer(self.epochs, "training.epochs", 1)
        _number(self.learning_rate, "training.learning_rate", positive=True)
        _number(self.weight_decay, "training.weight_decay", positive=False)
        if not isinstance(self.checkpoint_dir, Path):
            raise ConfigurationError("training.checkpoint_dir must be a Path")


@dataclass(frozen=True)
class Config:
    """Immutable settings for the initial workload."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _section(
    document: Mapping[str, object], name: str, allowed: set[str]
) -> dict[str, object]:
    section = document.get(name, {})
    if not isinstance(section, dict):
        raise ConfigurationError(f"{name} must be a TOML table")
    unknown = section.keys() - allowed
    if unknown:
        raise ConfigurationError(f"unknown {name} settings: {', '.join(sorted(unknown))}")
    return section


def load_config(path: Path) -> Config:
    """Load TOML settings, rejecting misspellings, invalid values, and read errors.

    Missing sections and fields use documented dataclass defaults. File-based
    paths, including defaults, resolve relative to the file. Backend-specific
    precision compatibility must be validated when execution is requested.
    """
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as cause:
        raise ConfigurationError(f"cannot load configuration {path}: {cause}") from cause

    unknown = document.keys() - {"runtime", "data", "model", "training"}
    if unknown:
        raise ConfigurationError(f"unknown configuration sections: {', '.join(sorted(unknown))}")
    runtime = _section(document, "runtime", {"device", "precision"})
    data = _section(
        document, "data", {"root", "image_size", "batch_size", "num_workers", "seed", "subset_size"}
    )
    model = _section(document, "model", {"name", "pretrained", "num_classes"})
    training = _section(
        document, "training", {"epochs", "learning_rate", "weight_decay", "checkpoint_dir"}
    )
    try:
        device = Device(runtime.get("device", "auto"))
        precision = Precision(runtime.get("precision", "fp32"))
    except (ValueError, TypeError) as cause:
        raise ConfigurationError("runtime.device or runtime.precision is unsupported") from cause

    base = path.resolve().parent
    subset = data.get("subset_size")
    model_name = model.get("name", "convnext_tiny")
    if not isinstance(model_name, str):
        raise ConfigurationError("model.name must be a string")
    return Config(
        runtime=RuntimeConfig(device=device, precision=precision),
        data=DataConfig(
            root=_path(data.get("root", "datasets/food-101"), "data.root", base),
            image_size=_integer(data.get("image_size", 224), "data.image_size", 1),
            batch_size=_integer(data.get("batch_size", 8), "data.batch_size", 1),
            num_workers=_integer(data.get("num_workers", 0), "data.num_workers", 0),
            seed=_integer(data.get("seed", 42), "data.seed", 0),
            subset_size=None if subset is None else _integer(subset, "data.subset_size", 1),
        ),
        model=ModelConfig(
            name=model_name,
            pretrained=_boolean(model.get("pretrained", True), "model.pretrained"),
            num_classes=_integer(model.get("num_classes", 101), "model.num_classes", 1),
        ),
        training=TrainingConfig(
            epochs=_integer(training.get("epochs", 1), "training.epochs", 1),
            learning_rate=_number(
                training.get("learning_rate", 0.001), "training.learning_rate", positive=True
            ),
            weight_decay=_number(
                training.get("weight_decay", 0.01), "training.weight_decay", positive=False
            ),
            checkpoint_dir=_path(
                training.get("checkpoint_dir", "artifacts/checkpoints"),
                "training.checkpoint_dir",
                base,
            ),
        ),
    )
