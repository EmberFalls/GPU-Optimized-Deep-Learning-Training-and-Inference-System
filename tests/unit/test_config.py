"""Regression tests for validated configuration and file-relative paths."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from inferforge.config import (
    Config,
    DataConfig,
    Device,
    ModelConfig,
    Precision,
    RuntimeConfig,
    TrainingConfig,
    load_config,
)
from inferforge.errors import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "train.toml"


def test_empty_file_uses_defaults_without_creating_artifact_directories(config_path: Path) -> None:
    config_path.write_text("", encoding="utf-8")

    config = load_config(config_path)

    assert config.runtime == RuntimeConfig()
    assert config.model == ModelConfig()
    assert config.data.root == config_path.parent / "datasets/food-101"
    assert config.training.checkpoint_dir == config_path.parent / "artifacts/checkpoints"
    assert config.data.subset_size is None
    assert not config.data.root.exists()
    assert not config.training.checkpoint_dir.exists()


def test_explicit_settings_and_relative_paths(config_path: Path) -> None:
    config_path.write_text(
        """\
[runtime]
device = "cpu"
precision = "bf16"
[data]
root = "../images"
image_size = 256
batch_size = 4
num_workers = 2
seed = 7
subset_size = 64
[model]
name = "convnext_tiny"
pretrained = false
num_classes = 101
[training]
epochs = 3
learning_rate = 0.0005
weight_decay = 0.0
checkpoint_dir = "saved"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.runtime == RuntimeConfig(device=Device.CPU, precision=Precision.BF16)
    assert config.data == DataConfig(
        root=(config_path.parent / "../images").resolve(),
        image_size=256,
        batch_size=4,
        num_workers=2,
        seed=7,
        subset_size=64,
    )
    assert config.model.pretrained is False
    assert config.training == TrainingConfig(
        epochs=3,
        learning_rate=0.0005,
        weight_decay=0.0,
        checkpoint_dir=config_path.parent / "saved",
    )


def test_absolute_path_is_preserved(config_path: Path) -> None:
    root = (config_path.parent / "external-data").resolve()
    config_path.write_text(f"[data]\nroot = '{root.as_posix()}'\n", encoding="utf-8")

    assert load_config(config_path).data.root == root


def test_paths_do_not_depend_on_working_directory(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path.write_text('[data]\nroot = "images"\n', encoding="utf-8")
    elsewhere = config_path.parent / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_config(config_path).data.root == config_path.parent / "images"


@pytest.mark.parametrize("device", list(Device))
@pytest.mark.parametrize("precision", list(Precision))
def test_runtime_preferences_parse_without_hardware_detection(
    config_path: Path, device: Device, precision: Precision
) -> None:
    config_path.write_text(
        f"[runtime]\ndevice = '{device.value}'\nprecision = '{precision.value}'\n",
        encoding="utf-8",
    )

    assert load_config(config_path).runtime == RuntimeConfig(device=device, precision=precision)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[unexpected]\nx = 1", "unknown configuration sections"),
        ("[runtime]\ndevcie = 'cpu'", "unknown runtime settings"),
        ("[data]\nbtach_size = 4", "unknown data settings"),
        ("[model]\nunknown = true", "unknown model settings"),
        ("[training]\nepoch = 1", "unknown training settings"),
        ("runtime = 1", "runtime must be a TOML table"),
        ("data = []", "data must be a TOML table"),
        ("[runtime]\ndevice = 'invalid'", "unsupported"),
        ("[runtime]\nprecision = 'int8'", "unsupported"),
        ("[runtime]\ndevice = true", "unsupported"),
        ("[data]\nbatch_size = true", "data.batch_size"),
        ("[data]\nbatch_size = 0", "data.batch_size"),
        ("[data]\nbatch_size = 1.5", "data.batch_size"),
        ("[data]\nimage_size = -1", "data.image_size"),
        ("[data]\nnum_workers = -1", "data.num_workers"),
        ("[data]\nseed = -1", "data.seed"),
        ("[data]\nsubset_size = 0", "data.subset_size"),
        ("[data]\nroot = ''", "data.root"),
        ("[data]\nroot = 42", "data.root"),
        ("[model]\nname = 'vit'", "model.name"),
        ("[model]\nname = 1", "model.name"),
        ("[model]\nnum_classes = 100", "101 classes"),
        ("[model]\npretrained = 'true'", "model.pretrained"),
        ("[training]\nepochs = 0", "training.epochs"),
        ("[training]\nlearning_rate = 0.0", "training.learning_rate"),
        ("[training]\nlearning_rate = true", "training.learning_rate"),
        ("[training]\nlearning_rate = inf", "training.learning_rate"),
        ("[training]\nlearning_rate = nan", "training.learning_rate"),
        ("[training]\nweight_decay = -1", "training.weight_decay"),
        ("[training]\ncheckpoint_dir = '   '", "training.checkpoint_dir"),
    ],
)
def test_invalid_settings_raise_configuration_error(
    config_path: Path, document: str, message: str
) -> None:
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(config_path)


@pytest.mark.parametrize("payload", [b"[malformed", b"\xff"])
def test_invalid_files_preserve_parse_error(config_path: Path, payload: bytes) -> None:
    config_path.write_bytes(payload)

    with pytest.raises(ConfigurationError, match="cannot load configuration") as caught:
        load_config(config_path)

    assert isinstance(caught.value.__cause__, ValueError)


def test_missing_file_preserves_read_error(config_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot load configuration") as caught:
        load_config(config_path)

    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    "create",
    [
        lambda: DataConfig(batch_size=0),
        lambda: DataConfig(num_workers=-1),
        lambda: DataConfig(subset_size=0),
        lambda: ModelConfig(name="vit"),
        lambda: ModelConfig(num_classes=100),
        lambda: TrainingConfig(epochs=0),
        lambda: TrainingConfig(learning_rate=float("nan")),
        lambda: TrainingConfig(weight_decay=-0.1),
    ],
)
def test_direct_construction_validates_values(create: Callable[[], object]) -> None:
    with pytest.raises(ConfigurationError):
        create()


def test_configuration_is_immutable() -> None:
    config = Config()

    with pytest.raises(FrozenInstanceError):
        # Deliberately violate the frozen contract to verify runtime enforcement.
        config.data.batch_size = 32  # type: ignore[misc]
