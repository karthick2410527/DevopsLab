import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare import load_dataset


def test_load_dataset():
    df = load_dataset()

    # Dataset should not be empty
    assert len(df) > 0

    # Regression target column
    assert "target" in df.columns

    # No missing values
    assert df.isnull().sum().sum() == 0


def test_prepare_script_runs(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    (tmp_path / "params.yaml").write_text(
        "prepare:\n"
        "  test_size: 0.2\n"
        "  random_state: 42\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "src" / "prepare.py"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    assert (tmp_path / "data" / "train.csv").exists()
    assert (tmp_path / "data" / "test.csv").exists()