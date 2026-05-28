from pathlib import Path
import subprocess
import sys
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from config import config


def test_train_predict_score_smoke():
    output_dir = ROOT / config["output_dir"]
    if output_dir.exists():
        shutil.rmtree(output_dir)

    subprocess.run(
        [sys.executable, "code/src/train.py", "--num_epochs_override", "2"],
        cwd=ROOT,
        check=True,
    )

    summary = (output_dir / "final_score.txt").read_text(encoding="utf-8")
    assert "Best epoch: 2" in summary

    subprocess.run([sys.executable, "code/src/predict.py"], cwd=ROOT, check=True)
    assert (ROOT / "output" / "result.csv").exists()

    subprocess.run([sys.executable, "test/score_self.py"], cwd=ROOT, check=True)
    result = pd.read_csv(ROOT / "temp" / "tmp.csv")
    assert "Final Score" in result.columns
    assert pd.notna(result.loc[0, "Final Score"])


if __name__ == "__main__":
    test_train_predict_score_smoke()
    print("phase11 smoke test OK")
