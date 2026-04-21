import shutil
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    print("\n\nfix_lib_conflict\n\n")
    python_exe = sys.executable

    # 1. Gỡ các package dễ xung đột
    subprocess.run(
        [
            python_exe,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "numpy",
            "scipy",
            "pytorch-lightning",
            "lightning",
            "torchmetrics",
            "open-clip-torch",
        ],
        check=True,
    )

    # 2. Xóa cache pip
    pip_cache = Path("/root/.cache/pip")
    if pip_cache.exists():
        shutil.rmtree(pip_cache)

    # 3. Cài lại từ requirements.txt
    print("\n\n run requirements.txt\n\n")
    subprocess.run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            "/kaggle/input/datasets/tmtrnhelloworld/hybridcbmplcode/requirements.txt",
        ],
        check=True,
    )
