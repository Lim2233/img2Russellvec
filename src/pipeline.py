import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

STEPS = [
    "python src/CLI_Tools/CLI_SliceCsv.py",
    "python src/CLI_Tools/CLI_emotion2VAVec2.py",
    "python src/CLI_Tools/CLI_makeImgBigAndColorful.py",
    "python src/CLI_Tools/CLI_getFeatureA.py"
]
INPUTDIR="Data/rawCsv"
OUTPUTDIR="Data/output"

def run_pipeline(input_dir, output_dir):
    work_dir = Path(tempfile.mkdtemp())
    prev = Path(input_dir)
    for i, cmd in enumerate(STEPS, 1):
        if i == len(STEPS):
            cur = Path(output_dir)
        else:
            cur = work_dir / f"step_{i}"
        cur.mkdir(parents=True, exist_ok=True)

        full_cmd = f"{cmd} {prev} {cur}"
        print(f"[{i}/{len(STEPS)}] {full_cmd}")
        subprocess.run(full_cmd, shell=True, check=True)
        prev = cur
    print(f"Done. Output: {output_dir}")

if __name__ == "__main__":
    run_pipeline(INPUTDIR, OUTPUTDIR)