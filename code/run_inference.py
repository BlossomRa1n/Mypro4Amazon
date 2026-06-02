import subprocess
import sys
import os

# 确保当前目录在 sys.path 中
os.chdir(os.path.dirname(os.path.abspath(__file__)))

scripts = ['inference_full.py', 'inference.py']

for i, script in enumerate(scripts, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(scripts)}] Running {script}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)

print("\nAll inference scripts completed successfully.")
