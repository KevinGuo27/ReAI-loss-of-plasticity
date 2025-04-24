import os

label_modes = ["raw", "one_hot", "multi_hot", "embedding"]
for mode in label_modes:
    print(f"\n🚀 Running training with label mode: {mode}")
    os.system(f"python -m pipeline.train --label-mode {mode}")