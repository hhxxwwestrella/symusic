# Use as in for p in `python pick_random_files.py `; do cp $p mini_midis/; done

import random
from pathlib import Path

def pick_random_files(directory: str, n: int = 50):
    files = list(Path(directory).glob('*'))
    if len(files) < n:
        raise ValueError(f"Only {len(files)} files in directory, cannot pick {n}.")
    return random.sample(files, n)

if __name__ == "__main__":
    dir_path = "../mirex2025/sym-music-gen/data/filtered_aria/train"  # change this to your directory
    selected = pick_random_files(dir_path, 50)
    for f in selected:
        print(f)
