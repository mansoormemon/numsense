import os

from numsense import SegmentDisplayReader


samples_dir = "samples"

if __name__ == "__main__":
    reader = SegmentDisplayReader()
    samples = sorted(os.listdir(samples_dir))
    for s in samples:
        text = reader(f"{samples_dir}/{s}")
        print(text)
