import os

from numsense import SegmentDisplayReader


samples_dir = "samples"

if __name__ == "__main__":
    reader = SegmentDisplayReader(conf_thresh=0.35)
    samples = sorted(os.listdir(samples_dir))
    for s in samples:
        ret_val = reader(f"{samples_dir}/{s}")
        print(ret_val)
