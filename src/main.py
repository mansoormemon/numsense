import os

import cv2 as cv

from numsense import SegmentDisplayReader


samples_dir = "samples"

if __name__ == "__main__":
    reader = SegmentDisplayReader()
    samples = sorted(os.listdir(samples_dir))
    for s in samples:
        img, predictions = reader(f"{samples_dir}/{s}")
        number = ""
        for x, y, digit in predictions:
            number += digit
            cv.putText(img, digit, (x - 16, y + 24), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        print(number)
        cv.imshow(number, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
