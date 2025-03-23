import cv2 as cv
import numpy as np

from .typeface import Typeface
from .typefaces import DEFAULT_TYPEFACE


class SegmentDisplayReader:
    def __init__(self, conf_thresh=0.6, typeface=DEFAULT_TYPEFACE):
        self.conf_thresh = conf_thresh

        self.typeface = Typeface(typeface)
        self.typeface.compile()

    def __preprocess_image(self, img, blur_strength=5, closure_tolerance=7):
        blurred = cv.GaussianBlur(img, (blur_strength, blur_strength), 0)
        _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        kernel = np.ones((closure_tolerance, closure_tolerance), np.uint8)
        closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        return closed

    def __find_bounding_boxes(self, processed_img, min_area=128):
        contours, _ = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) >= min_area]
        return [cv.boundingRect(cnt) for cnt in filtered_contours]

    def __match_digit_templates(self, roi):
        recognized_digit = None
        best_score = float("-inf")
        for digit, template in self.typeface.templates.items():
            resized_template = cv.resize(template, roi.shape[::-1])
            result = cv.matchTemplate(roi, resized_template, cv.TM_CCOEFF_NORMED)
            score = np.max(result)
            if score > best_score:
                best_score = score
                recognized_digit = digit
        return recognized_digit if best_score >= self.conf_thresh else None

    def __sort_regions(self, text_blocks, y_threshold=10):
        text_blocks.sort(key=lambda block: block[1])
        lines = []
        current_line = []
        for block in text_blocks:
            _, y, _ = block
            if not current_line:
                current_line.append(block)
            else:
                _, last_y, _ = current_line[-1]
                if abs(y - last_y) <= y_threshold:
                    current_line.append(block)
                else:
                    lines.append(sorted(current_line, key=lambda b: b[0]))
                    current_line = [block]
        if current_line:
            lines.append(sorted(current_line, key=lambda b: b[0]))
        return [block for line in lines for block in line]

    def __call__(self, img_path):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"'{img_path}' not found")

        processed_img = self.__preprocess_image(img)
        bounding_boxes = self.__find_bounding_boxes(processed_img)

        recognized_digits = []
        for x, y, w, h in bounding_boxes:
            roi = processed_img[y : y + h, x : x + w]
            recognized_digit = self.__match_digit_templates(roi)
            if recognized_digit is not None:
                recognized_digits.append((x, y, recognized_digit))
            else:
                recognized_digits.append((x, y, "-"))
        recognized_digits = self.__sort_regions(recognized_digits)
        recognized_number = "".join(str(digit) for _, _, digit in recognized_digits)
        return recognized_number
