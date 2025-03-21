import os

import cv2 as cv

from PIL import Image, ImageDraw


class Typeface:
    def __init__(self, data, out_dir="~/.numsense/typefaces"):
        self.name = data["name"]
        self.w, self.h, self.thickness = data["dimensions"].values()
        self.digits = data["digits"]
        self.segments = self.__compute_segments()
        self.out_dir = os.path.expanduser(f"{out_dir}/{self.name}")
        self.templates = {}

    def __compute_segments(self):
        st = self.thickness
        sl = self.w - 2 * st
        h = self.h
        return {
            "A": [(st, 0), (st + sl, st)],
            "B": [(st + sl, st), (self.w, st + sl)],
            "C": [(st + sl, h // 2), (self.w, h // 2 + sl)],
            "D": [(st, h - st), (st + sl, h)],
            "E": [(0, h // 2), (st, h // 2 + sl)],
            "F": [(0, st), (st, st + sl)],
            "G": [(st, h // 2 - st // 2), (st + sl, h // 2 + st // 2)],
        }

    def supported_characters(self):
        return list(self.digits.keys())

    def render(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for digit in self.digits:
            img = Image.new("L", (self.w, self.h), "black")
            draw = ImageDraw.Draw(img)
            for seg in self.digits[digit]:
                draw.rectangle(self.segments[seg], fill="white")
            img.save(os.path.join(self.out_dir, f"{digit}.png"))

    def load(self):
        for c in self.supported_characters():
            template_path = os.path.join(self.out_dir, f"{c}.png")
            self.templates[c] = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

    def compile(self):
        self.render()
        self.load()
