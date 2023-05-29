import torch


class ColorbasedSIFT(object):
    def __init__(self, extractor1, extractor2):
        self.extractor1 = extractor1
        self.extractor2 = extractor2

    def __call__(self, img):
        return torch.cat((self.extractor1(img), self.extractor2(img)))

    def __repr__(self):
        return f"ColorbasedSIFT(extractor1={self.extractor1}, extractor2={self.extractor2})"
