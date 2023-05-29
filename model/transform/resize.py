import torch
from torchvision.transforms.functional import resize, pad


class ResizeRatio(object):
    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, img):
        """
        resize image into new dimension with respected to given ratio
        """
        w, h = img.shape[1], img.shape[2]

        return resize(img, [int(w * self.ratio), int(h * self.ratio)])

    def __repr__(self):
        return f"ResizeRatio(ratio={self.ratio})"


class ResizeRatioPad(object):
    def __init__(self, width, height, constant=0, **kwargs):
        self.w = width
        self.h = height
        self.c = constant

    def __call__(self, img):
        # image tensor: channels, height, width
        ratio = (self.w/self.h < 1)
        ratio1 = (img.shape[2]/img.shape[1] < 1)
        if ratio1 != ratio:
            img = torch.rot90(img, dims=[1, 2])

        w1, h1 = img.shape[2], img.shape[1]
        diff = self.w / self.h - w1 / h1

        if diff > 1e-3:
            # pad zeros to increase width
            gap = int(self.w * h1 / self.h) - w1
            img = pad(img, [int(gap/2), 0], self.c, "constant")
        elif diff < -1e-3:
            # pad zeros to increase height
            gap = int(self.h * w1 / self.w) - h1

            img = pad(img, [0, int(gap/2)], self.c, "constant")

        return resize(img, size=[self.h, self.w])

    def __repr__(self):
        return f"ResizeRatioPadZero(w={self.w}, h={self.h})"
