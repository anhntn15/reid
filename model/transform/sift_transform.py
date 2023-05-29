import cv2
import numpy
import numpy as np
import torch

from utils.log_config import get_custom_logger

logger = get_custom_logger(__name__, 'DEBUG')


class SIFTTransform(object):
    def __init__(self, nfeatures: int = 20, contrastThreshold=0.09):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold)
        self.nfeatures = nfeatures
        self.contrastThreshold = contrastThreshold

    def __call__(self, img):
        """
        transform an image to new representation using its SIFT features
        :return: a tensor with shape (nfeatures, 128)
        """
        x = numpy.asarray(img)
        kp, des = self.sift.detectAndCompute(x, None)

        if des is None:
            logger.debug(f'SIFT failed to detect keypoint')
            des = numpy.zeros(shape=(self.nfeatures, 128))

        if des.shape[0] < self.nfeatures:
            logger.debug(f'SIFT detect only {des.shape[0]} keypoints')
            # handle case number of detected features < nfeatures: pad zeros
            des = np.vstack((des, np.zeros(shape=(self.nfeatures - des.shape[0], 128))))

        des = des[:self.nfeatures, :]

        # des = numpy.expand_dims(des, axis=0)  # add one dimension as gray image (1,w,h)
        return torch.from_numpy(des).float().flatten()

    def __repr__(self):
        return f"SIFTTransform(nfeatures={self.nfeatures}, contrastThreshold={self.contrastThreshold})"
