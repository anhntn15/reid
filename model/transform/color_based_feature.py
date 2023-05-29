import enum

import torch


class FeatureType(enum.Enum):
    BRIGHTNESS = "brightness"
    AVERAGE_COLOR = "average-color"
    COLOR_VARIANCE = "color-variance"
    AB = "ab"  # brightness + average-color
    BC = "bc"  # brightness + color-variance
    AC = "ac"  # average-color + color-variance
    ABC = "abc"  # all features


class ColorBasedFeature(object):
    def __init__(self, h: int, w: int, feature_type: str):
        """
        :param h: height of subimage
        :param w: width of subimage
        :param feature_type: type of feature to extract from subimage region
        """
        self.feature_type = FeatureType(feature_type)
        self.sub_h = h
        self.sub_w = w

    def _get_feature_value(self, subimg) -> list:
        if self.feature_type == FeatureType.BRIGHTNESS:
            return [subimg.mean().item()]
        if self.feature_type == FeatureType.AVERAGE_COLOR:
            return subimg.mean(dim=[1, 2]).tolist()
        if self.feature_type == FeatureType.COLOR_VARIANCE:
            return [subimg.std().item()]
        if self.feature_type == FeatureType.BC:
            return [subimg.mean().item(), subimg.std().item()]
        if self.feature_type == FeatureType.AC:
            return subimg.mean(dim=[1, 2]).tolist() + [subimg.std().item()]
        if self.feature_type == FeatureType.AB:
            return subimg.mean(dim=[1, 2]).tolist() + [subimg.mean().item()]
        if self.feature_type == FeatureType.ABC:
            return subimg.mean(dim=[1, 2]).tolist() + [subimg.mean().item(), subimg.std().item()]

    @staticmethod
    def get_multiple_factor(feature_type):
        if feature_type == FeatureType.BRIGHTNESS:
            return 1
        if feature_type == FeatureType.AVERAGE_COLOR:
            return 3
        if feature_type == FeatureType.COLOR_VARIANCE:
            return 1
        if feature_type == FeatureType.BC:
            return 2
        if feature_type == FeatureType.AC:
            return 4
        if feature_type == FeatureType.AB:
            return 4
        if feature_type == FeatureType.ABC:
            return 5

    def count_num_subimages(self, img_h: int, img_w: int) -> int:
        """
        calculate size of features vector after transformed
        :param img_h: height of applied img
        :param img_w: width of applied img
        """

        h_step, w_step = int(self.sub_h / 2), int(self.sub_w / 2)
        return (int(img_h / h_step) - 1 - (img_h % h_step == 0)) * (int(img_w / w_step) - 1 - (img_w % w_step == 0)) * (
            self.get_multiple_factor(self.feature_type))

    def __call__(self, img):
        # In PyTorch, images are represented as [channels, height, width]
        h, w = img.shape[1], img.shape[2]
        h_step, w_step = int(self.sub_h / 2), int(self.sub_w / 2)
        features = []
        for hi in range(0, h - self.sub_h, h_step):
            for wi in range(0, w - self.sub_w, w_step):
                subimg = img[:, hi:hi + self.sub_h, wi:wi + self.sub_w]
                features.extend(self._get_feature_value(subimg))
        return torch.FloatTensor(features)

    def __repr__(self):
        return f"ColorBasedFeature(h={self.sub_h}, w={self.sub_w}, feature_type={self.feature_type})"
