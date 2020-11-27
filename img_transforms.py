import numpy as np
import imgaug
import random
import configparser
import sys

from imgaug import augmenters as iaa
from imgaug import parameters as iap

class DefaultImageTransform:
    def __init__(self, resize:int=224, max_ratio_change:float=1.2):
        self.resize = resize
        self.max_ratio_change = max_ratio_change

    def _resize_crop_and_pad(self, img):
        img = np.array(img)
        h, w = img.shape[0], img.shape[1]
        m, M = np.min((h, w)), np.max((h, w))

        if self.max_ratio_change:
            if self.max_ratio_change > 1:
                max_new_m = int(m*self.max_ratio_change*(self.resize/M))
                new_m = np.min((max_new_m, self.resize))
                min_new_M = int(M/self.max_ratio_change*(self.resize/m))
                new_M = np.max((min_new_M, self.resize))
                size = (new_M, new_m) if h>=w else (new_m, new_M)
            elif self.max_ratio_change == 1:
                size = ("keep-aspect-ratio", self.resize) if h>=w else (self.resize, "keep-aspect-ratio")

            resize = iaa.Resize(size)
            crop_and_pad = iaa.Sequential([
                iaa.CropToFixedSize(width=self.resize, height=self.resize),
                iaa.PadToFixedSize(width=self.resize, height=self.resize, position='center')
            ])

            img = resize.augment_image(img)
            img = crop_and_pad.augment_image(img)

        else:
            resize = iaa.Resize((self.resize, self.resize))
            img = resize.augment_image(img)

        return img


class ImgAugTransform(DefaultImageTransform):
    def __init__(self, resize: int=224, max_ratio_change:float=1.2,
                 crop_freq:float=0.05, max_cropped_part:float=0.3,
                 geometric_freq:float=0.2, flip_freq:float=0.3,
                 color_freq:float=0.2, pooling_freq:float=0.1,
                 dropout_freq:float=0.3, noise_freq:float=0.3,
                 blurs_freq:float=0.3, alphanoise_freq:float=0.3,
                 contrast_freq:float=0.3, turn_gray_freq:float=0,
                 emboss_freq:float=0, snow_freq:float=0,
                 fog_freq:float=0, rain_freq:float=0,
                 jpegcompress_freq:float=0.1,
                 print_config:bool=False, keep_unchanged:float=0.1):

        super().__init__(resize, max_ratio_change)
        self.crop_freq = crop_freq
        self.max_cropped_part = max_cropped_part
        self.geometric_freq = geometric_freq
        self.flip_freq = flip_freq
        self.color_freq = color_freq
        self.pooling_freq = pooling_freq
        self.dropout_freq = dropout_freq
        self.noise_freq = noise_freq
        self.blurs_freq = blurs_freq
        self.alphanoise_freq = alphanoise_freq
        self.contrast_freq = contrast_freq
        self.turn_gray_freq = turn_gray_freq
        self.emboss_freq = emboss_freq
        self.snow_freq = snow_freq
        self.fog_freq = fog_freq
        self.rain_freq = rain_freq
        self.jpegcompress_freq = jpegcompress_freq
        self.keep_unchanged = keep_unchanged

        self.print_config = print_config

        augs_list = []

        if self.crop_freq > 0:
            augs_list.append(iaa.Sometimes(self.crop_freq, self._get_crop_tr()))

        if self.geometric_freq > 0:
            augs_list.append(iaa.Sometimes(self.geometric_freq, self._get_geometric_tr()))

        if self.flip_freq > 0:
            augs_list.append(iaa.Sometimes(self.flip_freq, self._get_flip_tr()))

        if self.color_freq > 0:
            augs_list.append(iaa.Sometimes(self.color_freq, self._get_color_tr()))

        if self.pooling_freq > 0:
            augs_list.append(iaa.Sometimes(self.pooling_freq, self._get_pooling_tr()))

        if self.dropout_freq > 0:
            augs_list.append(iaa.Sometimes(self.dropout_freq, self._get_dropout_tr()))

        if self.noise_freq > 0:
            augs_list.append(iaa.Sometimes(self.noise_freq, self._get_noise_tr()))

        if self.blurs_freq > 0:
            augs_list.append(iaa.Sometimes(self.blurs_freq, self._get_blurs_tr()))

        if self.alphanoise_freq > 0:
            augs_list.append(iaa.Sometimes(self.alphanoise_freq, self._get_alphanoise_tr()))

        if self.contrast_freq > 0:
            augs_list.append(iaa.Sometimes(self.contrast_freq, self._get_contrast_tr()))

        if self.turn_gray_freq > 0:
            augs_list.append(iaa.Sometimes(self.turn_gray_freq, self._get_turn_gray_tr()))

        if self.emboss_freq > 0:
            augs_list.append(iaa.Sometimes(self.emboss_freq, self._get_emboss_tr()))

        if self.snow_freq > 0:
            augs_list.append(iaa.Sometimes(self.snow_freq, self._get_snow_tr()))

        if self.fog_freq > 0:
            augs_list.append(iaa.Sometimes(self.fog_freq, self._get_fog_tr()))

        if self.rain_freq > 0:
            augs_list.append(iaa.Sometimes(self.rain_freq, self._get_rain_tr()))

        if self.jpegcompress_freq > 0:
            augs_list.append(iaa.Sometimes(self.jpegcompress_freq, self._get_compr_tr()))

        self.aug_for_resized = iaa.Sequential(augs_list, random_order=True)

        if print_config:
            config = configparser.ConfigParser()
            config['ImgAugTransform'] = self.__dict__
            del config['ImgAugTransform']['aug_for_resized']
            config.write(fp=sys.stdout)

    def __call__(self, img):
        if self.resize:
            img = self._resize_crop_and_pad(img)

        if random.uniform(0,1) < self.keep_unchanged:
            return img
        return self.aug_for_resized.augment_image(img)

    def _get_crop_tr(self):
        return iaa.Crop(sample_independently=True, percent=((0, self.max_cropped_part/2)))

    def _get_geometric_tr(self):
        return iaa.OneOf([
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-180, 180),
                shear=(-16, 16), #its an angle value in degrees
                order=[0, 1, 3],
                mode=imgaug.ALL),
            iaa.PiecewiseAffine(scale=(0.01,0.04), nb_rows=[3,4,5], nb_cols=[3,4,5]),
            iaa.PerspectiveTransform(scale=(0.01,0.1)),
            iaa.ElasticTransformation(),
            # Aug that can be used by it is too heavy, in my opinion
            iaa.WithPolarWarping(
                [iaa.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}),
                 iaa.CropAndPad(percent=(-0.02, 0.02))]
                ),
        ])

    def _get_flip_tr(self):
        return iaa.OneOf([
            iaa.Fliplr(),
            iaa.Flipud()
        ])

    def _get_color_tr(self):
        return iaa.OneOf([
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-20, 20)),
            iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True),
            iaa.ChangeColorTemperature((5000, 20000)),
            iaa.pillike.EnhanceColor(),
            iaa.ChannelShuffle()
        ])

    def _get_pooling_tr(self):
        return iaa.OneOf([
            iaa.AveragePooling(2),
            iaa.MaxPooling(2),
            iaa.MinPooling(2),
            iaa.MedianPooling(2)
        ])

    def _get_dropout_tr(self):
        return iaa.OneOf([
            iaa.CoarseDropout(p=(0.02, 0.05), size_px=(3,6), per_channel=0.5),
            iaa.CoarseSaltAndPepper(p=(0.02, 0.05), size_px=(3,6), per_channel=0.5),
            iaa.CoarseSalt(p=(0.02, 0.05), size_px=(3,6), per_channel=0.5),
            iaa.CoarsePepper(p=(0.02, 0.05), size_px=(3,6), per_channel=0.5),
            iaa.Cutout(nb_iterations=(1, 5), size=0.1, squared=False)
        ])

    def _get_noise_tr(self):
        return iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=(-0.15*255,0.15*255), scale=(0.0,0.2*255), per_channel=0.5),
            iaa.AdditivePoissonNoise(lam=(0,35), per_channel=0.5),
            iaa.AdditiveLaplaceNoise(loc=(-0.15*255,0.15*255), scale=(0.0,0.1*255), per_channel=0.5),
            iaa.MultiplyElementwise((0.6,1.4), per_channel=0.5),
            iaa.ImpulseNoise(),
            iaa.SaltAndPepper()
        ])

    def _get_blurs_tr(self):
        return iaa.OneOf([
            iaa.GaussianBlur(sigma=(0,1.5)),
            iaa.AverageBlur(k=((1,5),(1,5))),
            iaa.MedianBlur(k=[3,5]),
            iaa.BilateralBlur(d=5, sigma_color=(25,150), sigma_space=(25,150)),
            iaa.MotionBlur(k=(3,6))
        ])

    def _get_alphanoise_tr(self):
        return iaa.OneOf([
            iaa.Sequential([
                iaa.SimplexNoiseAlpha(second=iaa.Multiply((0.1,0.2)), size_px_max = (2,4),
                                      upscale_method='cubic', per_channel=0.2, iterations = 1),
                iaa.Multiply((1.5,1.8))
            ]),
            iaa.Sequential([
                iaa.FrequencyNoiseAlpha(exponent=-4, second=iaa.Multiply(0.2),
                                        per_channel=0.2, size_px_max = 4,
                                        upscale_method='cubic', iterations = 1),
                iaa.Multiply(1.3)
            ]),
            iaa.Sequential([
                iaa.FrequencyNoiseAlpha(exponent=(-4,-2), second=iaa.Multiply(0.2),
                                        per_channel=0.2, size_px_max = (4,10),
                                        upscale_method='cubic', iterations = 1),
                iaa.Multiply(1.3)
            ]),
            iaa.CloudLayer(intensity_mean=[50,100,150,200], intensity_freq_exponent=-4,
                           intensity_coarse_scale=1, alpha_min=0, alpha_multiplier=0.5,
                           alpha_size_px_max=5, alpha_freq_exponent=-8.0, sparsity=1,
                           density_multiplier=1.5)])

    def _get_contrast_tr(self):
        return iaa.OneOf([
            iaa.GammaContrast(gamma=[0.5, 0.7, 1.2, 1.5] ),
            iaa.SigmoidContrast(gain=[4,5,6,7], cutoff=[0.4, 0.6], per_channel=0.2),
            iaa.LogContrast(gain=(0.8,1.3), per_channel=0.2),
            iaa.LinearContrast(alpha=[0.7, 1.5, 2.0],  per_channel=0.2),
            iaa.AllChannelsCLAHE(clip_limit=[1,2,3], tile_grid_size_px=[2,5,10], per_channel=0.2),
            iaa.CLAHE(clip_limit=[1,2,3], tile_grid_size_px=[2,5,10])
        ])

    def _get_turn_gray_tr(self):
        return iaa.Grayscale((0.5,1))

    def _get_emboss_tr(self):
        return iaa.Emboss(alpha=0.5, strength=[0.1,0.5, 0.7])

    def _get_snow_tr(self):
        return iaa.Snowflakes()

    def _get_fog_tr(self):
        return iaa.Fog()

    def _get_rain_tr(self):
        return iaa.Rain()

    def _get_compr_tr(self):
        return iaa.JpegCompression([95,93,90,85,80,70,60])


class ImgResizeAndPad(DefaultImageTransform):
    def __init__(self, resize=224, max_ratio_change = 1.2):
        super().__init__(resize, max_ratio_change)

    def __call__(self, img):
        return self._resize_crop_and_pad(img)
