import random
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import numbers
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import rotate


class RandomLRFlip(object):
    '''
    If the prediction has a left/right bias, one can apply this augmentationn
    on the training data.
    '''
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        u = np.random.uniform(0,1)
        if u < self.p:
            img = np.fliplr(img)
        return img


class CenterCrop(object):
    """
    Performs center crop of an image of a certain size.
    Modified version from torchvision.

    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        w, h = img.shape[0],img.shape[1]
        tw, th, = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img[y1:y1+th,x1:x1+tw] #img.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop(object):
    """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        # Note: image_data_format is 'channel_last'
        # assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = self.size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx)]

class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image_array_rescaled = ndimage.zoom(image, [new_h/h, new_w/w, 1])

        return image_array_rescaled
'''
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    #assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

'''
def correct_gamma16(img, gamma):
    """
    Corrects gamma of a 16-bit image
    """
    img = np.array(img).astype(np.float64)
    img = (img / 65535.) ** gamma
    img = np.uint16(img * 65535)
    img = Image.fromarray(img)
    return img


def correct_gamma8(img, gamma):
    """
    Corrects gamma of an 8-bit image
    """

    img = np.array(img).astype(np.float64)
    img = (img / 255.) ** gamma
    img = np.uint8(img * 255)
    img = Image.fromarray(img)
    return img


class CorrectGamma(object):
    """
    Does random gamma correction

    """

    def __init__(self, g_min, g_max, res=8):
        self.g_min = g_min
        self.g_max = g_max
        self.res = res

    def __call__(self, img):
        gamma = random.random() * self.g_max + self.g_min
        if self.res == 8:
            return correct_gamma8(img, gamma)
        return correct_gamma16(img, gamma)


class Jitter(object):
    """
    Makes a crop of a fixed size with random offset

    """

    def __init__(self, crop_size, j_min, j_max):
        self.crop_size = crop_size
        self.j_min = j_min
        self.j_max = j_max

    def __call__(self, img):
        x1 = random.randint(self.j_min, self.j_max)
        y1 = random.randint(self.j_min, self.j_max)
        return img.crop([x1, y1, x1 + self.crop_size, y1 + self.crop_size])


class Rotate(object):
    """
    Performs random rotation
    """
    def __init__(self, a_min, a_max, interp=Image.BICUBIC):
        self.a_min = a_min
        self.a_max = a_max
        self.interp = interp

    def __call__(self, img):
        '''
        This function used ndimage on PIL images. Otherwise
        used rotate for numpy array.
        :param img:
        :return:
        '''
        angle = random.uniform(self.a_min, self.a_max)
        #return img.rotate(angle, resample=self.interp)
        return rotate(img, angle, reshape=False)


class CorrectBrightness(object):
    """
    Performs random rotation

    """

    def __init__(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(self.b_min, self.b_max)
        return enhancer.enhance(factor)


class CorrectContrast(object):
    """
    Performs random rotation

    """

    def __init__(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(self.b_min, self.b_max)
        return enhancer.enhance(factor)
