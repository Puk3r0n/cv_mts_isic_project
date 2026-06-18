import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def brightness(image, alpha):
    if alpha == 0:
        return image
    return ImageEnhance.Brightness(image).enhance(2 ** alpha)


def contrast(image, alpha):
    if alpha == 1.0:
        return image
    return ImageEnhance.Contrast(image).enhance(alpha)


def gamma(image, alpha):
    if alpha == 1.0:
        return image
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.clip(arr ** alpha, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def hue_shift(image, alpha):
    if alpha == 0:
        return image
    hsv = np.asarray(image.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(alpha)) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def gaussian_noise(image, alpha):
    if alpha == 0:
        return image
    rng = np.random.default_rng(42)
    arr = np.asarray(image).astype(np.float32)
    return Image.fromarray(np.clip(arr + rng.normal(0.0, alpha, arr.shape), 0, 255).astype(np.uint8))


def gaussian_blur(image, alpha):
    if alpha == 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=alpha))


def jpeg(image, alpha):
    if alpha >= 100:
        return image
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(alpha))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def occlusion(image, alpha):
    if alpha == 0:
        return image
    width, height = image.size
    side = int(((alpha / 100) * width * height) ** 0.5)
    rng = random.Random(42)
    x = rng.randint(0, max(0, width - side))
    y = rng.randint(0, max(0, height - side))
    arr = np.asarray(image).copy()
    arr[y:y + side, x:x + side] = 128
    return Image.fromarray(arr)


PERTURBATIONS = {
    "brightness": (brightness, [-0.5, -0.25, 0, 0.25, 0.5], "EV", 0),
    "contrast": (contrast, [0.5, 0.75, 1.0, 1.25, 1.5], "factor", 1.0),
    "gamma": (gamma, [0.5, 0.75, 1.0, 1.5, 2.0], "exponent", 1.0),
    "hue": (hue_shift, [0, 8, 16, 32, 64], "degrees", 0),
    "noise": (gaussian_noise, [0, 5, 10, 20, 40], "sigma_uint8", 0),
    "blur": (gaussian_blur, [0, 1, 2, 4, 8], "radius_px", 0),
    "jpeg": (jpeg, [10, 20, 40, 60, 80, 100], "quality", 100),
    "occlusion": (occlusion, [0, 5, 10, 20, 30], "percent_area", 0),
}

INVARIANT_RANGES = {
    "brightness": [-0.25, 0.25],
    "contrast": [0.75, 1.25],
    "gamma": [0.75, 1.5],
    "hue": [0, 16],
    "jpeg": [80, 100],
}
