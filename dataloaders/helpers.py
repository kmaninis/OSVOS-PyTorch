import numpy as np
import cv2


def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def overlay_mask(im, ma, color=np.array([255, 0, 0])/255.0):
    assert np.max(im) <= 1.0

    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    alpha = 0.5

    # fg    = im*alpha + np.ones(im.shape)*(1-alpha) * np.array([23,23,197])/255.0
    fg = im * alpha + np.ones(im.shape) * (1 - alpha) * color  # np.array([0,0,255])/255.0

    # Whiten background
    alpha = 1.0
    bg = im.copy()
    bg[ma == 0] = im[ma == 0] * alpha + np.ones(im[ma == 0].shape) * (1 - alpha)
    bg[ma == 1] = fg[ma == 1]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg
