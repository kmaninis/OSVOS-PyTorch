import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import random


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


def point_in_segmentation(seg, thres=.5):
    """
    Return random representative point inside segmentation mask, selected in
    the region where the distance transform dt is larger than thres * max(dt)
    seg: binary segmentation
    return: point in format (x, y)
    """
    dt = distance_transform_edt(seg)
    dt = dt > thres * dt.max()

    inds_y, inds_x = np.where(dt > 0)
    pix_id = random.randint(0, len(inds_y) - 1)

    point = inds_x[pix_id], inds_y[pix_id]

    return list(point)


def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name


def make_gaussian(size, sigma=10, center=None):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def make_gt(img, labels, sigma=10):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center {'x': x, 'y': y}
    sigma: sigma of the Gaussian.
    """

    h, w = img.shape[:2]

    gt = make_gaussian((h, w), sigma, labels)

    return gt


def jsonline2mat(line):
    """
    Convert json with fields 'x' and 'y' to matrix
    line: example [{u'x': 355, u'y': 132}, {u'x': 455, u'y': 139}, {u'x': 379, u'y': 231}]
    """
    x = np.nan * np.zeros((len(line), 2))
    for i in range(0, len(line)):
        x[i, 0] = line[i]['x']
        x[i, 1] = line[i]['y']
    return x


def overlay_mask_tool(img, mask, transparency=1.0):
    """
    Overlay a h x w x 3 mask to the image
    img: h x w x 3 image
    mask: h x w x 3 mask
    transparency: between 0 and 1
    """
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask[:, :, 0]) * img[:, :, 0] + mask[:, :, 0] * (
        transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask[:, :, 1]) * img[:, :, 1] + mask[:, :, 1] * (
        transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask[:, :, 2]) * img[:, :, 2] + mask[:, :, 2] * (
        transparency + (1 - transparency) * img[:, :, 2])
    return im_over
