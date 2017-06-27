#!/usr/bin/env python3

# Group 6
# Hugo Moraes Dzin 8532186
# Matheus Gomes da Silva Horta 8532321
# Raul Zaninetti Rosa 8517310

import cv2
import numpy as np


def pasteFace(target, new_face):
    # Resize to target's dimensions
    new_y, new_x = target.shape[:2]
    new_face = cv2.resize(new_face, (new_x, new_y))
    
    # Calculate mask position and size
    rows, cols = target.shape[:2]
    mask_h = int(np.round(rows * 0.9))
    mask_w = int(np.round(cols * 0.75))

    center = (cols // 2, rows // 2)
    dims = (mask_w // 2, mask_h // 2)

    mask = np.zeros(target.shape, dtype=np.uint8)
    cv2.ellipse(mask, center, dims, 0, 0, 360, (1, 1, 1), -1)

    anti_mask = np.ones(target.shape, dtype=np.uint8)
    anti_mask -= mask

    new_face = hist_match_wrapper(new_face, target)

    # Paste face
    masked_face = new_face * mask
    target *= anti_mask
    target += masked_face


def hist_match_wrapper(source, template):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2YCR_CB)

    channels = [None] * 3

    for i in range(3):
        channels[i] = np.uint8(hist_match(source[:, :, i], template[:, :, i]))

    return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCR_CB2BGR)


# SOURCE: https://stackoverflow.com/a/33047048/3638744
# Uses histogram from <template> to modify <source>
def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
