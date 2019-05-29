#!/usr/local/bin/env python
import argparse
import cv2
import os
import numpy as np
import json


def float2byte(img):
    return 255 * (img - np.min(img)) / np.ptp(img).astype(np.uint8)


def contrast_magnitude(img, blur_size):
    """
    Computes the pixel-wise contrast magnitude of an image using approximation of 2nd derivative Sobel kernel.
    :param img: The grayscale image to process
    :param blur_size: The width of the Gaussian blur kernel to apply
    :return: 2d array of contrast magnitudes
    """
    blurred_img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    kernel_d2x = np.array([1, -2, 1]).shape = (1, 3)
    kernel_d2y = np.array([1, -2, 1]).shape = (3, 1)
    d2x = cv2.filter2D(blurred_img, cv2.CV_32F, kernel_d2x)
    d2y = cv2.filter2D(blurred_img, cv2.CV_32F, kernel_d2y)
    contrast = (np.absolute(d2x) + np.absolute(d2y)) / 2.0

    return contrast


def build_disparity_map(grayscale_imgs, focal_dists, gradient_blur_size, focalplane_blur_size, pyramid_levels=1):
    """
    Selects the highest-contrast pixel for each x,y coordinate across the image sequence.
    :param grayscale_imgs: The image sequence
    :param focal_dists: The focal distances of each image in the image sequence
    :param gradient_blur_size: The width of the Gaussian blur kernel to apply for contrast measurements
    :param focalplane_blur_size: The width of the median blur kernel to apply for focal plane measurements
    :param pyramid_levels: The number of pyramid levels to downsample the image by prior to focal plane blurring
    :return: 2d array of focal distances (i.e. the focal distance of the grayscale image with the highest contrast at each x,y)
    """
    contrast_imgs = []
    for img in grayscale_imgs:
        contrast_imgs.append(contrast_magnitude(img, gradient_blur_size))

    contrast_imgs = np.asarray(contrast_imgs)
    contrast_argmax = np.argmax(contrast_imgs, axis=0).astype(np.float32)

    # Downsample the contrast_argmax array prior to applying median blur
    for l in range(pyramid_levels):
        contrast_argmax = cv2.pyrDown(contrast_argmax)
    border_width = focalplane_blur_size // 2
    contrast_argmax = cv2.copyMakeBorder(contrast_argmax, border_width, border_width, border_width, border_width, cv2.BORDER_REPLICATE)
    contrast_argmax = cv2.medianBlur(contrast_argmax, focalplane_blur_size)
    contrast_argmax = contrast_argmax[border_width:-border_width, border_width:-border_width]
    for l in range(pyramid_levels):
        contrast_argmax = cv2.pyrUp(contrast_argmax)

    # Assign focal distance of corresponding image to each pixel
    contrast_argmax = contrast_argmax.astype(np.int64).clip(0, focal_dists.size - 1)
    disparity_map = np.take(focal_dists, contrast_argmax)
    return disparity_map, contrast_argmax


def build_focusstack_img(bgr_imgs, contrast_argmax):
    """
    Assembles a focus-stacked RGB image from a sequence of variable-focus images.
    :param bgr_imgs: The image sequence
    :param contrast_argmax: 2d array of optimal focal distance indeces
    :return: composite RGB image of pixels sourced from the image sequence
    """
    b_channels = []
    g_channels = []
    r_channels = []
    for img in bgr_imgs:
        b, g, r = cv2.split(img)
        b_channels.append(b)
        g_channels.append(g)
        r_channels.append(r)

    b_channels = np.asarray(b_channels)
    g_channels = np.asarray(g_channels)
    r_channels = np.asarray(r_channels)

    fstack_b = np.zeros(contrast_argmax.shape, np.int64)
    fstack_g = np.zeros(contrast_argmax.shape, np.int64)
    fstack_r = np.zeros(contrast_argmax.shape, np.int64)

    for index, val in np.ndenumerate(contrast_argmax):
        fstack_b.itemset(index, b_channels[val][index])
        fstack_g.itemset(index, g_channels[val][index])
        fstack_r.itemset(index, r_channels[val][index])

    focusstack_img = cv2.merge((fstack_b, fstack_g, fstack_r))

    return focusstack_img


def load_data(basedir):
    groundtruth_filename = os.path.join(basedir, 'groundtruth.json')

    if not os.path.isfile(groundtruth_filename):
        raise Exception("File not found: {}".format(groundtruth_filename))

    with open(groundtruth_filename) as groundtruth_file:
        groundtruth_data = json.load(groundtruth_file)

    grayscale_imgs = []
    bgr_imgs = []
    focal_dists = []
    for img_info in groundtruth_data:
        filename = os.path.join(basedir, img_info['filename'])
        if not os.path.isfile(filename):
            raise Exception("File not found: {}".format(filename))

        # TODO: Apply correction for perspective distortion (need a dataset with known camera intrinsics)
        gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        grayscale_imgs.append(gray_img)
        bgr_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        bgr_imgs.append(bgr_img)
        focal_dists.append(img_info['focal_distance_cm'])

    input_shape = grayscale_imgs[0].shape
    i = 0
    for img in grayscale_imgs:
        if input_shape != img.shape:
            raise Exception("Image '{}' has invalid dimensions. Should be [{}] but was [{}].".format(groundtruth_data[i]['filename'], input_shape, img.shape))
        i += 1

    focal_dists = np.asarray(focal_dists, dtype=np.float32)

    return grayscale_imgs, bgr_imgs, focal_dists


def main():
    parser = argparse.ArgumentParser(description="Depth from focus calculator")

    parser.add_argument('-d', '--data-dir', action='store', default='data/balcony/', help='Directory containing images and a groundtruth.json file')
    parser.add_argument('-g', '--gradient-blur', action='store', default=3, type=int, help='Gaussian blur to apply for contrast detection')
    parser.add_argument('-m', '--focalplane-blur', action='store', default=105, type=int, help='Median blur to apply for focal plane measurements')
    parser.add_argument('-p', '--pyramid-levels', action='store', default=1, type=int, help='Number of pyramid levels to downsample gradient image by prior to focal plane blurring')

    args = parser.parse_args()

    basedir = args.data_dir
    gradient_blur_size = args.gradient_blur
    focalplane_blur_size = args.focalplane_blur
    pyramid_levels = args.pyramid_levels

    # Load the image sequence, compute contrasts, and build the focus stack
    grayscale_imgs, bgr_imgs, focal_dists = load_data(basedir)
    disparity_map, contrast_argmax = build_disparity_map(grayscale_imgs, focal_dists, gradient_blur_size, focalplane_blur_size, pyramid_levels)
    focusstack_img = build_focusstack_img(bgr_imgs, contrast_argmax)

    cv2.imwrite('focus_stack_g{}-m{}-p{}.png'.format(gradient_blur_size, focalplane_blur_size, pyramid_levels), focusstack_img)

    # Format the disparity map for viewing
    disparity_img = float2byte(disparity_map)
    disparity_img = cv2.cvtColor(disparity_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('depth_map_g{}-m{}-p{}.png'.format(gradient_blur_size, focalplane_blur_size, pyramid_levels), disparity_img)


if __name__ == '__main__':
    main()
