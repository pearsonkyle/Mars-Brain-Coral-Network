import matplotlib as mpl
mpl.use('Agg')
import time
import os
import cv2
import glob
import glymur
import pickle
import argparse
import numpy as np
import tensorflow as tf
from skimage.util.shape import view_as_windows
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, label

from train_classifier import make_rnet, make_mnet, make_cnet, blockDCT

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", '--weights', help='which pretrained weights to use for the network', type=str,
                         #default="models/mobilenet_256_spatial_2_0.h5")
                         default="models/mobilenet_128_spatial_2_0.h5")
                         #default="models/cnn_256_spatial_2_0.h5")
                         #default="models/cnn_256_dct_2_4.h5")

    parser.add_argument("-i", "--input", help="HiRISE Image Filepath", type=str,
                         #default="training/data/brain_coral/PSP_001410_2210/PSP_001410_2210_RED.JP2")
                         default="images/ESP_016287_2205_RED.JP2")
                         #default="images/ESP_016215_2190_RED.JP2")
                         #default="images/ESP_060698_2220_RED.JP2")
                         #default="images/ESP_077488_2205_RED.JP2")

    parser.add_argument('--threads', action='store', default=8, type=int, 
                        help='Number of threads to use when decoding JPEG2000 files')

    return parser.parse_args()

if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()

    # set number of threads for glymur to decode with
    glymur.set_option('lib.num_threads', args.threads)

    # parse model params from name in weights files (eg. cnn_128_spatial_2_2.h5)
    parts = os.path.basename(args.weights).split("_")
    model_type = parts[0]                    # cnn, mobilenet, resnet
    size = int(parts[1])                     # tiling size
    stride = int(parts[1])
    dtype = parts[2]                         # spatial or dct
    res = int(parts[3].split(".")[0])        # resolution factor for glymur

    # make model
    if model_type == "cnn":
        model_fn = make_cnet
    elif model_type == "mobilenet":
        model_fn = make_mnet
    elif model_type == "resnet":
        model_fn = make_rnet

    if dtype == 'spatial':
        input_size = (size, size)
    elif dtype == 'dct':
        input_size = (size//8, size//8, 64)
    
    # load weights
    model = model_fn(input_size, 2)
    model.summary()
    model.load_weights(args.weights)
    print("weights loaded!")

    # check if input exists
    if not os.path.exists(args.input):
        print("input file does not exist")

    idata = glymur.Jp2k(args.input).read(rlevel=res).astype(np.float32)

    sfactor = 1#0.9 # stride factor, how much to overlap tiles (1-x)

    BI = view_as_windows(idata, size, step=int(stride*sfactor))
    print(" tiles in map:",BI.shape)
    
    X = BI.reshape(-1, size, size)

    if dtype == 'dct':
        X = blockDCT(X)

    pred = model.predict(X, batch_size=16, verbose=1)

    # reshape back (BI.shape[0], BI.shape[1], classes)
    BO = pred.reshape(BI.shape[0], BI.shape[1], 2)

    # create mask using max of channel
    mask = BO.argmax(axis=2) == 0 # 0th channel is brain coral

    # clean mask by removing single pixels not surrounded by other pixels, erosion is too aggressive
    groups, ngroups = label(mask, structure=np.ones((3,3)))

    # remove groups smaller than 1 pixel
    for i in range(1, ngroups+1):
        if (groups == i).sum() < 2:
            mask[groups == i] = False

    # save mask using cv2
    maskfile = os.path.splitext(args.input)[0] + "_classifier_mask.png"
    print("saving mask to", maskfile)
    cv2.imwrite(maskfile, mask.astype(np.uint8)*255)