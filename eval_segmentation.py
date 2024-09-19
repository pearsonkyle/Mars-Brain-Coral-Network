import os
import gc
import cv2
import glymur
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from skimage.util.shape import view_as_windows
from scipy.ndimage import binary_erosion, binary_dilation, label

from train_unet import make_unet3

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", '--weights', help='which pretrained weights to use for the network', type=str,
                        #default="models/unet_512_0_rescale_ft_weights.h5")
                        default="models/unet_512_0_normalize_ft_weights.h5")


    parser.add_argument("-i", "--input", help="HiRISE Image Filepath", type=str,
                         #default="images/PSP_001410_2210_RED_A_01_ORTHO.JP2")
                         #default="images/ESP_016287_2205_RED_A_01_ORTHO.JP2")
                         default="images/ESP_037262_1845_RED_A_01_ORTHO.JP2")
                         #default="images/ESP_077488_2205_RED.JP2")

    # add arg for low-memory mode (float16)
    parser.add_argument('--lowmem', action='store_true', help='use float16 to save memory (slower but should work on 32GB Mac M2Pro)')

    parser.add_argument('--threads', action='store', default=8, type=int, 
                        help='Number of threads to use when decoding JPEG2000 files')

    parser.add_argument('--fast', action='store_true', help='skips robust clean up of mask (removes single pixels not surrounded by other pixels)')

    parser.add_argument("--gpu", default=0, type=int, help='specify which gpu to use')

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()

    # set number of threads for glymur to decode with
    glymur.set_option('lib.num_threads', args.threads)

    # Restrict TensorFlow to use a particular GPU, if one is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if args.gpu > len(gpus):
                raise(f"gpu number {args.gpu_num} not supported")
            tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # convert model name to size and res parameters
    # e.g. unet_512_0_rescale_ft_weights.h5
    parts = args.weights.split("_")
    args.size = int(parts[1])
    args.res = int(parts[2])
    preprocessing = parts[3]

    # make model
    unet, encoder = make_unet3(input_shape=(args.size,args.size), output_channels=2, preprocessing=preprocessing)

    # load weights
    if os.path.exists(args.weights):
        if 'encoder' in args.weights:
            encoder.load_weights(args.weights)
            segmentor = encoder
        else:
            unet.load_weights(args.weights)
            segmentor = unet

        segmentor.summary()
        print(f"Loaded weights from {args.weights}")
    else:
        raise(f"Could not find weights file {args.weights}")

    # load image
    print(f"Loading image {args.input}...")
    idata = glymur.Jp2k(args.input).read(rlevel=args.res)

    # tile image, reduces boxy artifacts
    overlap = 0.5
    BI = view_as_windows(idata, args.size, step=int(args.size*overlap))
    BIshape = BI.shape # for memory opt. later
    
    # mask out black parts of image
    imask = np.mean(BI, axis=(2,3))>10

    # flatten array
    BIR = BI[imask].reshape(-1, args.size, args.size)

    # change precision of prediction to float16 to save memory
    if args.lowmem:
        pred = np.zeros((BIR.shape[0], *segmentor.output.shape[1:]), dtype=np.float16)
    else:
        pred = np.zeros((BIR.shape[0], *segmentor.output.shape[1:]))

    # batch size depends on computer memory
    batch = 1000

    # clean up memory
    del BI, idata

    # print number of batches
    print(f"Segmenting {BIR.shape[0]} tiles in {BIR.shape[0]//batch+1} batches of {batch} tiles")

    # batch data based on gpu memory
    for i in range(0,BIR.shape[0],batch):
        sub=slice(i,i+batch)

        # predict
        if args.lowmem:
            # cast to float16 to save memory, will be slower
            pred[sub] = segmentor.predict(BIR[sub], batch_size=8, verbose=args.verbose).astype(np.float16)
        else:
            pred[sub] = segmentor.predict(BIR[sub], batch_size=8, verbose=args.verbose)

        # clean up memory for really large images
        if BIR.shape[0] > batch and i%batch == 0:
            _ = gc.collect()

    # clean up memory
    del BIR
    _ = gc.collect()

    print('Reshaping output...')
    # only take center of each prediction, this ignores edge effects in the cnn
    crop = slice(int(segmentor.output.shape[1]*(0.5-overlap/2)),int(segmentor.output.shape[1]*(0.5+overlap/2)))
    pred = pred[:, crop,crop]

    # reshape to original image size
    BO = np.zeros((BIshape[0], BIshape[1], pred[0].shape[0], pred[0].shape[1], pred[0].shape[2]))
    BO[imask] = pred

    # convert to image with multiple channels
    heatmap = BO.swapaxes(1,2).reshape(BIshape[0]*pred[0].shape[0], BIshape[1]*pred[0].shape[1], pred[0].shape[2] )

    # clean up memory
    del BO, pred, imask

    # create mask using max of channel
    mask = heatmap[:,:,0] > 0.55 # 0th channel is brain coral

    # clean up memory
    del heatmap

    # clean up mask
    print("Cleaning up mask...")
    mask = binary_erosion(mask, iterations=4)

    # clean mask by removing single pixels not surrounded by other pixels, erosion is too aggressive
    groups, ngroups = label(mask, structure=np.ones((3,3)))

    if not args.fast:
        # remove groups smaller than 3 pixel, may take ~1-2 hours
        for i in tqdm(range(1, ngroups+1)):
            if (groups == i).sum() < 512*512*0.5:
                mask[groups == i] = False

    mask = binary_dilation(mask, iterations=4)

    # save mask to disk
    maskfile = os.path.splitext(args.input)[0] + "_segmentation_mask.png"
    cv2.imwrite(maskfile, mask.astype(np.uint8)*255)
    print("Saved mask to", maskfile)