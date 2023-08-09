import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import gc
import cv2
import glymur
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", '--res', help='which resolution to use for the network', type=int,
                        default=3)

    parser.add_argument("-i", "--input", help="HiRISE Image Filepath", type=str,
                         #default="training/data/brain_coral/PSP_001410_2210/PSP_001410_2210_RED.JP2")
                         #default="images/ESP_016287_2205_RED.JP2")
                         #default="images/ESP_077488_2205_RED.JP2")
                         #default="images/ESP_016215_2190_RED.JP2")
                         default="images/ESP_052385_2205_RED.JP2")
                         #default="images/ESP_060698_2220_RED.JP2")

    parser.add_argument('--threads', action='store', default=8, type=int, 
                        help='Number of threads to use when decoding JPEG2000 files')

    parser.add_argument('-s', '--save_training_mask', action='store_true', help='save segmenation mask named training_mask.jpg')

    return parser.parse_args()

def plot(image, heatmap, filename, alpha=0.25, dpi=0):

    fig,ax = plt.subplots(1,figsize=(10,10*heatmap.shape[0]/heatmap.shape[1]))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    ax.imshow(image,cmap='binary_r')
    ax.imshow(heatmap,interpolation='none',cmap='Greens',alpha=alpha,extent=[0,image.shape[1],image.shape[0],0],vmin=0,vmax=1)
    
    plt.axis("off")
    plt.savefig(filename)
    # else:
    #     plt.savefig(filename, dpi=2*np.max(np.array(heatmap.shape)//10))

    plt.close()
    print(f" {filename} saved!")

if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()

    # set number of threads for glymur to decode with
    glymur.set_option('lib.num_threads', args.threads)

    # load image
    print(f"Loading image {args.input}...")
    idata = glymur.Jp2k(args.input).read(rlevel=args.res)

    # open mask from file
    maskfile_segmentation = os.path.splitext(args.input)[0] + "_segmentation_mask.png"
    mask_segmentation = cv2.imread(maskfile_segmentation, cv2.IMREAD_GRAYSCALE)
    # resize mask to match image
    mask_segmentation = resize(mask_segmentation, idata.shape, order=0)
    
    if args.save_training_mask:
        # save mask_classifier as jpg named training_mask.jpg
        cv2.imwrite(os.path.splitext(args.input)[0]+"_training_mask.png", mask_segmentation)
        #print(f" {os.path.splitext(args.input)[0]}_training_mask.jpg saved!")
    else:
        # save overlay
        plot(idata, mask_segmentation, os.path.splitext(args.input)[0]+"_overlay_segmentation.png")
        
        maskfile_classifier = os.path.splitext(args.input)[0] + "_classifier_mask.png"
        mask_classifier = cv2.imread(maskfile_classifier, cv2.IMREAD_GRAYSCALE)
        # resize mask to match image
        mask_classifier = resize(mask_classifier, idata.shape, order=0)
        plot(idata, mask_classifier, os.path.splitext(args.input)[0]+"_overlay_classifier.png")    
