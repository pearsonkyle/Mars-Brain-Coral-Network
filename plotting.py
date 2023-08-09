import os
import time
import glob
import json
import pickle
import argparse 
import numpy as np
import matplotlib.pyplot as plt

import glymur

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Choose a directory to save data"
    parser.add_argument("-o", "--output", help=help_, type=str, default="HiRISE/")

    parser.add_argument("-i", "--input", help="Image or glob string", type=str, default="")

    return parser.parse_args()

def plot(image, heatmap, filename, alpha=0.25, dpi=0):

    fig,ax = plt.subplots(1,figsize=(10,10*heatmap.shape[0]/heatmap.shape[1]))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    ax.imshow(image,cmap='binary_r')
    ax.imshow(heatmap,interpolation='none',cmap='jet',alpha=alpha,extent=[0,image.shape[1],image.shape[0],0],vmin=0,vmax=1)
    
    plt.axis("off")
    if dpi:
        plt.savefig(filename, dpi=dpi)
    else:
        plt.savefig(filename, dpi=2*np.max(np.array(heatmap.shape)//10))

    plt.close()
    print(f" {filename} saved!")

def save_image(data, filename, cmap='jet', vmin=0, vmax=1):
    sizes = np.shape(data)     
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data,vmin=vmin,vmax=vmax, cmap=cmap, interpolation='none')
    plt.savefig(filename, dpi = sizes[0]) 
    plt.close()

def plotzoom(image, heatmap, image_name, size, coords, stride):
    # TODO fix
    ymax, xmax = coords.max(0)*stride
    ymin, xmin = coords.min(0)*stride
    window_shape = [size, size]
    f,ax = plt.subplots(1,figsize=(15,14*dy/dx))
    ax.imshow(image,cmap='binary_r')
    im = ax.imshow(heatmap,cmap='jet',alpha=0.25,extent=[window_shape[1]*0.5,image.shape[1]-window_shape[1]*0.5,image.shape[0]-window_shape[0]*0.5,window_shape[0]*0.5],vmin=0,vmax=1)
    ax.plot([xmin, xmin+window_shape[0]], [ymin, ymin], 'w-')
    ax.plot([xmin, xmin], [ymin, ymin+window_shape[0]], 'w-')
    ax.plot([xmin, xmin+window_shape[0]], [ymin+window_shape[0], ymin+window_shape[0]], 'w-')
    ax.plot([xmin+window_shape[0], xmin+window_shape[0]], [ymin, ymin+window_shape[0]], 'w-')

    plt.tight_layout()
    fname = os.path.join(image_name,"{}_ZOOM_{}_{}.png".format(image_name.split('/')[-1], window_shape[0],stride))
    ax.set_xlim([xmin-stride,xmax+stride])
    ax.set_ylim([ymax+stride,ymin-stride])
    plt.savefig(fname, dpi=250)
    plt.close()
    print(f" {fname} saved!")


def plot_samples(data, masks=[], indices=[], filename="samples.png"):
    f,ax = plt.subplots(7,7,figsize=(20,20))
    # get random indices that don't repeat
    
    if len(indices) == 0:
        indices = np.sort(np.random.choice(np.arange(0,len(data)),
                               min(7*7,len(data)),
                               replace=False))
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_axis_off()
            try:
                ri = indices[i*ax.shape[1]+j]
                im = ax[i,j].imshow(data[ri],cmap='binary_r')#,vmin=0,vmax=255)
                ax[i,j].set_title(i*ax.shape[1]+j)

                if len(masks) > 0:
                    # TODO in future extend beyond 2 classes with colormap
                    try:
                        ax[i,j].imshow(masks[ri]*0.5,alpha=0.25,vmin=0,vmax=1,cmap='jet')
                    except:
                        ax[i,j].imshow(masks[ri][:,:,0]*0.5,alpha=0.25,vmin=0,vmax=1)
                        ax[i,j].imshow(masks[ri][:,:,1],alpha=0.25,vmin=0,vmax=1)
            except:
                continue
    plt.colorbar(im, ax=ax[i,j])
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    print("{} saved".format(filename))
    plt.close()
    return indices

if __name__ == "__main__":

    args = parse_args()

    if "*" in args.input:
        imgs = glob.glob(args.input)
    else:
        imgs = [args.input]

    for i,img in enumerate(imgs):

        tstart = time.time()

        print(img)
        print(" loading...")
        idata = glymur.Jp2k(img)[:].astype(np.float32)

        # save to output directory
        fname = "output/{}/".format(os.path.basename(img).split(".")[0])

        try:
            data = pickle.load(open(os.path.join(fname, "heatmaps.pkl"),"rb"))
        except:
            continue

        plot(idata, data['heatmap'], fname, 64)
        
        if len(data['coords']) > 10000:
            plotzoom(idata, data['heatmap'], fname, 64, data['coords'])
