import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import time
import glob
import pickle
import argparse 
import numpy as np
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import urllib.request
from urllib.parse import urljoin
import glymur

from utils import find_links

def get_hirise_image(fname):
    """
    Search HiRISE for image number

    Parameters
    ----------
    fname : str
        HiRISE image name

    Returns
    -------
    str
        URL to HiRISE image
    """
    base = 'https://www.uahirise.org/'
    url = urljoin(base, fname)
    links = find_links(url)
    for j, link in enumerate(links):
        # anaglyph is the stereo pair
        if "RED.JP2" in link and "ANAGLYPH" not in link:
            return link
    return ""


def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Choose a directory of images to process"
    parser.add_argument("-d", "--datadir", help=help_, type=str, default="training/")

    help_ = "Segmentation input size [px] -> S x S"
    parser.add_argument("-s", "--size", help=help_, type=int, default=256)

    parser.add_argument("-r", "--res", help="resolution to decode JPEG2000 files at (0 is highest)", type=int,
                        default=1)

    parser.add_argument("-t", "--threads", help="number of threads for background class", default=4, type=int)

    return parser.parse_args()

def plot_samples(data, masks=[], filename="samples.png"):
    f,ax = plt.subplots(10,10,figsize=(20,20))
    # get random indices that don't repeat
    indices = np.random.choice(np.arange(0,len(data)),
                               min(15 *20,len(data)),
                               replace=False)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_axis_off()
            try:
                ri = indices[i*ax.shape[1]+j]
                ax[i,j].imshow(data[ri],cmap='binary')#,vmin=0,vmax=255)
                if len(masks) > 0:
                    # TODO in future extend beyond 2 classes with colormap
                    ax[i,j].imshow(masks[ri][:,:,0]*0.5,alpha=0.5,vmin=0,vmax=1)
                    ax[i,j].imshow(masks[ri][:,:,1],alpha=0.5,vmin=0,vmax=1)
            except:
                continue
    plt.tight_layout()
    plt.savefig(filename)
    print("{} saved".format(filename))
    plt.close()


def create_samples(datadir, size, res=0, disk="/projects/mdap/HiRISE/", preprocess=False):

    print("Creating training data...")
    # get list of all images
    class_dirs = glob.glob(os.path.join(datadir,"*/"))
    classes = [os.path.basename(d[:-1]) for d in class_dirs]
    # reverse order so brain coral class if first
    
    # ensure brain_coral is first in class_dirs
    if "brain_coral" in classes:
        idx = classes.index("brain_coral")
        class_dirs.insert(0, class_dirs.pop(idx))
        classes.insert(0, classes.pop(idx))
    

    print("detected classes:", classes)
    
    # alloc training samples
    data = {
        'image':[],
        'mask':[]
    }

    # loop through each class and extract the mask
    for i,cdir in enumerate(class_dirs):
        print(cdir)
        image_dirs = glob.glob(os.path.join(cdir,'*/'))
        
        # estimate number of bg samples needed to balance BC
        nsamples = 1500
        if 'background' in cdir:
            #nsamples = int(len(data['image'])/len(image_dirs))+1 # class ratio 50/50
            nsamples = int(2*len(data['image'])/len(image_dirs))+1 # 66/33 bg/bc
        
        # loop over images and extract tiles
        for j,idir in enumerate(image_dirs):
            print(idir)
            imgs = glob.glob(os.path.join(idir,'*JP2'))

            maskimg = os.path.join(idir, "training_mask.jpg")
            if not os.path.exists(maskimg):
                maskimg = os.path.join(idir, "training_mask.png")

            if len(imgs) == 0:
                fname = idir.split("/")[-2]
                # search HiRISE for image number
                img_url = get_hirise_image(fname)

                # search for image on disk
                if os.path.exists(os.path.join(disk, img_url.split("ESP/")[-1])): # TODO what about PSP images?
                    ainput = os.path.join(disk, img_url.split("ESP/")[-1])
                    print('image found on disk')
                else:
                    file_name = os.path.join(cdir, fname, f"{fname}_RED.JP2")
                    if not os.path.exists(file_name):

                        try:
                            site = urllib.request.urlopen(img_url)
                        except:
                            continue
                        fsize = site.length/1024/1024
                        print(f"downloading {fsize:.1f} mb from:")
                        print(f" {img_url}")

                        try:
                            urllib.request.urlretrieve(img_url, file_name)
                        except:
                            urllib.request.urlretrieve(img_url, file_name)

                    ainput = file_name
            else:
                ainput = imgs[0]

            if os.path.exists(maskimg):
                print("Opening: ",ainput)
                try:
                    tstart = time.time()
                    if "brain" in idir:
                        if os.path.exists(os.path.join(idir, "resolution50")):
                            if res == 1: # 50 m resolution
                                idata = glymur.Jp2k(ainput).read(rlevel=0)
                            elif res == 0:
                                # only using for 1024 ws
                                idata = glymur.Jp2k(ainput).read(rlevel=0)
                                #print(f"25 m resolution not available : {idir}")
                                #continue # not high enough resolution
                            else:
                                idata = glymur.Jp2k(ainput).read(rlevel=res-1)
                        else:
                            # 25 m resolution
                            idata = glymur.Jp2k(ainput).read(rlevel=res)
                    else:
                        # any native res for background
                        idata = glymur.Jp2k(ainput).read(rlevel = res)
                    decode = time.time()-tstart
                    print(f" decoding time: {decode:.2f} s")

                    # aye aye, images are two different formats...
                    if ".png" in maskimg:
                        imask = cv2.imread(maskimg, cv2.IMREAD_COLOR)[:,:,0]
                    else:
                        imask = cv2.imread(maskimg, cv2.IMREAD_GRAYSCALE)
                except:
                    print("Error opening image")
                    continue

                # resize mask if different than image (e.g. resolution changes during decoding)
                if imask.shape[0] != idata.shape[0]:
                    imask = cv2.resize(imask, (idata.shape[1],idata.shape[0]), interpolation=cv2.INTER_NEAREST)

                # rasterize image
                BI = view_as_windows(idata, size, step=int(size*0.75))
                BIR = BI.reshape(-1, size, size)

                # rasterize mask
                BM = view_as_windows(imask, (size,size), step=int(size*0.75))
                BMR = BM.reshape(-1, size, size)

                # only sample areas with a majority masked
                percent = np.sum(BMR, axis=(1,2)) / (BMR.shape[1]*BMR.shape[2]) / 255

                # skip areas of image with no data
                dark = np.mean(BIR, axis=(1,2))

                if "background" not in cdir:
                    # if most of image is label
                    amask = (dark > np.percentile(dark, 33)) & (percent > 0.90)

                    # randomly choose a fraction of images
                    gmask = np.zeros(amask.shape,dtype=bool)
                    ri = np.random.choice(np.argwhere(amask).flatten(),min(amask.sum(),2000),replace=False)
                else:
                    # just background + skip black areas
                    amask = (dark > np.percentile(dark, 33)) & (percent > 0.75)

                    # randomly choose a fraction of images
                    gmask = np.zeros(amask.shape,dtype=bool)
                    ri = np.random.choice(np.argwhere(amask).flatten(),min(amask.sum(),nsamples),replace=False)

                gmask[ri] = True

                # expand dims based on num classes
                carr = np.zeros((*BMR[gmask].shape, len(classes)))
                carr[:,:,:,i] = BMR[gmask]/255

                # save some samples from that image to check for data leakage
                if gmask.sum():

                    plot_samples(BIR[gmask], carr, os.path.join(idir,f"samples_{size}_{res}.pdf"))

                    if preprocess:
                        for k,raw_img in enumerate(BIR[gmask]):
                            # normalize image
                            raw_img = raw_img.astype(np.float32)
                            raw_img = (raw_img - np.mean(raw_img)) / np.std(raw_img)
                            #data['image'].append(cv2.Laplacian(cv2.GaussianBlur(raw_img,(3,3),0), cv2.CV_64F))
                            data['image'].append(raw_img)
                            data['mask'].append(carr[k])
                    else:
                        data['mask'].extend(carr) 
                        data['image'].extend(BIR[gmask])
                    print("total samples:",len(data['image']))

                del(idata)
                del(imask)
                del(carr)

    if len(data['image']) != 0:
        data['classes'] = classes
        data['mask'] = np.asarray(data['mask'], dtype=bool)
        data['image'] = np.asarray(data['image'], dtype=np.uint16)

        onehot = data['mask'].mean(axis=(1,2))
        class_count = (onehot>0.5).sum(0)
        for i,cl in enumerate(classes):
            percent = class_count[i]/sum(class_count)
            print(f"{cl}: {class_count[i]}/{sum(class_count)} ({100*percent:.2f} %)")
        print(f"Resolution: {res}")

        print("saving data to disk...")
        #pickle.dump(data, open("training/training_samples_{}.pkl".format(size),"wb"))
        plot_samples(data['image'], data.get("mask",[]), f"training/samples_{size}_{res}.pdf")
        return data['image'], data['mask'], data['classes']
    else:
        print("no data found")
        return None, None, None


if __name__ == '__main__':

    args = parse_args()
    glymur.set_option('lib.num_threads', args.threads)
    X,y,c = create_samples(datadir=args.datadir, size=args.size, res=args.res, preprocess=False)