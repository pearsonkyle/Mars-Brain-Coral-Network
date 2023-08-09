import matplotlib as mpl
mpl.use('Agg')
import os
import gc
import time
import glob
import json
import shutil
import pickle
import argparse 
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
from skimage.transform import downscale_local_mean

from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glymur

import tensorflow as tf

from plotting import save_image, plot_samples
from train_unet import make_unet3 as make_unet
from train_dct import make_cnet, blockDCT
from utils import hiriseFileGenerator, hiriseLinkGenerator, find_links, hiriseScratchGenerator, download_image

clean = lambda x: x.split('=')[1].strip('"\n\r').strip()
clean_num = lambda x: float(clean(x).split(" ")[0])

# images to skip based on memory constraints
skip_images = ['ESP_011442_2255']

class pipeline():
    def __init__(self, state_data, imageGenerator):
        self.sv = state_data
        self.imageGenerator = imageGenerator
        self.load()

    def load(self):
        if 'rescale' in self.sv['settings'].get("segmentor",""):
            unet, encoder = make_unet(input_shape=(self.sv['settings']['size'],self.sv['settings']['size']), 
                                      output_channels=2, preprocessing='rescale')
        else:
            unet, encoder = make_unet(input_shape=(self.sv['settings']['size'],self.sv['settings']['size']), 
                                      output_channels=2)
        
        if self.sv['settings']['segmentor']:
            if 'unet' in self.sv['settings']['segmentor']:
            	self.segmentor = unet
            else:
            	self.segmentor = encoder
            self.segmentor.load_weights(self.sv['settings']['segmentor'])
            print("segmentation weights loaded!")
        else:
            self.segmentor = unet

        self.classifier = make_cnet(input_shape=(int(self.sv['settings']['size']/2**self.sv['settings']['res'])//8,
                                                 int(self.sv['settings']['size']/2**self.sv['settings']['res'])//8,64), 
                                    output_channels=2)

        if self.sv['settings']['classifier']:
            self.classifier.load_weights(self.sv['settings']['classifier'])
            print('classifier weights loaded!')

    def get_image(self, link, res):

        imgname = os.path.basename(link).split('.')[0].replace("_RED","")

        # create directory to save image
        dirname = os.path.join(self.sv['settings']['output'], imgname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        fname = os.path.join(dirname, f"{imgname}.JP2")

        # download
        if 'https' in link:

            # query link and write to disk
            if not os.path.exists(fname):
                print(f"Downloading {link}")

                response = urllib.request.urlopen(link, timeout=600)
                with open(fname, "wb") as fimage:
                    fimage.write(response.read())
                del response
            else:
                print(f"Decoding (res={res}) {fname}")
            # decode
            idata = glymur.Jp2k(fname).read(rlevel=res).astype(np.float32)
        elif os.path.exists(link):
            print(f"decoding {imgname} at 1/{2**(2*res)} resolution...")
            idata = glymur.Jp2k(link).read(rlevel=res).astype(np.float32)
        else:
            print("image does exist...trying to download (TODO)")
            # TODO 
        return idata

    def get_label(self, link):

        imgname = os.path.basename(link).split('.')[0].replace("_RED","")
        dirname = os.path.join(self.sv['settings']['output'], imgname)
        if 'https' in link:   
            # create directory to save image
            
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            fname = os.path.join(dirname, f"{imgname}.LBL")
            # find and download label
            links = find_links(os.path.join('https://www.uahirise.org/', imgname.replace("_RED","")), pattern="RED.LBL")
            response = urllib.request.urlopen(links[0], timeout=600)
            with open(fname, "wb") as flabel:
                flabel.write(response.read())
            
            with open(fname, "r") as flabel:
                lines = flabel.readlines()        

        else:
            try:
                with open(link.replace("JP2","LBL"), "r") as flabel:
                    lines = flabel.readlines()
            except:
                print(f"{imgname} has broken label")
                links = find_links(os.path.join('https://www.uahirise.org/', imgname.replace("_RED","")), pattern="RED.LBL")
                response = urllib.request.urlopen(links[0], timeout=600)
                fname = os.path.join(dirname, f"{imgname}.LBL")
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                
                with open(fname, "wb") as flabel:
                    flabel.write(response.read())
                
                with open(fname, "r") as flabel:
                    lines = flabel.readlines()        

        # parse line by line for info
        ldata = {}
        for j, line in enumerate(lines):
            if "RATIONALE_DESC" in line:
                ldata['title'] = clean(line)[1:]
            if "MAP_SCALE" in line and "METERS/PIXEL" in line:
                ldata['resolution'] = clean_num(line)
            if "LOCAL_TIME" in line:
                ldata['local_time'] = clean_num(line)
            if "MAXIMUM_LATITUDE" in line:
                maxlat = clean_num(line)
            if "MINIMUM_LATITUDE" in line:
                minlat = clean_num(line)
            if "EASTERNMOST_LONGITUDE" in line:
                maxlong = clean_num(line)
            if "WESTERNMOST_LONGITUDE" in line:
                minlong = clean_num(line)
            if "MRO:OBSERVATION_START_TIME" in line:
                ldata['acq_date'] = clean(line)

        ldata['long']=0.5*(maxlong+minlong)
        ldata['lat']=0.5*(maxlat+minlat)

        return ldata

    def classify(self, img):
        csize = int(self.sv['settings']['size']/2**self.sv['settings']['res'])
        B = view_as_windows(img, csize, step=csize)
        B_dct = blockDCT(B.reshape(-1, csize, csize))
        pred = self.classifier.predict(B_dct, batch_size=16, verbose=1)
        heatmap = pred.reshape(B.shape[0], B.shape[1], pred.shape[-1])
        _ = gc.collect()
        return heatmap

    def segment(self, img, image_name="mosaic.png"):
        stride = self.sv['settings']['size']
        size = self.sv['settings']['size']
        overlap = 0.8

        BI = view_as_windows(img, size, step=int(stride*overlap))
        
        imask = np.mean(BI, axis=(2,3))>10
        BIR = BI[imask].reshape(-1, size, size)
        pred = np.zeros((BIR.shape[0], *self.segmentor.output.shape[1:]))

        if size > 512:
            batch = 500
        else:
            batch = 1000
        # batch data based on gpu memory
        for i in range(0,BIR.shape[0],batch):
            sub=slice(i,i+batch)
            pred[sub] = self.segmentor.predict(BIR[sub], batch_size=8, verbose=1)
            
            # clean up memory for really large image, really slow :(
            if BIR.shape[0] > batch and i%batch == 0:
                _ = gc.collect()
	
        # only take center of each prediction, this ignores edge effects in the cnn
        crop = slice(int(self.segmentor.output.shape[1]*(0.5-overlap/2)),int(self.segmentor.output.shape[1]*(0.5+overlap/2)))
        pred = pred[:, crop,crop]

        # reshape to original image size
        BO = np.zeros((BI.shape[0], BI.shape[1], pred[0].shape[0], pred[0].shape[1], pred[0].shape[2]))
        BO[imask] = pred

        # convert to image with multiple channels
        heatmap = BO.swapaxes(1,2).reshape(BI.shape[0]*pred[0].shape[0],BI.shape[1]*pred[0].shape[1], pred[0].shape[2] )
    
        # pad to match img
        dy = int(img.shape[1] - heatmap.shape[1])
        dx = int(img.shape[0] - heatmap.shape[0])
        heatmap = np.pad(heatmap, ((0,dx), (0,dy), (0,0)))

        # determine index of class
        ci = self.sv['settings']['classes'].index("brain coral")

        # resize image to 4K resolution
        sf = int(heatmap.shape[1]/4096)
        heatmapr = downscale_local_mean(heatmap[:,:,ci], (sf,sf))

        # clean up mask
        hmask = np.round(heatmapr)
        hmask = binary_closing(hmask)
        hmask = binary_erosion(hmask)
        hmask = binary_dilation(hmask)
        percent = np.round(hmask.sum()/(hmask.shape[0]*hmask.shape[1]),4)

        # check percent coverage of class
        percent = hmask.sum()/hmask.flatten().shape[0]

        # create mosaic plot
        if percent > 0.02:
            BH = view_as_windows(heatmap[:,:,ci], size, step=int(stride*overlap))
            imask = np.mean(BH, axis=(2,3)) > 0.75
            BHR = BH[imask].reshape(-1, size, size)
            indices = plot_samples(BI[imask].reshape(-1, size, size), [], [], image_name)
            plot_samples(BI[imask].reshape(-1, size, size), BHR, indices, image_name.replace("mosaic","mosaic_heatmap"))

            # create numbers and labels plot
            pos_samples = np.argwhere(imask)[indices]
            newmask = np.zeros(imask.shape)
            aspect = imask.shape[1]/imask.shape[0]

            fig,ax = plt.subplots(1,figsize=(19*aspect,19))
            plt.subplots_adjust(0,0,1,1)
            newmask[pos_samples[:,0], pos_samples[:,1]] = 1
            ax.imshow(newmask,cmap='binary_r')
            for n,pos in enumerate(pos_samples): ax.text(pos[1]-0.5, pos[0]+0.5,str(n), fontsize=7, color='red',zorder=1)
            ax.set_axis_off()
            plt.savefig(image_name.replace("mosaic","mosaic_numbers"))
            plt.close()
            # convert indices to X,Y
        return heatmapr, hmask, percent

    def run(self, image=None, reprocess=False):
        count = 0
        for i, link in enumerate(self.imageGenerator):

            imgname = os.path.basename(link).split('.')[0].replace("_RED","")
            dirname = os.path.join(self.sv['settings']['output'], imgname)

            # mode to process single image as input
            if image:
                # searchs generator for image
                if image in imgname:
                    continue

            # skip if data exists
            if os.path.exists(os.path.join(dirname, "heatmap.jpg")) and not reprocess:
                print("Skipping image because results exist:", dirname)
                continue
            if imgname in self.sv['empty'] and not reprocess:
                print("Skipping image because it is in the empty list:", imgname)
                continue
            if imgname in skip_images:
                continue
            # Load LBL file
            label = self.get_label(link)

            # TODO make argument for cutoff
            if abs(label['lat']) > 75:
                #print(f" skipping b.c latitude: {label['lat']:.2f}")
                
                continue

            print(i, imgname)
            #print("find image location or download")
            #import pdb; pdb.set_trace()
            

            tstart = time.time()
            img = self.get_image(link, res=self.sv['settings']['res'])
            decode4_time = time.time()-tstart
            mb = os.stat(link).st_size/1024/1024
            print(f" decoding time: {decode4_time:.2f} s ({mb:.1f} mb)")

            # classify
            heatmap = self.classify(img)
            classify_time = time.time() - tstart - decode4_time

            # compute fraction of image covered in class
            ci = self.sv['settings']['classes'].index("brain coral")

            percent = heatmap[:,:,ci].sum()/heatmap[:,:,ci].flatten().shape[0]
            count += 1
            
            # segment image
            if percent > 0.05:
                print(f" Classifier Coverage: ~{percent*100:.2f} %")

                # create directory to save images                
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                save_image(downscale_local_mean(img, (4,4)), os.path.join(dirname, "image.jpg"), cmap="gray", vmin=img.min(), vmax=img.max())
                save_image(heatmap[:,:,ci], os.path.join(dirname, "heatmap_lowres.jpg"), cmap="jet")

                # segmentation
                if self.sv['settings']['segmentor']:

                    del img ## clear low-res image
                    tstart = time.time()

                    # change res based on resolution in label
                    img = self.get_image(link, res=0) # full spatial res
    
                    decode1_time = time.time()-tstart
                    heatmap, hmask, percent = self.segment(img, os.path.join(dirname, "mosaic.jpg"))
                    segment_time = time.time() - tstart - decode1_time
                    print(f" decoding time: {decode4_time:.2f} s ({mb:.1f} mb)")
                    print(f" segment time: {segment_time:.2f} s")

                    # if percent coverage is large
                    if percent > 0.02:
                        # save masks
                        save_image(heatmap, os.path.join(dirname, "heatmap.jpg"), cmap="jet")
                        save_image(hmask*0.5, os.path.join(dirname, "mask.jpg"), cmap="jet")
                    else:
                        print("removing empty dir:", dirname)
                        self.sv['empty'][imgname] = {'fpath':link}
                        # delete image from disk and memory
                        shutil.rmtree(dirname)
                    del hmask, heatmap
                else:
                    decode1_time = 0
                    segment_time = 0
                # fullres = binary_erosion(np.round(heatmap),iterations=2)
                # outline = np.logical_xor(fullres, binary_dilation(fullres))
                # np.argwhere(outline)
                # pickle.dump(np.argwhere(outline), open(os.path.join(dirname, "heatmap_fullres.pkl"), "wb"))

                # save state
                self.sv['data'][imgname] = {
                    'fpath': link,
                    'percent': percent,
                    'title': label['title'],
                    'resolution': label['resolution'],
                    'local_time': label['local_time'],
                    'acq_date': label['acq_date'],
                    'long': label['long'],
                    'lat': label['lat'],
                    'decode_1/4_time':decode4_time,
                    'classify_time':classify_time,
                    'decode_1/1_time':decode1_time,
                    'segment_time':segment_time
                }
                print(f" Segmentation coverage: {self.sv['data'][imgname]['percent']*100:.2f} %")

                del img
                _ = gc.collect()
                
            else:
                print("removing empty dir:", dirname)
                self.sv['empty'][imgname] = {'fpath':link}
                # delete image from disk and memory
                shutil.rmtree(dirname)
                del img

            # save sv to disk
            with open(os.path.join(self.sv['settings']['output'], "statevector.json"), 'w') as f:
                json.dump(self.sv, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Choose a directory to save data"
    parser.add_argument("-o", "--output", help=help_, type=str, default="output/")

    # classifier settings
    parser.add_argument("-wc", "--classifier", help="Weights for classifier", type=str, required=False, default="models/cnn_256_dct_2.h5") 
    parser.add_argument("-r", "--res", help="resolution to decode JPEG2000 files at (0 is highest)", type=int, default=2)
    # [1/1, 1/4, 1/16/, 1/64]

    # segmentation settings
    parser.add_argument("-s", "--size", help="Segmentation input size [px] -> S x S", type=int, default=512)
    parser.add_argument("-ws", "--segmentor", help="Weights for segmentation model", type=str, required=False, default="unet_512_0_normalize_ft_weights.h5")

    # using for scraping from disk or from file
    parser.add_argument('-u', '--base_url', action='store', type=str, help='seed URL to search for links on',
                        default = '/scratch/mdap/data')
                        #default="https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/")

    parser.add_argument('--threads', default=16, type=int, help='Number of threads to use when decoding JPEG2000 files')
    parser.add_argument("--gpu", default=0, type=int, help='specify which gpu to use')
    parser.add_argument("--order", default=1, type=int, help='specify the image list order (1=forward, -1=reverse)')
    parser.add_argument("--image", default=None, type=str, help='specify a single image to process')
    parser.add_argument("--reprocess", default=False, action="store_true", help="reprocess data and create outputs")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    glymur.set_option('lib.num_threads', args.threads)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the last GPU
        try:
            if args.gpu > len(gpus):
                raise(f"gpu number {args.gpu_num} not supported")
            tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


    # load state data
    if os.path.exists(os.path.join(args.output,"statevector.json")):
        state_data = json.load(open(os.path.join(args.output,"statevector.json"),"r"))
        print("state vector loaded")
        print(f"flagged images: {len(state_data['data'])}")
        print(f"empty images: {len(state_data['empty'])}")
    else:
        state_data = {
            'settings':{
                'size':args.size,
                'res':args.res,
                'output':args.output,
                'classifier':args.classifier,
                'segmentor':args.segmentor,
                'classes':['brain coral','background']
            },
            'data':{},
            'empty':{}
        }

    # override settings with arg inputs
    state_data['settings']['res'] = args.res
    state_data['settings']['segmentor'] = args.segmentor
    state_data['settings']['classifier'] = args.classifier
    state_data['settings']['size'] = args.size
    state_data['settings']['output'] = args.output
    state_data['settings']['classes'] = ['brain coral','background'] 
    # must be in the same order as output channels

    # choose online or local generator
    if 'https' in args.base_url:
        hiriseGenerator = hiriseLinkGenerator
    elif 'scratch' in args.base_url:
        hiriseGenerator = hiriseScratchGenerator
    else:
        hiriseGenerator = hiriseFileGenerator

    # run
    mars_pl = pipeline(state_data, hiriseGenerator(args.base_url, order=args.order))
    running = True
    while running:
        try:
            mars_pl.run(reprocess=args.reprocess) # TODO pass image from arguments
            running = False
        except KeyboardInterrupt:
            break
        except Exception as err:
            print(err)
    
    # python pipeline.py -s 512 -ws models/unet_512_0_normalize_ft_weights.h5 -o output_512_normalize/ --gpu 6
