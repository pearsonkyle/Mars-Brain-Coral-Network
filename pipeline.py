import os
import time
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from itertools import cycle
import requests

from fake_useragent import UserAgent

from utils import hiriseFileGenerator, hiriseLinkGenerator

def parse_args():  
    parser = argparse.ArgumentParser()

    help_ = "Choose a directory to save data"
    parser.add_argument("-o", "--output", help=help_, type=str, default="D:/MDAP/output")

    # classifier settings
    parser.add_argument("-wc", "--classifier", help="Weights for classifier", type=str, required=False, default="models/mobilenet_128_spatial_2_1.h5")

    # segmentation settings
    parser.add_argument("-ws", "--segmentor", help="Weights for segmentation model", type=str, required=False, default="models/unet_512_0_normalize_ft_weights.h5")

    # using for scraping from disk or from file
    parser.add_argument('-u', '--base_url', action='store', type=str, help='seed URL to search for links on',
                        #default = '/scratch/mdap/data') # if you have files on disk
                        default="https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/")

    parser.add_argument('--threads', default=8, type=int, help='Number of threads to use when decoding JPEG2000 files')
    parser.add_argument('--vpn', action='store_true', help='toggle vpn on/off')
    return parser.parse_args()

# helper functions for cleaning data from label file
clean = lambda x: x.split('=')[1].strip('"\n\r').strip()
clean_num = lambda x: float(clean(x).split(" ")[0])

# make user agent for vpn
ua = UserAgent(browsers=['edge', 'chrome', 'firefox', 'safari'], os=['win', 'mac', 'linux'])

# cycle through vpn commands to prevent rate limiting
vpn_commands = cycle([
    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "Los Angeles"',
    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "United States"',
    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "San Francisco"',
    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "Phoenix"',
    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "Salt Lake City"',
#    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "Denver"',
#    r'"C:\Program Files\NordVPN\NordVPN.exe" -c -g "Dallas"',
    r'"C:\Program Files\NordVPN\NordVPN.exe" -d',
])

def main():

    # parse command line arguments
    args = parse_args()
    
    # make output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # load state data
    if os.path.exists(os.path.join(args.output,"statevector.json")):
        state_data = json.load(open(os.path.join(args.output,"statevector.json"),"r"))
        print("state vector loaded")
        print(f"flagged images: {len(state_data['data'])}")
        print(f"empty images: {len(state_data['empty'])}")
    else:
        state_data = {
            'settings':{
                'output':args.output,
                'classifier':args.classifier,
                'segmentor':args.segmentor,
                'classes':['brain coral','background']
            },
            'data':{},
            'empty':{}
        }

    # choose online or local generator
    if 'https' in args.base_url:
        hiriseGenerator = hiriseLinkGenerator(args.base_url)
    else:
        hiriseGenerator = hiriseFileGenerator(args.base_url)

    # random integer between 20-40 for vpn stuff
    rand_int = np.random.randint(50,100)

    # loop over generator
    for i, img_url in enumerate(hiriseGenerator):

        # check sv if image has been processed
        img_name = os.path.basename(img_url).split('.')[0]

        if img_name in state_data['data'] or img_name in state_data['empty']:
            print("skipping image because it has been processed")
            continue

        # cycle vpn every 20-40 images
        if args.vpn and i%rand_int == 0:
            # iterate through vpn commands
            vpn_command = next(vpn_commands)
            print(vpn_command)
            p = Popen(vpn_command, shell=True, stdout=PIPE, stderr=PIPE)
            #stdout, stderr = p.communicate()
            rand_int = np.random.randint(15,30)
            time.sleep(5)

        # create new directory in output
        dirname = os.path.join(state_data['settings']['output'], img_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        # download image to disk
        file_name = os.path.join(dirname, f"{img_name}.JP2")

        print(f"downloading from:")
        print(f" {img_url}")

        # try to download image
        try:
            site = requests.get(img_url, headers={'User-Agent': ua.random})
            with open(file_name, 'wb') as f:
                f.write(site.content)
        except Exception as e:
            print(e)
            print("image could not be downloaded")
            time.sleep(10)
            continue
    
        # download the LBL file
        lbl_url = img_url.replace('.JP2','.LBL')
        lbl_name = os.path.join(dirname, f"{img_name}.LBL")
        try:
            site = requests.get(lbl_url, headers={'User-Agent': ua.random})
            with open(lbl_name, 'wb') as f:
                f.write(site.content)
        except Exception as e:
            print(e)
            print("label file could not be downloaded")
            continue

        # parse label data
        with open(lbl_name, "r") as flabel:
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

        ldata['lon']=0.5*(maxlong+minlong)
        ldata['lat']=0.5*(maxlat+minlat)

        # spawn subprocess to call eval_classifier.py
        p = Popen(['python', 'eval_classifier.py', '-i', file_name, '-w', args.classifier, '--threads', str(args.threads)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        #print(stdout)
        #print(stderr)

        # read directory for output
        maskfile = file_name.replace(".JP2",  "_classifier_mask.png")

        # load and sum pixels
        mask_img = plt.imread(maskfile)
        mask = mask_img > 0

        # if less than 4 pixels are coral, delete directory
        if mask.sum() <= 4:
            # delete directory and all contents
            shutil.rmtree(dirname)
            state_data['empty'][img_name] = {
                'url':img_url, 
                'mask_sum': int(mask.sum()),
                'lat':ldata['lat'],
                'lon':ldata['lon'],
                'scale':ldata['resolution'],
                'title':ldata['title'],
                'local_time':ldata['local_time'],
                'acq_date':ldata['acq_date']
            }
        else:
            state_data['data'][img_name] = {
                'url':img_url, 
                'mask_sum': int(mask.sum()),
                'lat':ldata['lat'],
                'lon':ldata['lon'],
                'scale':ldata['resolution'],
                'title':ldata['title'],
                'local_time':ldata['local_time'],
                'acq_date':ldata['acq_date'] 
            }

            # run the segmentation
            p = Popen(['python', 'eval_segmentation.py', '-i', file_name, '-w', args.segmentor, '--threads', str(args.threads), '--fast', '--lowmem'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            #print(stdout)
            #print(stderr)

            # read directory for output
            maskfile = file_name.replace(".JP2",  "_segmentation_mask.png")
            
            # load and sum pixels
            mask_img = plt.imread(maskfile)
            mask = mask_img > 0

            # if less than 2 eval windows are empty, delete directory
            if mask.sum() <= 512*512*2:
                # delete directory and all contents
                shutil.rmtree(dirname)

            # save state
            state_data['data'][img_name]['segmentation_mask_sum'] = int(mask.sum())
            state_data['data'][img_name]['segmentation_percent'] = float(mask.sum()) / float(mask.size) * 100

        # clean up
        del mask, mask_img

        # save state
        with open(os.path.join(state_data['settings']['output'], "statevector.json"), 'w') as f:
            json.dump(state_data, f, indent=4)

if __name__ == "__main__":

    while True:
        try:
            main()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            time.sleep(5*60)
            vpn_command = next(vpn_commands)
            p = Popen(vpn_command, shell=True, stdout=PIPE, stderr=PIPE)