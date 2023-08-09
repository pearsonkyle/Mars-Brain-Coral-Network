import os
import glob
import requests
import urllib.request

from bs4 import BeautifulSoup as bs

from urllib.parse import urljoin

def download_image(iname, disk):
    """
    Download HiRISE image from uahirise.org

    Parameters
    ----------
    iname : str
        HiRISE image name (e.g. PSP_010502_2090)

    disk : str
        path to save image

    Returns
    -------
    bool
        True if image was downloaded, False otherwise
    """
    # search HiRISE website for image url
    img_url = get_hirise_image(iname)
    
    # check if image is already on disk
    fname = os.path.join(disk, img_url.split('/')[-1])
    if os.path.exists(fname):
        print(f"image found on disk: {fname}")
    else:

        # get file info
        try:
            site = urllib.request.urlopen(img_url)
        except:
            print(f"URL could not be reached: {img_url}")
            return False

        # print some info
        fsize = site.length/1024/1024
        print(f"downloading {fsize:.1f} mb from:")
        print(f" {img_url}")

        # try to download image
        try:
            urllib.request.urlretrieve(img_url, file_name)
            return True
        except:
            try:
                urllib.request.urlretrieve(img_url, file_name)
                return True
            except:
                return False


def get_hirise_image(fname):
    """
    Get HiRISE image url from uahirise.org
    
    Parameters
    ----------
    fname : str
        HiRISE image name (e.g. PSP_010502_2090)
    
    Returns
    -------
    str
        url to HiRISE image
    """
    base = 'https://www.uahirise.org/'
    url = urljoin(base, fname)

    # search for all links on page
    links = find_links(url)

    # filter links based on extension
    for j, link in enumerate(links):

        # anaglyph is the stereo pair
        if "RED.JP2" in link and "ANAGLYPH" not in link:
            return link
    return ""

def find_links(URL, pattern="", full=True):
    """
    Find all links on a webpage

    Parameters
    ----------
    URL : str
        url to webpage
    
    pattern : str
        filter links based on pattern
    
    full : bool
        return full url or not
    
    Returns
    -------
    list
        list of links with pattern
    """
    # get webpage
    r = requests.get(URL)

    # parse webpage
    soup = bs(r.content, 'lxml')
    urls = []

    # find all links
    if pattern:
        items = soup.select('[href*="{}"]'.format(pattern))
    else:
        items = soup.select('[href]')

    # append links to list and return
    for item in items:
        if full:
            urls.append(urljoin(URL,item['href']))
        else:
            urls.append(item['href'])
    return list(set(urls))

def hiriseLinkGenerator(base_url, pattern="RED.QLOOK.JP2", order=1):
    # for finding files online
    links = find_links(base_url)
    for l,link in enumerate(links[::order]):
        olinks = find_links(link)
        for ol, olink in enumerate(olinks[::order]):
            dlinks = find_links(olink, pattern=pattern)
            for dl, dlink in enumerate(dlinks):
                yield dlink

def find_files(base_dir, pattern="*/"):
    # for scraping files online
    return glob.glob(os.path.join(base_dir,pattern))

def hiriseFileGenerator(base_dir, pattern="*RED*.JP2", order=1):
    # for finding files on disk
    links = find_files(base_dir)
    for l,link in enumerate(links[::order]):
        olinks = find_files(link)
        for ol, olink in enumerate(olinks[::order]):
            dlinks = find_files(olink, pattern)
            for dl, dlink in enumerate(dlinks):
                yield dlink

def hiriseScratchGenerator(base_dir, pattern="*RED*.JP2", order=1):
    # for finding files on scratch volume
    links = find_files(base_dir)
    for l,link in enumerate(links[::order]):
        olinks = find_files(link, pattern)
        for ol, olink in enumerate(olinks[::order]):
            yield olink

if __name__ == "__main__":

    # count the number of files online
    #elinks = [link for link in hiriseLinkGenerator("https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/")]
    
    # count the number of files on disk
    elinks = [f for f in hiriseFileGenerator("/projects/mdap/HiRISE/ESP")]
    plinks = [f for f in hiriseFileGenerator("/projects/mdap/HiRISE/PSP")]
    tlinks = [f for f in hiriseFileGenerator("/gpfsm/ccds01/nobackup/temp/aaltinok/HiRISE/TRA/")]

    print(f"ESP: {len(elinks)}")
    print(f"PSP: {len(plinks)}")
    print(f"TRA: {len(tlinks)}")