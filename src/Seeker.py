import cv2
import requests 
import pandas as pd
from pathlib import PurePath 
import os
import urllib.request


class Seeker():

    def __init__(self):
        print("opening image data ...")
        self.image_data = pd.read_csv(PurePath(os.getcwd(), 'FEC_dataset','FEC_dataset','faceexp-comparison-data-train-public.csv'),  header=None, error_bad_lines=False)
        self.downloaded = set([]) # saves all downloaded images at a time
        self.path_map = {}
        
    def seek_images(self):
        
        for i in range(self.image_data.shape[0]):
            
            sample = ('','','')
            sample = (self.image_data.loc[i].loc[0], self.image_data.loc[i].loc[5], self.image_data.loc[i].loc[10])

            # download images 
            for i in range(2):
                if sample[i] not in self.downloaded:
                    self.downloadImage(sample[i])
            
    def downloadImage(self, url):
        print(f"Downloading Image at |-> {url}")
        filename = url[url.rfind("/")+1:]

        r = requests.get(url, allow_redirects=True)
        open(f'{os.getcwd()}/IMG/{filename}', 'wb').write(r.content)
        print("Finished Downloading Image . . .")
        self.downloaded.add(url)

