import os
import urllib.request
import gzip
import shutil
import codecs
import numpy
from typing import List

def download_archives(
    urls: List[str],
    datapath: str
) -> None:

    if not os.path.exists(datapath):
        os.makedirs(datapath)

    for url in urls:
        filename = url.split('/')[-1]
        if not os.path.exists(datapath + filename):
            urllib.request.urlretrieve (url, datapath + filename)
     
def extract_files(
    datapath: str,
    remove_archives: bool = True
) -> None:

    files = os.listdir(datapath)

    for file in files:

        if file.endswith('gz'):

            with gzip.open(datapath+file, 'rb') as f_in:
                with open(datapath+file.split('.')[0], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            if remove_archives:
                os.remove(datapath + file)

def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)

def save_ndarrays(
    datapath: str,
    remove_ubytes: bool = True
) -> None:

    files = os.listdir(datapath)

    data_dict = {}

    for file in files:

        if file.endswith('ubyte'):

            with open (datapath + file,'rb') as f:

                data = f.read()
                type = get_int(data[:4])   # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
                length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)

                if type == 2051:

                    category = 'images'
                    num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                    num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                    parsed = numpy.frombuffer(data,dtype = numpy.uint8, offset = 16)  # READ THE PIXEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length,num_rows,num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]       

                elif type == 2049:

                    category = 'labels'
                    parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8) # READ THE LABEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]    

                if length == 10000:
                    set = 'test'

                elif length == 60000:
                    set = 'train'

                data_dict[set + '_' + category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY   

                path = os.path.join('./data/mnist', set + '_' + category)
                numpy.save(path, parsed) 

                if remove_ubytes:
                    os.remove(datapath + file)

def main(
    datapath: str,
    urls: List[str]
) -> None:

    print('PREPARING MNSIT DATASET...')

    download_archives(urls=urls, datapath=datapath)
    extract_files(datapath=datapath, remove_archives=True)
    save_ndarrays(datapath=datapath, remove_ubytes=True)

    print('MNIST DATASET READY')
    

if __name__ == '__main__':

    datapath = './data/mnist/'  

    URLS = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    main(urls=URLS, datapath=datapath)