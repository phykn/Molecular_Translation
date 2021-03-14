import os
import sys
import re
import random
import datetime
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool 
from functools import partial
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class CFG:
    max_smiles_length = 150
    min_atom_number = 4
    max_image_size = 300
    padding = 50
    output_file_name = 'dataset.csv'

def get_smiles_list(SMILES_DATA_FILE_PATH, data_size=5000, max_length=150):
    # DataFile Source: 
    # https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SID.gz 
    print('Loading: %s'%(SMILES_DATA_FILE_PATH))    
    with open(SMILES_DATA_FILE_PATH, 'r') as f:
        samples = f.readlines()
    random.shuffle(samples)

    SMILES_LIST = []
    i = 0
    for sample in samples:
        SMILES = re.split('\s+', sample)
        SMILES = SMILES[1]
        if len(SMILES) <= max_length:
            i += 1
            SMILES_LIST.append(SMILES)

        if i >= data_size:
            break
    del samples

    return SMILES_LIST

def generate_data(SMILES, max_image_size=300, padding=20):
    # Set Molecule
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return None, None, None
    elif mol.GetNumAtoms() < CFG.min_atom_number:
        return None, None, None
    else:
        InChI = Chem.MolToInchi(mol)
        
        # Draw Image
        img = Draw.MolToImage(mol, size=(1000, 1000))
        img = np.array(img)
        
        # Cut Marginal background
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        
        xy, wh = np.where(img > 0)
        x = np.maximum(0, np.min(xy)-padding)
        y = np.max(xy)+padding
        w = np.maximum(0, np.min(wh)-padding)
        h = np.max(wh)+padding 
        
        img = img[x:y, w:h]

        # Resize Image
        hight, width = img.shape
        hight_ratio = max_image_size/hight
        width_ratio = max_image_size/width
        ratio = np.min([hight_ratio, width_ratio])
        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        
        # Color inversion
        img = 255 - img
        return SMILES, InChI, img

def GenSingleData(SMILES, max_image_size=300, padding=20, image_folder='/'):
    now = datetime.datetime.now().timestamp()
    now = str(now).replace('.', '')
    file_name = now + str(random.random())[3:7] + '.png'
    image_path = os.path.join(image_folder, file_name)

    SMILES, InChI, img = generate_data(SMILES, max_image_size=max_image_size, padding=padding)
    if img is None:
        df = {}
        df['file_name'] = [None]
        df['SMILES'] = [None]
        df['InChI'] = [None]
        df = pd.DataFrame(df)

    else:
        # Save Image
        cv2.imwrite(image_path, img)

        df = {}
        df['file_name'] = [file_name]
        df['SMILES'] = [SMILES]
        df['InChI'] = [InChI]
        df = pd.DataFrame(df)
        return df

if __name__ == '__main__':
    # argv_1 = SMILES_DATA_FILE_PATH
    # argv_2 = data_size

    # Arguments    
    SMILES_DATA_FILE_PATH = str(sys.argv[1])
    data_size = int(sys.argv[2])

    # Make Folder
    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    image_folder = os.path.join(data_folder, 'images')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Get SMILES LIST  
    SMILES_LIST = get_smiles_list(SMILES_DATA_FILE_PATH, data_size=data_size, max_length=CFG.max_smiles_length) 

    # Process
    func_fixed = partial(GenSingleData, max_image_size=CFG.max_image_size, padding=CFG.padding, image_folder=image_folder)
    pool = Pool(processes=multiprocessing.cpu_count()) 
    df = list(
        tqdm(
            pool.imap(func_fixed, SMILES_LIST), 
            total=len(SMILES_LIST)
            )
        )
    pool.close()
    pool.join()      
    df = pd.concat(df, axis=0)

    # Save Map File
    df.to_csv(os.path.join(data_folder, CFG.output_file_name), encoding='utf-8', index=False)