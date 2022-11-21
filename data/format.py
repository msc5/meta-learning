import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
import shutil
import zipfile
import glob
import os

# Omniglot

def format_omniglot(git_path):
    print('Unzipping Omniglot files from ', git_path)
    path = os.path.join(git_path, 'python')
    zips = os.path.join(path, '*.zip')
    target = os.path.join('datasets', 'omniglot')
    if os.path.exists(target): shutil.rmtree(target)
    for f in list(glob.glob(zips)):
        with zipfile.ZipFile(f, 'r') as zf:
            desc = f'    ==> {os.path.basename(f):<40}'
            for member in tqdm(
                    zf.infolist(),
                    desc=desc,
                    bar_format='{desc:30}|{bar:20}|     {rate_fmt}',
                ):
                try:
                    zf.extract(member, target)
                except zipfile.error as e:
                    pass
        
format_omniglot('../omniglot')
