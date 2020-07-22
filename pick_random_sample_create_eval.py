# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:45:41 2020

@author: Spyros
"""

import numpy as np
import os
import os, shutil

source = r"C:\path\to\data\file"
destination = r"\path\to\evaluation_data"

# List all files in dir
files = os.listdir(source)

# Select 0.5 of the files randomly 
random_files = np.random.choice(files, int(len(files)*.2),replace=False)



# Do something with the files
for x in random_files:
    shutil.move(source + x, destination)