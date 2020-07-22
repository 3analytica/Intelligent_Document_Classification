import os
import glob

files = glob.glob('C:/Users/Thanos/Desktop/images/Doc_Pool\*')
for f in files:
    os.remove(f)
    
    
    
    