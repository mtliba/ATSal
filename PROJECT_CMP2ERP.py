import numpy as np
from PIL import Image
from projection import projection_methods
import os

'''
use this code to project entire dataset of cube to ERP 
just specify you path in line 12 and 13
'''
stored_path= './final_equi'
in_path = './projected/'
os.mkdir(stored_path)

def CMP_2ERP(in_path,stored_path) :
    for pa in os.listdir(in_path):
        images = [img for img in os.listdir(in_path+pa) if img.endswith(".png")]
        images.sort()
        face={'0':'F', '1':'R', '2':'B', '3':'L', '4':'U', '5':'D'}
        img={}
        print('start inverse project of :',pa)
        for m in images:
            for f in face :
                if not os.path.exists(os.path.join(stored_path, pa,m)):
                    a = np.array((Image.open(os.path.join(in_path,pa, f,m))).resize((256,256)))
                    img[face[f]]=np.array(a)
            if not os.path.exists(os.path.join(stored_path, pa)):
                    os.mkdir(os.path.join(stored_path, pa))
            if not os.path.exists(os.path.join(stored_path, pa,m)):
                    out = methods.c2e(img, h=1024, w=2048, mode='bilinear',cube_format='dict')
                    Image.fromarray(out.astype(np.uint8)).save(os.path.join(stored_path, pa,m))
                    print('complete '+pa+'/'+m)
            else : 
                    print('done',pa+'/'+m)

if __name__ == "__main__":
    
    CMP_2ERP(in_path,stored_path)