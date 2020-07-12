import numpy as np
from PIL import Image
from projection import projection_methods
import os


'''
use this code to project entire dataset of ERP to CMP 
just specify you path in line 11 and 12
'''
stored_path= './CMP'
in_path = './dataset/'


def ERP_2CMP(in_path,stored_path):
    for pa in os.listdir(in_path):
        print(pa)
        path = os.path.join(in_path, pa)
        images = [img for img in os.listdir(path)]
        
        images.sort()
        # put in your path parent output folder 
        stored_path = os.path.join(in_path, pa)
        if not os.path.exists(stored_path):
            os.mkdir(stored_path)
            print("created")
        
        for m in images:
            img = np.array(Image.open(os.path.join(path, m)))
            if len(img.shape) == 2 :
                img = img[..., None]
            print(f'####project image {m}####')  
            # change value of face_w the width of cube face it should be x8 and the nearsest possible to you input dim model 
            # let mode 'bilinear' to keep the same type of interpolation with open cv 'cv.resize' function in our saliency model
            # let cube_format to 'dict' in order to project in dictionary format each key assigned to face as value 
            out = methods.e2c(img, face_w=256 , mode='bilinear', cube_format='dict')
            for face_key in out:
                face={'F':'0', 'R':'1', 'B':'2', 'L':'3', 'U':'4', 'D':'5'}
                print(f'---project{face_key }')
                if not os.path.exists(os.path.join(stored_path, face[face_key] )):
                    os.mkdir(os.path.join(stored_path, face[face_key] ))
                Image.fromarray(out[face_key].astype(np.uint8)).save(os.path.join(stored_path ,face[face_key] ,m))

if __name__ == "__main__":

    ERP_2CMP(in_path,stored_path)  