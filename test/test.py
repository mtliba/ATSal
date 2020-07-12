import os
import numpy as np
from tqdm import tqdm
import cv2
from model import Sal_based_Attention_module, SalEMA
import projection_methods
import datetime
import time
from PIL import Image
from data_loader import Test_dataset
from sys import argv
import torch
from torchvision import transforms, utils

from torch.utils import data
    

    
    

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def load_model(pt_model, new_model):
    temp = torch.load('./weight/'pt_model+'.pt')['state_dict']
    new_model.load_state_dict(temp)
    return new_model


def test(path_to_vid, path_output_maps, model_to_test=None):

    dataset = Test_dataset(
        root_path = path_to_vid)
    video_name_list = dataset.video_names() 
    loader = data.DataLoader(dataset, batch_size = 1,num_workers = 4,pin_memory = True)

    os.makedirs(path_output_maps, exist_ok=True)
    #print('start prediction for video :',path_to_images)
    #start = datetime.datetime.now().replace(microsecond=0)

    for i, video  in enumerate(loader):

        count = 0
        out_state = {'F':None, 'R':None, 'B':None, 'L':None, 'U':None, 'D':None}
        state = None 
        video_dst = os.path.join(path_output_maps, video_name_list[i])
        if not os.path.exists(video_dst):
            os.makedirs(video_dst, exist_ok=True)    

        for j, (clip,frames) in enumerate(video):
            clip = clip.type(torch.cuda.FloatTensor).transpose(0,1)
            for idx in range(clip.size()[0]):
                start_time = time.time()
                first_stream = ((model_to_test['attention'](clip[idx])).squeeze().cpu().numpy()).reshape(320,640)
                print('first strem complete for image :',count)
                print(frames[idx][0])
                img = np.array(Image.open(frames[idx][0]).resize((320,640)))
                if len(img.shape) == 2 :
                    img = img[..., None]
                out = projection_methods.e2c(img, face_w=256 , mode='bilinear', cube_format='dict')
                out_predict ={}
                for face_key in out:
                    cmp_face = out[face_key].astype(np.float32)
                    state = out_state[face_key]
                    if len(cmp_face.shape) == 2 :
                        cmp_face = cmp_face[..., None]
                    cmp_face -= [103.939, 116.779, 123.68]
                    cmp_face = torch.cuda.FloatTensor(cmp_face)
                    cmp_face = cmp_face.permute(2,0,1)
                    cmp_face = cmp_face.unsqueeze(0)
                    if face_key == 'U' or face_key == 'D':
                        state, out_face = model_to_test['poles'].forward(input_ = cmp_face, prev_state = state)   
                      
                    else :
                        state, out_face = model_to_test['equator'].forward(input_ = cmp_face, prev_state = state)
                
                    state = repackage_hidden(state)
                    out_face = out_face.squeeze()
                    out_face = out_face.cpu().numpy()
                    out_face = out_face*255/out_face.max()
                    out_face = cv2.resize(out_face,(256,256))
                    
                    if len(out_face.shape) == 2 :
                        out_face = out_face[..., None]

                    print('******** predict projected face :',face_key)
                    out_predict[face_key] = np.array(out_face.astype(np.uint8))
                    out_state[face_key]   = state 
                    
                second_stream = (projection_methods.c2e(out_predict, h=320, w=640, mode='bilinear',cube_format='dict')).reshape(320,640)
                second_stream= second_stream/255
                count+=1
                saliency_map = first_stream*second_stream
                print("elapsed time : --- %s ms ---" % (1000*(time.time() - start_time)))
                cv2.imwrite(os.path.join(video_dst,str(count).zfill(4)+'.png'),(255*saliency_map/saliency_map.max()).astype(np.uint8))
                print('complete :',os.path.join(video_dst,str(count).zfill(4)+'.png'))



def main(path_to_vid,output_path):

    assert torch.cuda.is_available(), \
        "CUDA is not available in your machine"
    # Create networks
    att_model = Sal_based_Attention_module()
    salema_copie = SalEMA()
    # load weight
    att_model = load_model("attention", att_model).cuda()
    Poles  = load_model("poles", salema_copie).cuda()
    Equator  = load_model("equator", salema_copie).cuda()
    
    # loop over videos
    print('dataset contains {} videos'.format(len(os.listdir(path_to_vid))))
    print(os.listdir(path_to_vid))
    model = {"attention":att_model,"poles":Poles,"equator":Equator}
    with torch.no_grad():
        test(path_to_vid, output_path,model)

if __name__ == "__main__":
    path_to_vid = argv[1]
    output_path = argv[2]
    main(path_to_vid,output_path)
