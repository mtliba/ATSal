import cv2 
import os 

my_path='./DATASET/Clips'
out_path= './Equi_frames'
os.mkdir(out_path)
for vid in os.listdir(my_path):
    # Read the video from specified path if your videos are in other format change extention .mp4
    cam = cv2.VideoCapture(os.path.join(my_path,vid)) 
    os.mkdir('./Equi_frames/'+vid[:-4])
    # frame 
    currentframe = 0
    names = [] 
    while(True): 
        
        # reading from frame 
        ret,frame = cam.read()
        if ret: 
            a = 4-len(str(currentframe))
            a = '0'*a + str(currentframe) 
            # if video is still left continue creating images 
            a= a+'.png'
            name = os.path.join(out_path,vid[:-4],a) 
            names.append(name)
            print(name)
            # writing the extracted images 
            cv2.imwrite(name, frame) 

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    print(names)
    cv2.destroyAllWindows()