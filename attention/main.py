import cv2
import os
import datetime
import numpy as np
import pickle
import torch
from torch.utils import data
from torchvision import transforms, utils
from torch import nn 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from data_loader import Static_dataset
from model import Sal_based_Attention_module,LOSS
mean = lambda x : sum(x)/len(x)

def train(train_loader, model, criterion, optimizer, epoch, state = 'model'):


    # Switch to train mode
    model.train()

    print("Now commencing {} epoch {}".format(state,epoch))

    losses = []
    nss_batch = []
    for  j,batch in enumerate(train_loader):
     
        start = datetime.datetime.now().replace(microsecond=0)
        loss = torch.tensor(0)
        nss_score = 0
        frame , gtruth = batch
        for i in tqdm(range(frame.size()[0])):
          #try:
            
            saliency_map ,fiaxtion_module = model(frame[i].unsqueeze(0))
            saliency_map    = saliency_map.squeeze(0)
            fiaxtion_module = fiaxtion_module.squeeze(0)

            total_loss , nss = criterion(saliency_map , gtruth[i] , fiaxtion_module)
            nss_score = nss_score +nss.item()
            #print("last loss ",last.data)
            #print("attention loss ",attention.data)
            loss = loss.item() + total_loss
            #loss = loss + attention
          #except:
            #print('error')
            #continue
        
            if j%5==0 and i==3:

                post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                utils.save_image(post_process_saliency_map, "./map/smap{}_batch{}_epoch{}_{}.png".format(i,j, epoch,state))
                if epoch==0:
                    utils.save_image(gtruth[0], "./map/{}_batch{}_grounftruth.png".format(i,j))
 
        loss.backward()
 
        optimizer.step()
        nss_score =nss_score/(frame.size()[0])
        end = datetime.datetime.now().replace(microsecond=0)
        print('\n\tEpoch: {}\on {}\t Batch: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch,state , j, (loss.data)/frame.size()[0], end-start))
        losses.append(loss.data/frame.size()[0])
        nss_batch.append(nss_score)
    return (mean(losses), mean(nss_batch)) 
def validate(val_loader, model, criterion, epoch):

  
    # Switch to train mode
    model.eval()

    

    losses = []
    nss_batch = []

    for  j,batch in enumerate(val_loader):
        print("load batch....")
        start = datetime.datetime.now().replace(microsecond=0)
        loss = 0
        nss_score = 0
        frame , gtruth  = batch
        for i in range(frame.size()[0]):
            #inpt = frame[i].unsqueeze(0).cuda()
            with torch.no_grad():
                saliency_map ,fiaxtion_module = model(frame[i].unsqueeze(0))
                saliency_map    = saliency_map.squeeze(0)
                fiaxtion_module = fiaxtion_module.squeeze(0)
                total_loss , nss = criterion(saliency_map , gtruth[i] , fiaxtion_module )
                
            nss_score = nss_score +nss
            #print("last loss ",last.data)
            #print("attention loss ",attention.data)
            loss = loss + total_loss

            #loss = loss + attention
        nss_score =nss_score/(frame.size()[0])
        end = datetime.datetime.now().replace(microsecond=0)
        print('\n\tEpoch: {}\Batch: {}\t Validation Loss: {}\t Time elapsed: {}\t'.format(epoch, j, loss.data/frame.size()[0], end-start))
        losses.append(loss.data/frame.size()[0])
        nss_batch.append(nss_score)
        

    return (mean(losses), mean(nss_batch))

def adjust_learning_rate(optimizer,learning_rate, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


model = Sal_based_Attention_module()
wrihgt = torch.load('./initial.pt')
model.load_state_dict(wrihgt)
model.cuda()

val_perc = 0.153

train_set = Static_dataset(
    root_path = './training_set/',
    load_gt = True,
    number_of_frames = 1839,
    resolution = (640, 320),
    val_perc = 0.153,
    split = "train")
print("Size of train set is {}".format(len(train_set)))
train_loader = data.DataLoader(train_set, batch_size= 128 ,shuffle=True ,num_workers= 0,drop_last=True)
valid_set = Static_dataset(
    root_path = './validation_set/',
    load_gt = True,
    number_of_frames = 300,
    resolution = (640, 320),
    val_perc = 0.153,
    split = "validation")
print("Size of validation set is {}".format(len(valid_set)))
val_loader = data.DataLoader(valid_set, batch_size= 10 ,num_workers= 0,drop_last=True)
criterion = LOSS()
optimizer_1 = torch.optim.Adam([
        {'params':model.parameters() , 'lr':1e-4,'weight_decay': 1e-4}])
optimizer_2 = torch.optim.Adam([
        {'params':model.attention_module.parameters() , 'lr':1e-3,'weight_decay': 1e-4}])


cudnn.benchmark = True 
criterion = criterion.cuda()
#Traning #
starting_time = datetime.datetime.now().replace(microsecond=0)
print("Training started at : {}".format(starting_time))
train_losses =[]
nss_accuracy =[]
nss_validate=[]
val_losses = []
start_epoch = 0
epochs = 60
plot_every = 1

model = model.cuda()

for epoch in tqdm(range(epochs)):
#for epoch in range(start_epoch, epochs+1):

        print('**** new epoch ****')
        # train for one epoch
        if  epoch % 2 == 0 and epoch < 40:
            adjust_learning_rate(optimizer_2,1e-3, epoch, decay_rate=0.1)
            train_loss_attention, nssaccuracy = train(train_loader, model, criterion, optimizer_2, epoch,state = 'attention')
            print("\t attention Epoch {}/{} done with train loss {} and nss score {}\n".format(epoch, epochs, train_loss_attention ,nssaccuracy))
        adjust_learning_rate(optimizer_1,1e-4, epoch, decay_rate=0.1)
        train_loss, nssaccuracy = train(train_loader, model, criterion, optimizer_1, epoch)

        print("Epoch {}/{} done with train loss {} and nss score {}\n".format(epoch, epochs, train_loss, nssaccuracy))


        if val_perc > 0:
            print("Running validation..")
            val_loss ,nssvalidate = validate(val_loader, model, criterion, epoch)
            print("Validation loss: {}\t  validation nss {}".format(val_loss,nssvalidate))

        if epoch % plot_every == 0:
            train_losses.append(train_loss.cpu())
            nss_accuracy.append(nssaccuracy)
            if val_perc > 0:
                val_losses.append(val_loss.cpu())
                nss_validate.append(nssvalidate.cpu())
        print("\n epoch finished at : {} \n Now saving..".format(  datetime.datetime.now().replace(microsecond=0)))
        if epoch%2==0 and epoch!=0:
              torch.save({
                  'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'optimizer1_state_dict': optimizer_1.state_dict(),
                'optimizer2_state_dict': optimizer_2.state_dict()
                  }, "./weight/train1_wheight_epoch_{}.pt".format(epoch))


        model = model.cuda()
        to_plot = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'nss_accuracy':nss_accuracy,
            'nss_validate':nss_validate
            }
        with open('./new_train_plot.pkl', 'wb') as handle:
            pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Training of new model started at {} and finished at : {} \n ".format( starting_time, datetime.datetime.now().replace(microsecond=0)))
torch.cuda.empty_cache()

   

