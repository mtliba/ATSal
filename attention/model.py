import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import  ReLU

class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)
        
        
    def forward(self, x):
        x = self.pool(x)
        return x


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M_2':
            layers += [Downsample(kernel_size= 2)]
        elif v == 'M_4':
            layers += [Downsample(kernel_size= 4)]
        else:
            conv = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv, ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'global_attention': [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 'M_4', 512, 512, 512],
    'based_AM'        : [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 512, 512, 512]

      }
global_attention = make_conv_layers(cfg['global_attention'])
based_AM = make_conv_layers(cfg['based_AM'])





from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.autograd import Variable
from torch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

#from Encoders import  based_AM

# create pooling layer
class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d(kernel_size)
        
        
    def forward(self, x):
        x = self.pool(x)
        return x

# create unpooling layer 
class Upsample(nn.Module):
    # specify the scale_factor for upsampling 
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

# create Add layer , support backprop
class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        result = torch.ones(tensors[0].shape).cuda()
        for t in tensors:
            result *= t
        return result

# create Multiply layer , supprot backprop
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, tensors):
        result = torch.zeros(tensors[0].shape).cuda()
        for t in tensors:
            result += t
        return result
# reshape vectors layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Sal_based_Attention_module(nn.Module):
    """
    In this model, we take salgan architecture and erase its last maxpooling layer and add attention module above its botellneck than mltiply attention module and salgan bottelneck results together 
    
    """
    def  __init__(self):
        super(Sal_based_Attention_module,self).__init__()

        

        # Create encoder based on VGG16 architecture as pointed on salgan architecture and apply aforementionned changes
        Based_Attention_Module = based_AM
        
        # select only first 5 conv blocks , here we keep same receptive field of VGG 212*212 
        # each neuron on bottelneck will see just (244,244) viewport during sliding  ,
        # input (640,320) , features numbers on bottelneck 40*20*512, exclude last maxpooling of salgan ,receptive
        # features number on AM boottlneck 10*5*128 
        # attentioin moduels receptive field enlarged (676,676)
        self.encoder = torch.nn.Sequential(*Based_Attention_Module)
        self.attention_module = torch.nn.Sequential(*[
            Downsample(kernel_size= 2),
            Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Downsample(kernel_size= 2),
            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
            Upsample(scale_factor=4 , mode='nearest' )

        ])
        
        #self.reshape = Reshape(-1,512,40,20)

        # define decoder based on VGG16 (inverse order and Upsampling layers , chose nearest mode)
        decoder_list=[
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        self.decoder = torch.nn.Sequential(*decoder_list)

        print("Model initialized, Sal_based_Attention_module")
        #rint("architecture len :",str(len(self.Sal_based_Attention_module)))

    def forward(self, input):
        #print(input.shape)
        x = self.encoder(input)
        #print(x.shape)
        y = self.attention_module(x)
        #print('atttention ',y.shape)
        repeted  = y.repeat(1,512,1,1)
        #print('repeted' ,repeted.shape)
        #reshaped = self.reshape(repeted)
        #print('reshaped' ,reshaped.shape)
        product  = x*repeted
        #print('product' ,product.shape)
        added    = x+product
        #print('added' ,added.shape)
        x = self.decoder(added)

        #batch_size_x = x.data.size()[0]
        #print(batch_size_x)
        #spatial_size_x = x.data.size()[2:]
        #print(spatial_size_x)

        #batch_size_y = y.data.size()[0]
        #print(batch_size_y)
        #spatial_size_y = y.data.size()[2:]
        #print(spatial_size_y)
        return x ,y # x is a saliency map at this point,y is the fixation map





import sys

from torch.nn.functional import interpolate 

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)
        trg = trg/torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)



# inn this implementation we consider standatrd normalization 

class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()

    #normalize saliency map
    def stand_normlize(self, x) :
       # res = (x - np.mean(x)) / np.std(x)
       # x should be float tensor 
       return (x - x.mean())/x.std()

    def forward(self, sal_map, fix):
        if sal_map.size() != fix.size():
           sal_map = interpolate(sal_map, size= (fix.size()[1],fix.size()[0]))
           print(sal_map.size())
           print(fix.size())
        # bool tensor 
        fix = fix > 0.1
        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        return sal_map[fix].mean()


# in this implementation we consider standatrd normalization 
class CC_Loss(nn.Module):
    def __init__(self):
        super(CC_Loss, self).__init__()
    def normliz(self ,x) :
        return (x - x.mean())
    def forward(self,saliency_map,gtruth):
         saliency_map = self.normliz(saliency_map)
         gtruth       = self.normliz(gtruth)
         return torch.sum(saliency_map * gtruth) / (torch.sqrt(torch.sum(saliency_map ** 2)) * torch.sqrt(torch.sum(gtruth ** 2)))








class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.NSSLoss = NSSLoss()
        self.CC_Loss = CC_Loss() 
    def forward(self,saliency_map , gtruth , fiaxtion_module):
        if fiaxtion_module.max()==0.0:
            print('output zero')
        attention =0.1*(self.NSSLoss(gtruth,gtruth)- self.NSSLoss(saliency_map,gtruth))
        #cc = 1.0-self.CC_Loss(saliency_map,gtruth)
        
        last =  0.8*self.KLDLoss(saliency_map,gtruth)+0.1*self.KLDLoss(fiaxtion_module,Downsample(16)(gtruth))+attention 
        return  last
