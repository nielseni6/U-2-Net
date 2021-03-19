#%matplotlib inline
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import matplotlib.pyplot as plt
# import torch.optim as optim
from matplotlib import cm
from matplotlib.cm import get_cmap, register_cmap
import matplotlib.colors as mcolors

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def imshow(img):
    img = img     # unnormalize
    npimg = img.detach().numpy()
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
#    plt.imshow(tpimg, cmap=cmap)

def VisualizeImageGrayscale(image_3d):
  r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
  """
  vmin = torch.min(image_3d)
  image_2d = image_3d - vmin
  vmax = torch.max(image_2d)
  return (image_2d / vmax)

def GetSmoothedMask(
  x_value, label, stdev_spread=.15, nsamples=25,
  magnitude=True):
    x_np = x_value.cpu().numpy()
    stdev = stdev_spread * (np.max(x_np) - np.min(x_np))
    
    total_gradients = torch.tensor(np.zeros_like(x_value.cpu()))
    for i in range(nsamples):
        noise = np.random.normal(0, stdev, x_value.shape)
        x_plus_noise = x_np + noise
        x_noise_tensor = torch.tensor(x_plus_noise, dtype = torch.float32)
        
        gradient = returnGradPred(x_noise_tensor.cuda(), label)
        
        if magnitude:
            total_gradients += abs(gradient.cpu())
        else:
            total_gradients += gradient.cpu()
    
    return total_gradients / nsamples

bce_loss = nn.BCELoss(size_average=True, reduce = False)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss

def returnGradPred(img,seg):

    img.requires_grad_(True)
    d0,d1,d2,d3,d4,d5,d6 = net(img)
    label = torch.tensor(seg)
    if (torch.cuda.is_available()):
        label = label.cuda()
    loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
    
    loss_dot = torch.tensordot(loss, seg, dims=2)
    loss_d2 = torch.mean(loss_dot)

    loss_d2.backward()
    
    Sc_dx = img.grad

    return Sc_dx#, pred

if __name__ == "__main__":
    # --------- 1. get image path and name ---------
    model_name='u2netp'#u2net



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    criterion = torch.nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        label_test = data_test['label']
        label_test = label_test.type(torch.FloatTensor)
        print(label_test.shape)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        # inputs_test_gray = torch.tensor([inputs_test[:,0,:,:]
        #                                  +inputs_test[:,1,:,:]
        #                                  +inputs_test[:,2,:,:]])
        
        #phi_c = sum(pred)/(pred.shape[1]*pred.shape[2])
        #phi_c = torch.tensor([[phi_c.clone().detach().cpu().numpy()]]).cuda()
        pred_ = torch.tensor([pred.clone().detach().cpu().numpy()]).cuda()
        prob = pred_
        t = Variable(torch.Tensor([0.5])).cuda()  # threshold
        class_sel = (prob > t).float() * 1
        class_sel_i = 1 - class_sel
        #print(phi_c.shape)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
#        save_output(img_name_list[i_test],pred,prediction_dir)
        grad_map = returnGradPred(inputs_test.clone().detach(),class_sel)
        vanilla_grad = grad_map.clone().detach().cpu()
        vanilla_grad_sq = abs(vanilla_grad)
        
        grad_map_i = returnGradPred(inputs_test.clone().detach(),class_sel_i)
        vanilla_grad_i = grad_map_i.clone().detach().cpu()
        vanilla_grad_sq_i = abs(vanilla_grad_i)
#        smoothgrad = GetSmoothedMask(inputs_test.clone().detach(),class_sel,magnitude=False)
#        smoothgrad_sq = GetSmoothedMask(inputs_test.clone().detach(),class_sel,magnitude=True)
#        
#        smoothgrad_i = GetSmoothedMask(inputs_test.clone().detach(),class_sel_i,magnitude=False)
#        smoothgrad_sq_i = GetSmoothedMask(inputs_test.clone().detach(),class_sel_i,magnitude=True)

        fig=plt.figure(figsize=(30, 20))
        length, width = 1, 6
        fig.add_subplot(length, width, 1).set_title('Input Image')
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(inputs_test.clone().detach().cpu())))
        plt.axis('off')
        
        fig.add_subplot(length, width, 2).set_title('Predicted Segmentation')
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(class_sel.clone().detach().cpu()))) # class_sel = Binarize(pred)
        plt.axis('off')
        
        # Show attribution for all pixels correspoding to the object class
        fig.add_subplot(length, width, 3).set_title('VanillaGrad classification 1') 
        vanilla_grad = vanilla_grad[0][0] + vanilla_grad[0][1] + vanilla_grad[0][2]
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(vanilla_grad)))
        plt.axis('off')
        
        fig.add_subplot(length, width, 4).set_title('VanillaGrad^2 classification 1')
        vanilla_grad_sq = vanilla_grad_sq[0][0] + vanilla_grad_sq[0][1] + vanilla_grad_sq[0][2]
#        cmap = cm.jet_r(smoothgrad_sq.cpu().numpy())[...,None]#[...,0]+cm.jet_r(smoothgrad_sq)[...,1]+cm.jet_r(smoothgrad_sq)[...,2]+cm.jet_r(smoothgrad_sq)[...,3]
#        smoothgrad_sq = torch.tensor(cmap[...,0])
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(vanilla_grad_sq)))
        plt.axis('off')
        
        # Show attribution for all pixels correspoding to the background class
        fig.add_subplot(length, width, 5).set_title('VanillaGrad classification 0')
        vanilla_grad_i = vanilla_grad_i[0][0] + vanilla_grad_i[0][1] + vanilla_grad_i[0][2]
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(vanilla_grad_i)))
        plt.axis('off')
        
        fig.add_subplot(length, width, 6).set_title('VanillaGrad^2 classification 0')
        vanilla_grad_sq_i = vanilla_grad_sq_i[0][0] + vanilla_grad_sq_i[0][1] + vanilla_grad_sq_i[0][2]
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(vanilla_grad_sq_i)))
        plt.axis('off')
        
        
        plt.show()
        #save_output(img_name_list[i_test]+'_grad',grad_map,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7