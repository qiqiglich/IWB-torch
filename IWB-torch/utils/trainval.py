import os, random, time, copy, math
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt

from PIL import Image
import sklearn.metrics 
import torch
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
from utils.plot_funcs import *
from utils.center_loss import CenterLoss
import scipy.stats as st


def train_model_stage2(dataloaders, model, lossFunc,
                optimizerW, schedulerW, pgdFunc=None,
                num_epochs=50, model_name= 'model', work_dir='./',
                device='cpu', freqShow=40, clipValue=1, print_each = 1,
                labelnames=None, img_num_per_class=None, args=None):
    trackRecords = {}
    trackRecords['weightNorm'] = []
    trackRecords['acc_test'] = []
    trackRecords['acc_train'] = []
    trackRecords['weights'] = []
    log_filename = os.path.join(work_dir, model_name+'_train.log')    
    since = time.time()
    best_loss = float('inf')
    best_acc = 0.
    best_perClassAcc = 0.0
    
    phaseList = list(dataloaders.keys())
    phaseList.remove('train')
    phaseList = ['train'] + phaseList

    fn = open(log_filename, 'a')
    fn.write('Training parameters lr:{}, bs:{}, ratio:0.05\n'.format(args.lr, args.bs))
    fn.close()

    NC_list = []
    N = sum(img_num_per_class)
    lambda1 = 1
    for Ni in img_num_per_class:
        NC_list.append((math.log((N)/(lambda1 * Ni+1))))

    pdist = nn.PairwiseDistance(p=2)
    for epoch in range(num_epochs):  
        if epoch%print_each==0:
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()


        # Each epoch has a training and validation phase
        for phase in phaseList:
            if epoch%print_each==0:
                print(phase)
            
            predList = np.array([])
            grndList = np.array([])
                
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                schedulerW.step()
                model.train()
            else:
                model.eval()  # Set model to training mode  
              
            running_loss_CE = 0.0
            running_loss = 0.0
            running_acc = 0.0
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0

            for sample in dataloaders[phase]:

                imageList, labelList = sample
                imageList = imageList.to(device)
                labelList = labelList.type(torch.long).view(-1).to(device)

                # zero the parameter gradients
                optimizerW.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    logits, features = model(imageList)

                    NC = torch.tensor(NC_list).repeat(imageList.size(0), 1).to(device)
                    norm_W = F.normalize(model.module.encoder.fc.weight)
                    nm_features = F.normalize(features)
                    HC = F.linear(nm_features, norm_W)
                    HC = ((1 - HC) / 2)
                    P = NC * HC * 0.01

                    W = torch.norm((model.module.encoder.fc.weight), p=2, dim=1).repeat(imageList.size(0), 1)

                    kl_loss = F.kl_div(W.log_softmax(dim=-1), P.softmax(dim=-1), reduction='sum')

                    error = lossFunc(logits, labelList) + kl_loss# + 0.01*info_loss

                    softmaxScores = logits.softmax(dim=1)

                    predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)                  
                    accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                    accRate = (accRate==0).type(torch.float).mean()
                    
                    predList = np.concatenate((predList, predLabel.cpu().numpy()))
                    grndList = np.concatenate((grndList, labelList.cpu().numpy()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        error.backward()
                        optimizerW.step()

                # statistics  
                iterCount += 1
                sampleCount += labelList.size(0)
                running_acc += accRate*labelList.size(0) 
                running_loss_CE += error.item() * labelList.size(0) 
                running_loss = running_loss_CE
                
                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgLoss_CE = running_loss_CE / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                """              
                if iterCount%freqShow==0:
                    print('\t{}/{} loss:{:.3f}, acc:{:.5f}'.
                          format(iterCount, len(dataloaders[phase]), print2screen_avgLoss, 
                                 print2screen_avgAccRate))
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss:{:.3f}, acc:{:.5f}\n'.
                             format( iterCount, len(dataloaders[phase]), print2screen_avgLoss, 
                                    print2screen_avgAccRate))
                    fn.close()
                 """
            epoch_error = print2screen_avgLoss

            confMat = sklearn.metrics.confusion_matrix(grndList, predList)
            # normalize the confusion matrix
            a = confMat.sum(axis=1).reshape((-1,1))
            confMat = confMat / a
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i,i]
            curPerClassAcc /= confMat.shape[0]
            if epoch % print_each==0:
                print('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}'.format(
                    epoch_error, print2screen_avgAccRate, curPerClassAcc))

            fn = open(log_filename,'a')
            fn.write('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}\n'.format(
                epoch_error, print2screen_avgAccRate, curPerClassAcc))
            fn.close()
                
            if phase=='train':
                if pgdFunc: # Projected Gradient Descent 
                    pgdFunc.PGD(model)
                      
                trackRecords['acc_train'].append(curPerClassAcc)
            else:
                trackRecords['acc_test'].append(curPerClassAcc)
                W = model.module.encoder.fc.weight.cpu().clone()
                # W = model.module.fc.weight.cpu().clone()
                tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
                trackRecords['weightNorm'].append(tmp)
                trackRecords['weights'].append(W.detach().cpu().numpy())
                
            if (phase=='val' or phase=='test') and curPerClassAcc>best_perClassAcc: #epoch_loss<best_loss:            
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name+'_best.paramOnly')
                torch.save(model.state_dict(), path_to_save_param)
                
                file_to_note_bestModel = os.path.join(work_dir, model_name+'_note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.5f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}.\n'.format(
                    epoch+1, best_loss, print2screen_avgAccRate, best_perClassAcc))
                fn.close()

    time_elapsed = time.time() - since
    trackRecords['time_elapsed'] = time_elapsed
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
    
    return trackRecords


def train_model_stage1(dataloaders, model, lossFunc,
                optimizerW, schedulerW, pgdFunc=None,
                num_epochs=50, model_name='model', work_dir='./',
                device='cpu', print_each=1, dataset_name=None
                ):
    trackRecords = {}
    trackRecords['weightNorm'] = []
    trackRecords['acc_test'] = []
    trackRecords['acc_train'] = []
    trackRecords['weights'] = []
    log_filename = os.path.join(work_dir, model_name + '_' + dataset_name + '_train.log')
    since = time.time()
    best_loss = float('inf')
    best_acc = 0.
    best_perClassAcc = 0.0

    phaseList = list(dataloaders.keys())
    phaseList.remove('train')
    phaseList = ['train'] + phaseList


    for epoch in range(num_epochs):
        if epoch % print_each == 0:
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
        fn = open(log_filename, 'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch + 1, num_epochs))
        fn.write('--' * 5 + '\n')
        fn.close()

        # Each epoch has a training and validation phase
        for phase in phaseList:
            if epoch % print_each == 0:
                print(phase)

            predList = np.array([])
            grndList = np.array([])

            fn = open(log_filename, 'a')
            fn.write(phase + '\n')
            fn.close()

            if phase == 'train':
                schedulerW.step()
                model.train()
            else:
                model.eval()  # Set model to training mode

            running_loss_CE = 0.0
            running_loss = 0.0
            running_acc = 0.0

            # Iterate over data.
            iterCount, sampleCount = 0, 0
            time.sleep(1)
            for sample in dataloaders[phase]:
                imageList, labelList = sample
                imageList = imageList.to(device)
                labelList = labelList.type(torch.long).view(-1).to(device)

                # zero the parameter gradients
                optimizerW.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits, features = model(imageList)
                    ce_error = lossFunc(logits, labelList)

                    error = ce_error
                    softmaxScores = logits.softmax(dim=1)

                    predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)
                    accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                    accRate = (accRate == 0).type(torch.float).mean()

                    predList = np.concatenate((predList, predLabel.cpu().numpy()))
                    grndList = np.concatenate((grndList, labelList.cpu().numpy()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        error.backward()
                        optimizerW.step()
                # statistics
                iterCount += 1
                sampleCount += labelList.size(0)
                running_acc += accRate * labelList.size(0)
                running_loss_CE += error.item() * labelList.size(0)
                running_loss = running_loss_CE

                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgLoss_CE = running_loss_CE / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount

            epoch_error = print2screen_avgLoss

            confMat = sklearn.metrics.confusion_matrix(grndList, predList)
            # normalize the confusion matrix
            a = confMat.sum(axis=1).reshape((-1, 1))
            confMat = confMat / a
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i, i]
            curPerClassAcc /= confMat.shape[0]
            if epoch % print_each == 0:
                print('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}'.format(
                    epoch_error, print2screen_avgAccRate, curPerClassAcc))

            fn = open(log_filename, 'a')
            fn.write('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}\n'.format(
                epoch_error, print2screen_avgAccRate, curPerClassAcc))
            fn.close()

            if phase == 'train':
                if pgdFunc:  # Projected Gradient Descent
                    pgdFunc.PGD(model)

                trackRecords['acc_train'].append(curPerClassAcc)
            else:
                trackRecords['acc_test'].append(curPerClassAcc)
                W = model.module.fc.weight.cpu().clone()
                # W = model.module.fc.weight.cpu().clone()
                tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
                trackRecords['weightNorm'].append(tmp)
                trackRecords['weights'].append(W.detach().cpu().numpy())

            if (
                    phase == 'val' or phase == 'test') and curPerClassAcc > best_perClassAcc:  # epoch_loss<best_loss:
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name + '_' + dataset_name + '_best.paramOnly')
                torch.save(model.state_dict(), path_to_save_param)

                file_to_note_bestModel = os.path.join(work_dir, model_name + '_' + dataset_name + '_note_bestModel.log')
                fn = open(file_to_note_bestModel, 'a')
                fn.write(
                    'The best model is achieved at epoch-{}: loss{:.5f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}.\n'.format(
                        epoch + 1, best_loss, print2screen_avgAccRate, best_perClassAcc))
                fn.close()

    time_elapsed = time.time() - since
    trackRecords['time_elapsed'] = time_elapsed
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    fn = open(log_filename, 'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()

    return trackRecords