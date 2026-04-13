import os
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

def save_model(model, model_dir, model_name):
    '''
    model: this is not the multigpu model
    '''
    #print(os.path.join(model_dir, (model_name + '____{0:.4f}.pth').format(accu)))
    with open(os.path.join(model_dir,'saved_model.txt'), 'w') as f:
        f.write('epoch:'+model_name)
    # torch.save(obj=model, f=os.path.join(model_dir, ('model.pth').format(accu)))
    torch.save(obj=model, f=os.path.join(model_dir, model_name))
    log.info('Saved model to {}'.format(os.path.join(model_dir, model_name)))

def save_images(images, path, rgb=True):
    makedir(path + 'images/')
    for idx, img in enumerate(images):
        if rgb:
            plt.imsave(path + 'images/' + str(idx) + '.png', np.squeeze(img))
        else:
            plt.imsave(path + 'images/' + str(idx) + '.png', np.squeeze(img), cmap='gray')
    log.info('Saved component images to {}'.format(path +'images/'))

def save_projections(model, dataloader, path):
    projections = []
    #zmean_projections = []
    features_projections = []
    y_projections = []
    labels = []
    with torch.no_grad():
        model.eval()
        for data, label in dataloader:
            data = data.to(model.device)
            _, out_infer = model(data)
            #if model.local_patches:
            #    data = model.decompose_image(data)
            h1 = model.qy_x.h1(data)
            projections.extend(out_infer['z'].detach().cpu().numpy())
            #zmean_projections.extend(out_infer['z_mean'].detach().cpu().numpy())
            y_projections.extend(out_infer['y'].detach().cpu().numpy())
            features_projections.extend(h1.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
            
    
    np.save(path + 'projections.npy', projections)
    #np.save(path + 'zmean_projections.npy', zmean_projections)
    np.save(path + 'y_projections.npy', y_projections)
    #np.save(path + 'feat_projections.npy', features_projections)
    np.save(path + 'labels.npy', labels)
    log.info('Saved projections to {}'.format(path + 'projections.npy'))
    

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)
