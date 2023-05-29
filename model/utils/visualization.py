import os

import matplotlib.pyplot as plt
import torch

from config import PROJECT_ROOT
from utils.image_processing import save_image


def visualize_constructed_image(model, data_loader, criterion, out):
    """
    visualize original and constructed images
    :param model: an instance of Pytorch model, if not presented, a model is loaded from 'model_path'
    :param criterion: the loss function used from training step
    :param data_loader: to sample image
    :param out: name of output folder
    :return:
    """

    out_fold = os.path.join(PROJECT_ROOT, 'tmp', out)

    if os.path.exists(out_fold):  # remove if existed
        import shutil
        shutil.rmtree(out_fold)
    os.makedirs(out_fold)

    losses = []
    with torch.no_grad():
        for idx, (img, name, ext) in enumerate(data_loader):
            constructed_img = model(img)  # get constructed image by trained model
            loss = criterion(constructed_img, img).item()  # get loss (difference) between original & constructed
            losses.append(loss)  # keep track of all loss to plot histogram later

            save_image('{}/{}{}'.format(out_fold, name[0], ext[0]),  # path to save original image
                       (img[0].numpy()*255).astype(int).transpose(1, 2, 0))  # rescale image from [0, 1] -> [0, 255]
            save_image('{}/{}_{}{}'.format(out_fold, name[0], round(loss, 3), ext[0]),  # path to save constructed img
                       (constructed_img[0].numpy()*255).astype(int).transpose(1, 2, 0))  # rescale img to [0, 255]

    plt.hist(losses)
    plt.savefig(os.path.join(out_fold, 'loss_histogram.png'), format='png')
    # plt.show()
