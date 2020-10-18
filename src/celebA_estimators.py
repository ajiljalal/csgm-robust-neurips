"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

import copy
import heapq
import torch
import numpy as np
import utils
import scipy.fftpack as fftpack
import sys
import os
import utils
import torch.nn.functional as F
from celebA_utils import *
from lpips import PerceptualLoss

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../pytorch_GAN_zoo'))

import hubconf



def get_measurements(x_hat_batch, A, measurement_type, hparams):
    batch_size = hparams.batch_size
    if measurement_type == 'dense':
        y_hat_batch =torch.mm(x_hat_batch, A)
    # TODO
    elif measurement_type == 'circulant':
        raise NotImplementedError
        sign_pattern = torch.Tensor(hparams.sign_pattern).to(hparams.device)
        y_hat_batch = utils.partial_circulant_torch(x_hat_batch, A, hparams.train_indices,sign_pattern)
    return y_hat_batch.view(batch_size, -1)

def mom_estimator(hparams):
    device = hparams.device

    kwargs = {'model_name': 'celebAHQ-256', 'useGPU': hparams.cuda}

    model = hubconf.PGAN(pretrained = True, **kwargs)

    model.netG.eval()
    for p in model.netG.parameters():
        p.requires_grad = False

    se = torch.nn.MSELoss(reduction='none')
    mom_batch_size = hparams.mom_batch_size
    batch_size = hparams.batch_size

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        A = torch.Tensor(A_val).to(device)
        y = torch.Tensor(y_val).to(device)

        shuffled_idx = torch.randperm(hparams.num_measurements)

        best_keeper = utils.BestKeeper((hparams.batch_size,hparams.n_input))
        best_keeper_z = utils.BestKeeper((hparams.batch_size,512))

        def sample(z):
            return model.test(z, getAvG=True, toCPU=(not hparams.cuda))

        def get_loss(xf, xg):
            # compute measurements
            yf_batch = torch.mm(xf.view(batch_size, -1), A)
            yg_batch = torch.mm(xg.view(batch_size, -1), A)

            # compute corresponding losses
            loss_1 = se(yf_batch, y)
            loss_2 = se(yg_batch, y)

            loss_3 = loss_1 - loss_2
            # now find median block of loss_1 - loss_2
            loss_3 = loss_1 - loss_2

            #shuffle the losses
            loss_3 = loss_3[:,shuffled_idx]
            loss_3 = loss_3[:,:mom_batch_size*(A.shape[0]//mom_batch_size)] # make the number of rows a multiple of batch size
            loss_3 = loss_3.view(batch_size,-1,mom_batch_size) # reshape
            loss_3 = loss_3.mean(axis=-1) # find mean on each batch
            loss_3_numpy = loss_3.detach().cpu().numpy() # convert to numpy

            median_idx = np.argsort(loss_3_numpy, axis=1)[:,loss_3_numpy.shape[1]//2] # sort and pick middle element

            # pick median block
            loss_batch = loss_3[range(batch_size), median_idx] # torch.mean(loss_1_mom - loss_2_mom)
            return loss_batch



        for i in range(hparams.num_random_restarts):

            zf_batch = torch.randn(hparams.batch_size, 512).to(device)
            zg_batch = torch.randn(hparams.batch_size, 512).to(device)
            z_output_batch = torch.zeros(hparams.batch_size,512).to(device)

            zf_batch.requires_grad_()
            zg_batch.requires_grad_()
            opt1 = utils.get_optimizer(zf_batch, hparams.learning_rate, hparams)
            opt2 = utils.get_optimizer(zg_batch, hparams.learning_rate, hparams)

            for j in range(hparams.max_update_iter):
                xf_batch, xg_batch = sample(zf_batch), sample(zg_batch)
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = xf_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                opt1.zero_grad()
                xf_batch, xg_batch = sample(zf_batch), sample(zg_batch)
                loss_f_batch = get_loss(xf_batch, xg_batch)
                loss_f =  loss_f_batch.mean()
                loss_f.backward()
                opt1.step()

                opt2.zero_grad()
                xf_batch, xg_batch = sample(zf_batch), sample(zg_batch)
                loss_g_batch = -1 * get_loss(xf_batch, xg_batch)
                loss_g = loss_g_batch.mean()
                loss_g.backward()
                opt2.step()


                logging_format = 'rr {} iter {} loss_f {} loss_g {}'
                print(logging_format.format(i, j, loss_f.item(), loss_g.item()))


                if j >= hparams.max_update_iter - 200:
                    z_output_batch += zf_batch.detach()

            z_output_batch = z_output_batch/200
            x_hat_batch = sample(z_output_batch)
            y_hat_batch = torch.mm(x_hat_batch.view(hparams.batch_size,-1), A)
            m_loss_batch = get_loss(x_hat_batch, xg_batch)

            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), m_loss_batch.detach().cpu().numpy())
            best_keeper_z.report(z_output_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), m_loss_batch.detach().cpu().numpy())
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator

