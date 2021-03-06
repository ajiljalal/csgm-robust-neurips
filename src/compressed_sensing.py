"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
from lpips import PerceptualLoss


def main(hparams):
    # set up perceptual loss
    device = 'cuda:0'
    percept = PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    utils.print_hparams(hparams)

    # get inputs
    xs_dict = model_input(hparams)

    estimators = utils.get_estimators(hparams)
    utils.setup_checkpointing(hparams)
    measurement_losses, l2_losses, lpips_scores, z_hats = utils.load_checkpoints(hparams)

    x_hats_dict = {model_type : {} for model_type in hparams.model_types}
    x_batch_dict = {}

    A = utils.get_A(hparams)
    noise_batch = hparams.noise_std * np.random.standard_t(2, size=(hparams.batch_size, hparams.num_measurements))



    for key, x in xs_dict.items():
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, key)
            is_saved = all([os.path.isfile(save_path) for save_path in save_paths.values()])
            if is_saved:
                continue

        x_batch_dict[key] = x
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()]
        x_batch = np.concatenate(x_batch_list)

        # Construct noise and measurements


        y_batch = utils.get_measurements(x_batch, A, noise_batch, hparams)

        # Construct estimates using each estimator
        for model_type in hparams.model_types:
            estimator = estimators[model_type]
            x_hat_batch, z_hat_batch, m_loss_batch = estimator(A, y_batch, hparams)

            for i, key in enumerate(x_batch_dict.keys()):
                x = xs_dict[key]
                y_train = y_batch[i]
                x_hat = x_hat_batch[i]

                # Save the estimate
                x_hats_dict[model_type][key] = x_hat

                # Compute and store measurement and l2 loss
                measurement_losses[model_type][key] = m_loss_batch[key]
                l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x)
                lpips_scores[model_type][key] = utils.get_lpips_score(percept, x_hat, x, hparams.image_shape)
                z_hats[model_type][key] = z_hat_batch[i]

        print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))

        # Checkpointing
        if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):
            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, lpips_scores, z_hats, save_image, hparams)
            x_hats_dict = {model_type : {} for model_type in hparams.model_types}
            print('\nProcessed and saved first ', key+1, 'images\n')

        x_batch_dict = {}

    # Final checkpoint
    if hparams.save_images:
        utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, lpips_scores, z_hats, save_image, hparams)
        print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))

    if hparams.print_stats:
        for model_type in hparams.model_types:
            print(model_type)
            measurement_loss_list = list(measurement_losses[model_type].values())
            l2_loss_list = list(l2_losses[model_type].values())
            mean_m_loss = np.mean(measurement_loss_list)
            mean_l2_loss = np.mean(l2_loss_list)
            print('mean measurement loss = {0}'.format(mean_m_loss))
            print('mean l2 loss = {0}'.format(mean_l2_loss))

    if hparams.image_matrix > 0:
        utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

    # Warn the user that some things were not processsed
    if len(x_batch_dict) > 0:
        print('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
        print('Consider rerunning lazily with a smaller batch size.')

if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--net', type=str, default='realnvp', help='Name of model. options = [realnvp, glow]')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--image-dtype', type=str, default='npy', help='type of image data. options:[npy, png,jpg]')
    PARSER.add_argument('--num-input-images', type=int, default=2, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=2, help='How many examples are processed together')


    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='dense', help='measurement type')
    PARSER.add_argument('--adversarial-epsilon', type=float, default=0.01, help='fraction of samples that are adversarially corrupted')
    PARSER.add_argument('--noise-std', type=float, default=1, help='std dev of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--downsample', type=int, default=None, help='downsampling factor')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None, help='model(s) used for estimation')
    PARSER.add_argument('--mom-batch-size', type=int, default=20, help='batch size for MOM tournament')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=1000, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')


    # Output
    PARSER.add_argument('--not-lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')

    PARSER.add_argument('--cuda', dest='cuda', action='store_true')
    PARSER.add_argument('--no-cuda', dest='cuda', action='store_false')
    PARSER.set_defaults(cuda=True)


    HPARAMS = PARSER.parse_args()
    HPARAMS.input_path = f'./test_images/{HPARAMS.dataset}'
    if HPARAMS.cuda:
        HPARAMS.device='cuda:0'
    else:
        HPARAMS.device = 'cpu:0'

    if HPARAMS.dataset in ['celebA', 'range']:
        if HPARAMS.net == 'pggan-512':
            HPARAMS.image_size = 512
        elif HPARAMS.net == 'pggan-256':
            HPARAMS.image_size = 256
        else:
            raise NotImplementedError
        HPARAMS.image_shape = (3, HPARAMS.image_size, HPARAMS.image_size)
        HPARAMS.n_input = np.prod(HPARAMS.image_shape)


        # TODO: integrate circulant measurements for m > 1000
        if HPARAMS.measurement_type == 'circulant':
            raise NotImplementedError
            HPARAMS.train_indices = np.random.randint(0, HPARAMS.n_input, HPARAMS.num_measurements )
            HPARAMS.sign_pattern = np.float32((np.random.rand(1,HPARAMS.n_input) < 0.5)*2 - 1.)


        from celebA_input import model_input
        from celebA_utils import view_image, save_image

    else:
        raise NotImplementedError


    HPARAMS.num_bad_measurements = int(HPARAMS.num_measurements * HPARAMS.adversarial_epsilon)
    HPARAMS.num_clean_measurements = HPARAMS.num_measurements - HPARAMS.num_bad_measurements

    main(HPARAMS)


