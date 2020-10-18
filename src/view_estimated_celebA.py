"""View estimated images for celebA"""
# pylint: disable = C0301, R0903, R0902

import numpy as np
import celebA_input
from celebA_utils import view_image
import utils
import matplotlib.pyplot as plt
import pickle as pkl
from metrics_utils import int_or_float, find_best
import glob

class Hparams(object):
    """Hyperparameters"""
    def __init__(self):
        self.input_type = 'full-input'
        self.input_path_pattern = './test_images/celebA/*.jpg'
        self.input_path = './test_images/celebA'
        self.num_input_images = 30
        self.image_matrix = 0
        self.image_shape = (3,64,64)
        self.image_size = 64
        self.n_input = np.prod(self.image_shape)
        self.measurement_type = 'gaussian'
        self.model_types = ['MAP', 'Langevin']


def view(xs_dict, patterns_images, patterns_lpips, patterns_l2, images_nums, hparams, **kws):
    """View the images"""
    x_hats_dict = {}
    lpips_dict = {}
    l2_dict = {}
    for model_type, pattern_image, pattern_lpips, pattern_l2 in zip(hparams.model_types, patterns_images, patterns_lpips, patterns_l2):
        outfiles = [pattern_image.format(i) for i in images_nums]
        x_hats_dict[model_type] = {i: plt.imread(outfile) for i, outfile in enumerate(outfiles)}
        with open(pattern_lpips, 'rb') as f:
            lpips_dict[model_type] = pkl.load(f)
        with open(pattern_l2, 'rb') as f:
            l2_dict[model_type] = pkl.load(f)
    xs_dict_temp = {i : xs_dict[i] for i in images_nums}
    utils.image_matrix(xs_dict_temp, x_hats_dict, lpips_dict, l2_dict, view_image, hparams, **kws)
# def view(xs_dict, patterns, images_nums, hparams, **kws):
#     """View the images"""
#     x_hats_dict = {}
#     for model_type, pattern in zip(hparams.model_types, patterns):
#         outfiles = [pattern.format(i) for i in images_nums]
#         x_hats_dict[model_type] = {i: plt.imread(outfile) for i, outfile in enumerate(outfiles)}
#     xs_dict_temp = {i : xs_dict[i] for i in images_nums}
#     utils.image_matrix(xs_dict_temp, x_hats_dict, view_image, hparams, **kws)


def get_image_nums(start, stop, hparams):
    """Get range of images"""
    assert start >= 0
    assert stop <= hparams.num_input_images
    images_nums = list(range(start, stop))
    return images_nums


def main():
    """Make and save image matrices"""
    hparams = Hparams()
    xs_dict = celebA_input.model_input(hparams)
    start, stop = 0, 5
    images_nums = get_image_nums(start, stop, hparams)
    is_save = True

    def formatted(f):
        return format(f, '.4f').rstrip('0').rstrip('.')
    legend_base_regexs = [
        ('MAP',
                './estimated/celebA/full-input/circulant/4.0/',
                     '/realnvp/annealed_map/*'),
            ('Langevin',
                    './estimated/celebA/full-input/circulant/4.0/',
                         '/realnvp/annealed_langevin/*')

    ]
    criterion = ['l2', 'mean']
    retrieve_list = [['l2', 'mean'], ['l2', 'std']]

    for num_measurements in [100,200,500,1000,2500,5000,7500,10000]:
        patterns_images, patterns_lpips, patterns_l2 = [], [] , []
        exists = True
        for legend, base, regex in legend_base_regexs:
            keys = map(int_or_float, [a.split('/')[-1] for a in glob.glob(base + '*')])
            list_keys = [key for key in keys]
            if num_measurements not in list_keys:
                exists = False
                break
            pattern = base + str(num_measurements) + regex
            _, best_dir = find_best(pattern, criterion, retrieve_list)
            pattern_images = best_dir + '/{0}.png'
            pattern_lpips = best_dir + '/lpips_scores.pkl'
            pattern_l2 = best_dir + '/l2_losses.pkl'
            patterns_images.append(pattern_images)
            patterns_lpips.append(pattern_lpips)
            patterns_l2.append(pattern_l2)
    # for num_measurements in [100, 250, 500, 1000, 2500,5000,7500, 10000]:
    #     pattern1_base = './estimated/celebA/full-input/circulant/4.0/' + str(num_measurements) + '/realnvp/annealed_map/None_200.0_10.0_20.0_4.0_False_sgd_0.001_0.0_2000_1/'
    #     pattern1_images = pattern1_base + '{0}.png'
    #     pattern1_lpips = pattern1_base + 'lpips_scores.pkl'
    #     pattern1_l2 = pattern1_base + 'l2_losses.pkl'
    #     # pattern2 = './estimated/celebA/full-input/circulant/16.0/' + str(num_measurements) + '/glow_map/1.0_0.0_0.01024_adam_0.001_0.0_2000_2/{0}.png'
    #     # pattern3 = './estimated/celebA/full-input/circulant/16.0/' + str(num_measurements) + '/glow_langevin/1.0_0.0_1.0204_sgd_1e-05_0.0_3001_1/{0}.png'
    #     # pattern2 = './estimated/celebA/full-input/gaussian/5.477/' + str(num_measurements) + '/map/1.0_0.012_0.0_adam_0.01_0.0_2000_2/{0}.png'
    #     pattern2_base = './estimated/celebA/full-input/circulant/4.0/' + str(num_measurements) + '/realnvp/annealed_langevin/None_None_200.0_10.0_20.0_4.0_False_sgd_0.0005_0.0_2000_1/'
    #     pattern2_images = pattern2_base + '{0}.png'
    #     pattern2_lpips = pattern2_base + 'lpips_scores.pkl'
    #     pattern2_l2 = pattern2_base + 'l2_losses.pkl'
    #     # if num_measurements == 5000:
    #     #     pattern3_base = './estimated_backup_old/celebA/full-input/gaussian/4.0/5000/langevin/1.0_0.0064_0.0_sgd_0.0001_0.0_1000_2/'
    #     # else:
    #     #     pattern3_base = './estimated_backup_old/celebA/full-input/gaussian/4.0/' + str(num_measurements) + '/langevin/1.0_' + formatted(32/num_measurements) + '_0.0_sgd_0.001_0.0_2000_2/'
    #     # pattern3_images = pattern3_base + '{0}.png'
    #     # pattern3_lpips = pattern3_base + 'lpips_scores.pkl'
    #     # pattern3_l2 = pattern3_base + 'l2_losses.pkl'
    #     # pattern4 = './estimated/celebA/full-input/gaussian/4.0/' + str(num_measurements) + '/langevin/1.0_0.0064_0.0_sgd_0.001_0.0_2000_1/{0}.png'
    #     # pattern3 = './estimated/celebA/full-input/gaussian/5.477/' + str(num_measurements) + '/langevin/1.0_0.03_0.0_sgd_0.0001_0.0_1000_2/{0}.png'
    #     # pattern4 = './estimated/celebA/full-input/gaussian/5.477/' + str(num_measurements) + '/langevin/1.0_0.03_0.0_sgd_0.0001_0.0_1000_2/{0}.png'
    #     patterns_images = [pattern1_images, pattern2_images]
    #     patterns_lpips = [pattern1_lpips, pattern2_lpips ]
    #     patterns_l2 = [pattern1_l2, pattern2_l2]
        # try:
        print(patterns_images)
        if exists:
            view(xs_dict, patterns_images, patterns_lpips, patterns_l2, images_nums, hparams)
            # patterns = [pattern2, pattern3]
            # view(xs_dict, patterns, images_nums, hparams)
            save_path = f'./results/celebA_reconstr_{num_measurements}_{criterion[0]}_nvp_orig_map_langevin.pdf'
            utils.save_plot(is_save, save_path)
        else:
            continue
        # except:
        #     pass

if __name__ == '__main__':
    main()
