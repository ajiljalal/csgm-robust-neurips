"""Inputs for celebA dataset"""

import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class NumpyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, input_path, transform=None):
        """
        Args:
            input_path (string): Path to directory containing directory of images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_path = input_path
        self.dir_name = glob.glob(os.path.join(input_path, '*'))[0]
        self.dir = self.transform = transform

    def __len__(self):
        image_paths = glob.glob(os.path.join(self.dir_name, '*'))

        return len(image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.join(self.dir_name, f'{idx}.npy')
        image = np.load(image_name)
        if self.transform:
            image = self.transform(image)
        sample = [image, 0] #{'image': image, 'label': 0}


        return sample


def get_full_input(hparams):
    """Create input tensors"""
    if hparams.image_dtype == 'npy':
        trans = torch.from_numpy
        dataset = NumpyDataset(hparams.input_path, transform = trans)
    elif hparams.image_dtype in ['png', 'jpeg']:
        trans = transforms.Compose([transforms.Resize((hparams.image_size,hparams.image_size)),transforms.ToTensor()])
        dataset = datasets.ImageFolder(hparams.input_path, transform=trans)

    if hparams.input_type == 'full-input':
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,drop_last=False,shuffle=False)
    elif hparams.input_type == 'random-test':
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,drop_last=False,shuffle=True)
    else:
        raise NotImplementedError

    dataiter = iter(dataloader)
    images = {i: next(dataiter)[0].view(-1).numpy() for i in range(hparams.num_input_images)}

    return images


def sample_generator_images(hparams):
    """Sample random images from the generator"""

    if hparams.model_name == 'pggan':
        return sample_pggan_images(hparams)
    else:
        raise NotImplementedError

#TODO
def sample_pggan_images(hparams):
    raise NotImplementedError
    # model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    # model = model.eval()
    # model = model.cuda()
    # z = model.sample_z(n=hparams.batch_size)
    # x = model.postprocess(model.inverse(z))
    # x = x.detach().cpu().numpy()

    # images = {i: image.reshape(1,-1) for (i, image) in enumerate(x)}
    # return images


def model_input(hparams):
    """Create input tensors"""

    if hparams.input_type in ['full-input', 'random-test']:
        images = get_full_input(hparams)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images
