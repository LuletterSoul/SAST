import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import os
from PIL import Image


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print(
                "WARNING: You have a CUDA device, so you should probably set cuda=True"
            )
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True,
                                   transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits):(k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        all_imgs = os.listdir(root)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


if __name__ == '__main__':

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # cifar = dset.CIFAR10(root='data/',
    #                      download=True,
    #                      transform=transforms.Compose([
    #                          transforms.Scale(32),
    #                          transforms.ToTensor(),
    #                          transforms.Normalize((0.5, 0.5, 0.5),
    #                                               (0.5, 0.5, 0.5))
    #                      ]))
    cast = CustomDataSet(root='/data/lxd/datasets/UserStudy_FID/CAST',
                         transform=tf)
    # cast = dset.ImageFolder(
    # root='/data/lxd/datasets/UserStudy/2021-05-16-CAST', transform=tf)
    warpgan = CustomDataSet(root='/data/lxd/datasets/UserStudy_FID/WarpGAN',
                            transform=tf)
    carigan = CustomDataSet(root='/data/lxd/datasets/UserStudy_FID/CariGAN',
                            transform=tf)
    print("Calculating Inception Score...")
    cast_score = print(
        inception_score(cast, cuda=True, batch_size=32, resize=True,
                        splits=10))

    warpgan_score = print(
        inception_score(warpgan,
                        cuda=True,
                        batch_size=32,
                        resize=True,
                        splits=10))

    carigan_score = print(
        inception_score(carigan,
                        cuda=True,
                        batch_size=32,
                        resize=True,
                        splits=10))
    print(f'CAST Inception Score: {cast_score}')
    print(f'WarpGAN Inception Score: {warpgan_score}')
    print(f'CariGAN Inception Score: {carigan_score}')
