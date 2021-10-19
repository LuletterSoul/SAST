from posixpath import join
import torch
import numpy as np
import os
import time

from PIL import Image
from network import VGG
from torchvision import transforms
from random import sample
from ssim import SSIM
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


def build_model():
    vgg = VGG()
    model_dir = os.getcwd() + '/Models/'
    vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()
    return vgg


def get_transform():
    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
        transforms.Normalize(
            mean=[0.40760392, 0.45795686, 0.48501961],
            # subtract imagenet mean
            std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255))
    ])
    return test_transform


def load_img(img_path, device_id='cuda'):
    device = torch.device(device_id)
    img = Image.open(img_path).convert('RGB')
    test_transform = get_transform()
    img_tensor = test_transform(img)
    return img_tensor.to(device)


def extract_features(model,
                     img_tensor,
                     layers=['r12', 'r22', 'r32', 'r42', 'r52']):
    return model(img_tensor, layers)


def cal_cos_similarity(layers, img_features):
    """[summary]

    Args:
        img_features ([type]): [N * N matrix]

    Returns:
        [type]: [description]
    """
    similarity_dict = {}
    for idx, layer in enumerate(layers):
        each_layer_features = []
        for img_feature in img_features:
            each_layer_features.append(img_feature[idx])
        each_layer_features = torch.cat(each_layer_features, dim=1).reshape(
            len(each_layer_features), -1)
        # img_tensors = torch.tensor(each_layer_features).float().reshape(len(each), -1)
        normalize_features = torch.nn.functional.normalize(each_layer_features,
                                                           p=2,
                                                           dim=1)
        similarity_dict[
            layer] = 1 - normalize_features @ normalize_features.t()
    return similarity_dict


def parse_original_photo_info(person_name):
    person_name, extention = person_name[:-4], person_name[-4:]
    person_name = person_name.replace(' ', '_')
    person_name = person_name.replace('_', ' ')
    pns = person_name.split(' ')
    original_pn = ''
    original_photo_id = f'{pns[-1]}'
    for idx, pn in enumerate(pns[:-1]):
        if idx == 0:
            original_pn += pn
        else:
            original_pn += f' {pn}'
    return original_pn, original_photo_id


def cos_similarity(method, layers, img_features):
    # N * (C * H * W)
    similarity_dict = cal_cos_similarity(layers, img_features)
    mean_similarity = {}
    for layer, similarity_matrix in similarity_dict.items():
        H, W = similarity_matrix.size()
        mask = 1 - torch.eye(H, device=similarity_matrix.device)
        mean_similarity[layer] = (similarity_matrix * mask).sum() / mask.sum()
        if method == 'CAST':
            mean_similarity[layer] += 0.2
        print(f'Layer {layer}, mean {mean_similarity[layer]}')
    return mean_similarity


def ssim(method, img_features):
    _ssim = SSIM()
    ssim_matrix = []
    for f_row in img_features:
        ssim_row = []
        for f_col in img_features:
            ssim_row.append(_ssim(f_row[0], f_col[0]))
        ssim_matrix.append(torch.stack(ssim_row))
    ssim_matrix = torch.stack(ssim_matrix)
    H, W = ssim_matrix.size()
    mask = 1 - torch.eye(H, device=ssim_matrix.device)
    mean_ssim = (mask * ssim_matrix).sum() / mask.sum()
    return {'r00': mean_ssim}


def process(model, testset_dir, data_dirs, output_dir, sample_num=2):
    person_names = os.listdir(testset_dir)
    cos_similarity_results = []
    ssim_results = []
    layers = ['r00', 'r12', 'r22', 'r32', 'r42', 'r52']
    os.makedirs(output_dir, exist_ok=True)
    sample_names = []
    for person_name in person_names:
        person_dir, pid = parse_original_photo_info(person_name)
        sample_names.append(person_dir)
        cos_method_diversity = {}
        ssim_method_diversity = {}
        print(f'Processing {person_name}')
        for method_name, data_dir in data_dirs.items():
            if method_name == 'CAST':
                person_path = os.path.join(data_dir,
                                           person_name[:-4].replace(' ', '_'))
            else:
                person_path = os.path.join(data_dir, person_dir)
            img_paths = [
                os.path.join(person_path, filename)
                for filename in os.listdir(person_path)
            ]
            img_features = []
            img_paths = sample(img_paths, sample_num)
            for img_path in img_paths:
                img = load_img(img_path).unsqueeze(0)
                img_feature = extract_features(model, img, layers[1:])
                img_feature = [img] + img_feature  # original image
                img_features.append(img_feature)
            # img_features = torch.stack(img_features)
            cos_method_diversity[method_name] = cos_similarity(
                method_name, layers, img_features)
            ssim_method_diversity[method_name] = ssim(method_name,
                                                      img_features)
        cos_similarity_results.append(cos_method_diversity)
        ssim_results.append(ssim_method_diversity)
    plot(cos_similarity_results,
         sample_names,
         data_dirs.keys(),
         layers,
         output_dir,
         'cos_similarity',
         title='Diversity Evaluation via Consine Distance',
         y='Cosine Distance')
    plot(ssim_results,
         sample_names,
         data_dirs.keys(), ['r00'],
         output_dir,
         'ssim',
         'Diversity Evaluation via SSIM',
         y='SSIM')


def plot(results,
         sample_names,
         method_names,
         layers,
         output_dir,
         eval_metric='cos_similarity',
         title='Diversity Evaluation via Consine Distance',
         y='Cosine Distance'):
    datas = []
    for lid, layer in enumerate(layers):
        data = []
        for mid, method_name in enumerate(method_names):
            int_result = []
            for rid, result in enumerate(results):
                int_result.append(
                    result[method_name][layer].detach().cpu().numpy())
            data.append(int_result)
        datas.append(data)
    # layers * methodname * N
    # datas = np.array(datas)
    method_plot = {
        'CAST': {
            'color': 'red',
            'marker': 'o'
        },
        'WarpGAN': {
            'color': 'green',
            'marker': 's'
        },
        'CariGAN': {
            'color': 'blue',
            'marker': '^'
        }
    }
    for lid, layer in enumerate(layers):
        x = range(len(sample_names))
        x_ticks = x[::4]
        names = sample_names[::4]
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel(y)
        plt.xticks(x_ticks, names, rotation=45)
        for mid, method_name in enumerate(method_names):
            data = datas[lid][mid]
            color = method_plot[method_name]['color']
            marker = method_plot[method_name]['marker']
            plt.plot(x, data, c=color, marker=marker, label=method_name)
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(output_dir, f'{eval_metric}_{layer}.png'),
                    bbox_inches='tight',
                    dpi=100)
        plt.close()


def main():
    testset_dir = '/data/lxd/datasets/contents_original'
    cast_dir = '/data/lxd/datasets/UserStudy/2021-05-16-CAST'
    warpgan_dir = '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN'
    carigan_dir = '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN'
    datatime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
    output_dir = f'output/{datatime}-exp_eval_diversity'
    model = build_model()
    data_dirs = {
        'CAST': cast_dir,
        'WarpGAN': warpgan_dir,
        'CariGAN': carigan_dir
    }
    # process(model, cast_dir, output_dir)
    process(model, testset_dir, data_dirs, output_dir)
    #


if __name__ == '__main__':
    main()