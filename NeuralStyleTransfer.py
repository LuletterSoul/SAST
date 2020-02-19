# %%

# %pylab inline
import os

image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'

from transforms import *
from torch.autograd import Variable
from torch import optim
import torchvision
from torchvision import transforms
from network import *
import argparse

# %%

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='images/contents',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='images/styles',
                    help='Directory path to a batch of style images')
parser.add_argument('--content_mask_dir', type=str, default='images/content_masks',
                    help='Directory path to a batch of content masks')
parser.add_argument('--style_mask_dir', type=str, default='images/style_masks',
                    help='Directory path to a batch of style masks')
parser.add_argument('--max_iter', type=int, default=500,
                    help='Max iterations of optimization for low resolution image')
parser.add_argument('--max_iter_hr', type=int, default=200,
                    help='Max iterations of optimization for high resolution image')

parser.add_argument('--update_step', type=int, default=1,
                    help='Update step of loss function and laplacian graph')

parser.add_argument('--update_step_hr', type=int, default=1,
                    help='Update step of loss function and laplacian graph')

parser.add_argument('--img_size', type=int, default=256,
                    help='Image size of low resolution')
parser.add_argument('--img_size_hr', type=int, default=512,
                    help='Image size of high resolution')

parser.add_argument('--kl', type=int, default=7,
                    help='K neighborhoods selection for laplacian graph')
parser.add_argument('--km', type=int, default=1,

                    help='K neighborhoods selection for mutex graph')
# parser.add_argument('--sigma', type=int, default=10,
#                     help='Weight of Variance loss ')

parser.add_argument('--batch_size', type=int, default=4)
# training options0
parser.add_argument('--save_dir',
                    default='./experiments/02-19_gatys_lw1e5_iter500_200_512_ul50_uh50_kl7_km1',
                    help='Directory to save the model')

args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

# pre and post processing for images
# img_size = 256
# img_size_hr = 512  # works for 8GB GPU, make larger if you have 12GB or more
# img_size = 512
# img_size_hr = 800  # works for 8GB GPU, make larger if you have 12GB or more

# these are layers settings:
# style_layers = []


args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

# pre and post processing for images
# img_size = 256
# img_size_hr = 512  # works for 8GB GPU, make larger if you have 12GB or more
# img_size = 512
# img_size_hr = 800  # works for 8GB GPU, make larger if you have 12GB or more

# these are layers settings:
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
# style_layers = ['r11','r21','r31','r41', 'r51']
# style_layers = []
# style_weights = []

content_layers = ['r42']
content_weights = [1e2]
# content_layers = []
# content_weights = []

# laplacia_layers = ['r32']
# feature maps size : [ 256, 64, 64]
# laplacia_layers = ['r32']
laplacia_layers = []
laplacia_weights = []
# laplacia_weights = [1e5 / n ** 2 for n in [256]]
# laplacia_weights = ['r32']

# mutex_layers = ['r52']
mutex_layers = []
# mutex_weights = [1e3 / n ** 2 for n in [512]]
mutex_weights = []
# mutex_weights = ['r32']

# std_weight = 1e6 / 5

loss_layers = style_layers + content_layers + laplacia_layers + mutex_layers
weights = style_weights + content_weights + laplacia_weights + mutex_weights
# laplacia_layers = ['r21', 'r31', 'r41']
# laplacia_layers = ['r21', 'r31', 'r41']
# laplacia_layers = ['r31', 'r41']
# laplacia_layers = []


# max_iter = 500
# max_iter_hr = 200
# update_step = 50
# max_iter = 1000
# max_iter_hr = 100


# these are good weights settings:
# style_weights = []
# style_weights = [1e2 / n ** 2 for n in [64, 128, 256, 512, 512]]
# laplacia_weights = []
# laplacia_weights = [1e3 / n ** 2 for n in [512]]
# laplacia_weights = [3 / n for n in [1, 2, 3]]

# vgg definition that conveniently let's you grab the outputs from any layer

# %%

# gram matrix and loss


# %%

prep = transforms.Compose([transforms.Resize(args.img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])

prep_hr = transforms.Compose([transforms.Resize(args.img_size_hr),
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                              transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                   # subtract imagenet mean
                                                   std=[1, 1, 1]),
                              transforms.Lambda(lambda x: x.mul_(255)),
                              ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                  std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                             ])
mask_tf = transforms.Compose([ToUnNormalizedTensor()])
postpb = transforms.Compose([transforms.ToPILImage()])


def postp(tensor):  # to clip results in the range [0,1]
    t = post_tensor(tensor)
    img = postpb(t)
    return img
    # return t


def post_tensor(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    return t


content_dataset = FlatFolderDataset(args.content_dir, args.content_mask_dir, prep, prep_hr, mask_tf)
style_dataset = FlatFolderDataset(args.style_dir, args.style_mask_dir, prep, prep_hr, mask_tf)
# content_hr_dataset = FlatFolderDataset(args.content_dir, prep)
# style_hr_dataset = FlatFolderDataset(args.style_dir, prep)

content_loader = data.DataLoader(
    content_dataset, batch_size=1, shuffle=False,
    num_workers=0)
style_loader = data.DataLoader(
    style_dataset, batch_size=1, shuffle=False,
    num_workers=0)

# content_hr_loader = data.DataLoader(
#     content_hr_dataset, batch_size=1, shuffle=False,
#     num_workers=0)
# style_hr_loader = data.DataLoader(
#     style_hr_dataset, batch_size=1, shuffle=False,
#     num_workers=0)

# # print(len(content_loader))
# print(len(style_loader))
# style_dataset content_iter = iter()
# style_iter = iter()
# %%

# get network
device = torch.device('cuda')
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

# %%

# load images, ordered as [style_image, content_image]
# img_dirs = [image_dir, image_dir]
# img_names = ['vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg']
# imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]
# imgs_torch = [prep(img) for img in imgs]
# if torch.cuda.is_available():
#     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
# else:
#     imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]

outputs = []
style_images = []
epoch = 0

for content_image, content_image_hr, content_mask, content_name in content_loader:
    # print(content_name)
    for style_image, style_image_hr, style_mask, style_name in style_loader:
        content_image = content_image.to(device)
        content_image_hr = content_image_hr.to(device)

        content_mask = content_mask.to(device)
        style_mask = style_mask.to(device)

        style_image = style_image.to(device)
        style_image_hr = style_image_hr.to(device)
        # style_image = style_image.squeeze(0)
        # content_image = content_image.squeeze(0)
        # style_image, content_image = imgs_torch
        # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
        opt_img = Variable(content_image.data.clone(), requires_grad=True)
        # %% # display images
        # for img in imgs:
        #     imshow(img).show()

        # %%

        # define layers, loss functions, weights and compute optimization targets

        # style_feats = vgg(style_image, style_layers)
        # content_feats = vgg(content_image, content_layers)
        # # content_complete_feats = vgg(content_image, style_layers)
        # laplacian_c_feats = vgg(content_image, laplacia_layers)
        # laplacian_s_feats = vgg(style_image, laplacia_layers)
        #
        # # init Laplacian graph
        # laplacian_graphs = [
        #     cal_laplacian_graph(laplacian_c_feats[idx], laplacian_s_feats[idx], laplacian_s_feats[idx], 7)
        #     for idx in
        #     range(len(laplacian_s_feats))]
        # # compute optimization targets
        #
        # loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers) + [ConsistencyLoss(l) for
        #                                                                                          l in laplacian_graphs]
        # if torch.cuda.is_available():
        #     loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        #
        # style_targets = [GramMatrix()(A).detach() for A in style_feats]
        # content_targets = [A.detach() for A in content_feats]
        # laplacia_targets = [A.detach() for A in laplacian_s_feats]
        # targets = style_targets + content_targets + laplacia_targets
        M = Maintainer(vgg, content_image, style_image, content_layers, style_layers, laplacia_layers,
                       device, args.kl,
                       args.km, content_mask, style_mask)

        # %%

        # run style transfer
        show_iter = 50
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]
        while n_iter[0] <= args.max_iter:
            def closure():
                optimizer.zero_grad()
                out = vgg(opt_img, loss_layers)
                # M.add_mutex_constrain(out[-len(mutex_layers):])
                layer_losses = [weights[a] * M.loss_fns[a](A, M.targets[a]) for a, A in enumerate(out)]
                torch.cuda.empty_cache()
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0] += 1
                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                if n_iter[0] % args.update_step == (args.update_step - 1) and not M.laplacian_updated:
                    pass
                    # Using output as content image to update laplacian graph dynamiclly during trainig.
                    # M.update_loss_fns_with_lg(out[-len(laplacia_layers) + -len(mutex_layers):-len(mutex_layers)],
                    #                           M.laplacian_s_feats)
                    # M.update_loss_fns_with_lg(out[-len(laplacia_layers):],
                    # M.laplacian_s_feats)
                    # M.laplacian_updated = True
                    # M.update_loss_fns_with_lg(out[len(content_layers) + len(style_layers):], M.laplacian_s_feats)
                    # print('Update: Laplacian graph and Loss functions: %d' % (n_iter[0] + 1))
                    # print('Update laplacian graph and loss functions: %d' % (n_iter[0] + 1))
                #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss


            optimizer.step(closure)

        # display result
        out_img = postp(opt_img.data[0].cpu().squeeze())
        # imshow(out_img)
        # gcf().set_size_inches(10,10)
        # torchvision.utils.save_image('./output.png', opt_img)

        # %%

        # make the image high-resolution as described in
        # "Controlling Perceptual Factors in Neural Style Transfer", Gatys et al.
        # (https://arxiv.org/abs/1611.07865)

        # hr preprocessing

        # prep hr images
        # imgs_torch = [prep_hr(img) for img in imgs]
        # if torch.cuda.is_available():
        #     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
        # else:
        #     imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
        # style_image, content_image = imgs_torch

        # Update Global Training Components

        M = Maintainer(vgg, content_image_hr, style_image_hr, content_layers, style_layers, laplacia_layers,
                       device, args.kl,
                       args.km, content_mask, style_mask)

        # now initialise with upsampled lowres result
        opt_img = prep_hr(out_img).unsqueeze(0)
        opt_img = Variable(opt_img.type_as(content_image_hr.data), requires_grad=True)

        # style_targets = [GramMatrix()(A).detach() for A in vgg(style_image_hr, style_layers)]
        # content_targets = [A.detach() for A in vgg(content_image_hr, content_layers)]
        # laplacia_targets = [GramMatrix()(A).detach() for A in vgg(style_image_hr, style_layers)]
        # targets = style_targets + content_targets + laplacia_targets

        # %%

        # run style transfer for high res
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]
        while n_iter[0] <= args.max_iter_hr:

            def closure():
                optimizer.zero_grad()
                out = vgg(opt_img, loss_layers)
                # M.add_mutex_constrain(out[-len(mutex_layers):])
                layer_losses = [weights[a] * M.loss_fns[a](A, M.targets[a]) for a, A in enumerate(out)]
                loss = sum(layer_losses)
                # loss = sum(layer_losses)
                torch.cuda.empty_cache()
                loss.backward()
                n_iter[0] += 1
                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    # print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data[0]))
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                if n_iter[0] % args.update_step_hr == (args.update_step_hr - 1) and not M.laplacian_updated:
                    # Using output as content image to update laplacian graph dynamiclly during trainig.
                    # M.update_loss_fns_with_lg(out[-len(laplacia_layers):],
                    #                           M.laplacian_s_feats)
                    # M.update_loss_fns_with_lg(out[-len(laplacia_layers) + -len(mutex_layers):-len(mutex_layers)],
                    #                           M.laplacian_s_feats)
                    pass
                    # M.laplacian_updated = True
                    # M.update_loss_fns_with_lg(out[len(content_layers) + len(style_layers):], M.laplacian_s_feats)
                    # print('Update: Laplacian graph and Loss functions: %d' % (n_iter[0] + 1))
                # k          print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss


            optimizer.step(closure)

        # display result
        out_img_hr = post_tensor(opt_img.data.cpu().squeeze()).unsqueeze(0)
        style_image_hr = post_tensor(style_image_hr.data.cpu().squeeze()).unsqueeze(0)
        # imshow(out_img_hr)
        style_images.append(style_image_hr)
        outputs.append(out_img_hr)

        if (epoch + 1) % args.batch_size == 0:
            style_images = torch.cat(style_images, dim=0)
            outputs = torch.cat(outputs, dim=0)
            o = torch.cat([style_images, outputs], dim=0)
            path = os.path.join(args.save_dir,
                                'total-{}-{}-{}.png'.format(epoch, content_name[0], style_name[0]))
            torchvision.utils.save_image(o, path, nrow=args.batch_size)
            print('Save to [{}]'.format(path))
            outputs = []
            style_images = []

        output_path = os.path.join(args.save_dir, f'{epoch}-{content_name[0]}-{style_name[0]}.png')
        torchvision.utils.save_image(out_img_hr, output_path)
        print('Done: [{}-{}].'.format(content_name[0], style_name[0]))

        epoch += 1
# gcf().set_size_inches(10,10)

# %%
