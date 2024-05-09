import torch
import os
# from utils.common import print_network
from skimage import img_as_ubyte
import utils
import argparse
import math
from tqdm import tqdm
from model import model
import torch.utils.data as data
from glob import glob
from PIL import Image
from spikingjelly.activation_based import functional
import torchvision.transforms.functional as TF
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', type=str, default='crop')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. -1 for CPU')
parser.add_argument('--data_path', type=str, default='D:/1Single_Image_Derain/data/Rain12/input')
parser.add_argument('--save_path', type=str, default='img_ESDNet_12')
parser.add_argument('--eval_workers', type=int, default=12)
parser.add_argument('--crop_size', type=int, default=80)
parser.add_argument('--overlap_size', type=int, default=8)
parser.add_argument('--weights', type=str, default='checkpoints/ESDNet/models/Rain200H/model_best.pth')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

crop_size = opt.crop_size
overlap_size = opt.overlap_size
batch_size = opt.batch_size


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class DataLoaderEval(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderEval, self).__init__()
        self.opt = opt
        imgs = glob(os.path.join(opt.data_path, '*.png')) + glob(os.path.join(opt.data_path, '*.jpg'))

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = imgs
        self.sizex = len(self.imgs)
        self.count = 0

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        inp_img = TF.to_tensor(inp_img)
        return inp_img, os.path.basename(inp_path)


def getevalloader(opt):
    dataset = DataLoaderEval(opt)
    print("Dataset Size:%d" % (len(dataset)))
    evalloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.eval_workers,
                                 pin_memory=True)
    return evalloader


def splitimage(imgtensor, crop_size=crop_size, overlap_size=overlap_size):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=batch_size, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=crop_size, resolution=(batch_size, 3, crop_size, crop_size)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


from collections import OrderedDict


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


if __name__ == '__main__':
    model_restoration = model.cuda()
    functional.set_step_mode(model_restoration, step_mode='m')
    functional.set_backend(model_restoration, backend='cupy')
    model_restoration.load_state_dict(torch.load(opt.weights))
    print("===>Testing using weights: ", opt.weights)
    model_restoration.cuda()
    model_restoration.eval()
    inp_dir = opt.data_path
    eval_loader = getevalloader(opt)
    result_dir = opt.save_path
    os.makedirs(result_dir, exist_ok=True)
    with torch.no_grad():
        for input_, file_ in tqdm(eval_loader, unit='img'):
            input_ = input_.cuda()
            B, C, H, W = input_.shape
            split_data, starts = splitimage(input_)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cuda()
                functional.reset_net(model_restoration)
                split_data[i] = split_data[i].cpu()

            restored = mergeimage(split_data, starts, resolution=(B, C, H, W))
            restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).numpy()
            for j in range(B):
                fname = file_[j]
                cleanname = fname
                save_file = os.path.join(result_dir, cleanname)
                save_img(save_file, img_as_ubyte(restored[j]))
