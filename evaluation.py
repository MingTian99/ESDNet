'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import os
import lpips
import argparse
import torch
from PIL import Image

parser = argparse.ArgumentParser(description='PSNR SSIM LPIPS', add_help=False)

parser.add_argument('--generated_images_path', default='',
                    help='Generated / Restored / Recovered images -path')
parser.add_argument('--target_path', default='', help='Ground-truth -path')
parser.add_argument('--Test_Y', default=True, help='True: test Y channel only; False: test RGB channels')
parser.add_argument('--Score_save_path', default='Score', help='save the psnr, ssim and lpips score by txt')
parser.add_argument('--init_model', default='alex', help='Initializing the model,alex or vgg')
parser.add_argument('--version', type=str, default='0.1')
parser.add_argument('--crop_border', type=int, default='4', help='same with scale')
args = parser.parse_args()


def main():
    # Configurations
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=args.init_model, version=args.version)
    crop_border = args.crop_border  # same with scale
    suffix = ''  # suffix for Gen images
    test_Y = args.Test_Y  # True: test Y channel only; False: test RGB channels
    PSNR_all = []
    SSIM_all = []
    LPIPS_all = []
    img_list = sorted(glob.glob(args.target_path + '/*'))
    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')
    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_GT = cv2.imread(img_path) / 255.
        img_type = Image.open(img_path).format

        if img_type == "JPEG":
            im_Gen = cv2.imread(os.path.join(args.generated_images_path, base_name + suffix + '.jpg')) / 255.
            im_Gen_lpips = lpips.im2tensor(
                lpips.load_image(os.path.join(args.generated_images_path, base_name + suffix + '.jpg')))
        elif img_type == "PNG":
            im_Gen = cv2.imread(os.path.join(args.generated_images_path, base_name + suffix + '.png')) / 255.
            im_Gen_lpips = lpips.im2tensor(
                lpips.load_image(os.path.join(args.generated_images_path, base_name + suffix + '.png')))
        else:
            print("不支持的图像类型")

        # im_GT_lpips = lpips.im2tensor(lpips.load_image(img_path)).to(device)
        im_GT_lpips = lpips.im2tensor(lpips.load_image(img_path))
        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if crop_border == 0:
            cropped_GT = im_GT_in
            cropped_Gen = im_Gen_in
        else:
            if im_GT_in.ndim == 3:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT_in.ndim == 2:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        if test_Y:
            PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
        else:
            PSNR = calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        LPIPS = loss_fn.forward(im_Gen_lpips, im_GT_lpips).item()
        print('{:4}.png \tPSNR: {:.4f} dB, \tSSIM: {:.4f},\tLPIPS: {:.4f}'.format(
            base_name, PSNR, SSIM, LPIPS))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
        LPIPS_all.append(LPIPS)
    Mean_format = 'Mean_PSNR: {:.4f}, Mean_SSIM: {:.4f},Mean_LPIPS: {:.4f}'
    Mean_PSNR = sum(PSNR_all) / len(PSNR_all)
    Mean_SSIM = sum(SSIM_all) / len(SSIM_all)
    Mean_LPIPS = sum(LPIPS_all) / len(LPIPS_all)
    print(Mean_format.format(Mean_PSNR, Mean_SSIM, Mean_LPIPS))

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_rgb_psnr(img1, img2):
    """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
    """
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = calculate_psnr(img1[:, :, i], img2[:, :, i])
        sum_psnr += this_psnr
    return sum_psnr / n_channels


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
