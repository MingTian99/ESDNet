import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from dataset_load import Dataload
import time
import utils
from model import model
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from losses import *
from torch.utils.tensorboard import SummaryWriter
import argparse
from spikingjelly.activation_based import functional

if __name__ == "__main__":
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    start_epoch = 1

    parser = argparse.ArgumentParser(description='Image Deraining')
    parser.add_argument('--train_dir', default='', type=str,
                        help='Directory of train images')
    parser.add_argument('--val_dir', default='', type=str,
                        help='Directory of validation images')
    parser.add_argument('--model_save_dir', default='./checkpoints/', type=str, help='Path to save weights')
    parser.add_argument('--pretrain_weights', default='./checkpoints/model_best.pth', type=str,
                        help='Path to pretrain-weights')
    parser.add_argument('--mode', default='ESDNet3', type=str)
    parser.add_argument('--session', default='DID-Data_new', type=str, help='session')
    parser.add_argument('--patch_size_train', default=64, type=int, help='training patch size')
    parser.add_argument('--patch_size_test', default=64, type=int, help='val patch size')
    parser.add_argument('--num_epochs', default=1000, type=int, help='num_epochs')
    parser.add_argument('--batch_size', default=12, type=int, help='batch_size')
    parser.add_argument('--val_epochs', default=1, type=int, help='val_epochs')
    parser.add_argument('--lr', default=1e-3, type=int, help='LearningRate')
    parser.add_argument('--min_lr', default=1e-7, type=int, help='min_LearningRate')
    parser.add_argument('--warmup_epochs', default=3, type=int, help='warmup_epochs')
    parser.add_argument('--clip_grad', default=1.0, type=float, help='clip_grad')
    parser.add_argument('--use_amp', default=False, type=bool, help='use_amp')
    parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
    args = parser.parse_args()

    start_lr = args.lr
    end_lr = args.min_lr
    clip_grad = args.clip_grad
    use_amp = args.use_amp
    mode = args.mode
    session = args.session
    patch_size_train = args.patch_size_train
    patch_size_test = args.patch_size_test
    model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
    utils.mkdir(model_dir)
    train_dir = args.train_dir
    val_dir = args.val_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    val_epochs = args.val_epochs
    num_workers = args.num_workers

    ######### Model ###########
    model_restoration = model
    model_restoration.cuda()

    functional.set_step_mode(model_restoration, step_mode='m')
    functional.set_backend(model_restoration, backend='cupy')

    # print number of model
    # get_parameter_number(model_restoration)
    # device_ids = 0
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    optimizer = optim.AdamW(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = args.warmup_epochs

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)

    # scheduler_cosine = optim.lr_scheduler.StepLR(step_size=50, gamma=0.8,
    #                                       optimizer=optimizer)  ####step_size epoch, best_epoch 445

    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)

    # scheduler.step()
    RESUME = False
    Pretrain = False
    model_pre_dir = 'checkpoints/ESDNet/models/DID-Data'
    ######### Pretrain ###########
    if Pretrain:
        utils.load_checkpoint(model_restoration, model_pre_dir)

        print('------------------------------------------------------------------------------')
        print("==> Retrain Training with: " + model_pre_dir)
        print('------------------------------------------------------------------------------')

    ######### Resume ###########
    if RESUME:
        path_chk_rest = utils.get_last_path(model_pre_dir, '_last.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        # model_restoration.load_state_dict(torch.load(model_pre_dir))
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    # criterion = nn.MSELoss().cuda()
    criterion_ssim = utils.SSIM().cuda()
    # criterion_L1 = nn.SmoothL1Loss().cuda()
    criterion_psnr = PSNRLoss().cuda()
    ######### DataLoaders ###########

    dataset_train = Dataload(data_dir=train_dir, patch_size=patch_size_train)
    train_loader = DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False,
                              pin_memory=True)

    dataset_val = Dataload(data_dir=val_dir, patch_size=patch_size_test)
    val_loader = DataLoader(dataset=dataset_val, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=False,
                            pin_memory=True)
    # train_dataset = get_training_data(train_dir, {'patch_size': patch_size_train})
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                           drop_last=False,
    #                           pin_memory=True)
    #
    # val_dataset = get_validation_data(val_dir, {'patch_size': patch_size_test})
    # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False,
    #                         pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    writer = SummaryWriter(model_dir)
    iter = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        train_psnr_val_rgb = []
        scaled_loss = 0
        model_restoration.train()
        # scheduler.step()
        for i, data in enumerate(tqdm(train_loader, unit='img'), 0):
            for param in model_restoration.parameters():
                param.grad = None
            target_ = data[1].cuda()
            input_ = data[0].cuda()
            restored = model_restoration(input_)
            if use_amp:
                with torch.cuda.amp.autocast():
                    ssim = criterion_ssim(restored, target_)
                    loss = 1 - ssim
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                functional.reset_net(model_restoration)
            else:
                # L1_Loss = criterion_L1(restored, target_)
                ssim = criterion_ssim(restored, target_)
                psnr = criterion_psnr(restored, target_)
                loss = 1-ssim
                loss.backward()
                scaled_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                optimizer.step()
                functional.reset_net(model_restoration)
            torch.cuda.synchronize()
            epoch_loss += loss.item()
            iter += 1
            for res, tar in zip(restored, target_):
                train_psnr_val_rgb.append(utils.torchPSNR(res, tar))
            psnr_train = torch.stack(train_psnr_val_rgb).mean().item()

            writer.add_scalar('loss/iter_loss', loss.item(), iter)
            writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('lr/epoch_loss', scheduler.get_lr()[0], epoch)
        #### Evaluation ####
        if epoch % val_epochs == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader, unit='img'), 0):
                target = data_val[1].cuda()
                input_ = data_val[0].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                functional.reset_net(model_restoration)

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(model_restoration.state_dict(), os.path.join(model_dir, "model_best.pth"))

            print("[epoch %d Training PSNR: %.4f --- best_epoch %d Test_PSNR %.4f]" % (epoch, psnr_train, best_epoch, best_psnr))
        if epoch % 50 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_last.pth"))
        scheduler.step()
        print("-" * 150)
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tTrain_PSNR: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}\tTest_PSNR: {:.4f}".format(
                epoch, time.time() - epoch_start_time, loss.item(), psnr_train, ssim, scheduler.get_lr()[0],
                best_psnr, ))
        print("-" * 150)
    writer.close()
