# modified from: https://github.com/yinboc/liif
import argparse
import os
import lpips
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import datasets
import models
import utils
import math
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def make_data_loader(config):
    train_ds_type = config['train_dataset']['type']
    radius = config['train_dataset']['radius']

    train_ds_cls = getattr(datasets, train_ds_type)
    train_ds = train_ds_cls(
        opts_dict=config['train_dataset'],
        radius=radius
        )
    train_sampler = datasets.DistSampler(
        dataset=train_ds,
        num_replicas=config['num_gpu'],
        rank=0,
        ratio=config['train_dataset']['repeat']
    )
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=config['train_dataset']['batch_size'], shuffle=False, num_workers=config['train_dataset']['num_worker_per_gpu'], pin_memory=True)
    num_iter_per_epoch = math.ceil(len(train_ds) * \
                                   config['train_dataset']['repeat'] / config['train_dataset']['batch_size'])
    return train_loader, num_iter_per_epoch

def prepare_training():
    if config.get('resume') is not None:
        if os.path.exists(config.get('resume')):
            sv_file = torch.load(config['resume'],map_location='cpu')
            print("torch load success: ", config['resume'])
            model = models.make(config['model']).cuda()
            model.load_state_dict(sv_file['model']['sd'], strict=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['args']['lr'])
            if config.get('multi_step_lr') is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        else:
            raise NotImplementedError('The path of desired checkpoint does not exist.')
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log("lr: {}".format(optimizer.param_groups[0]['lr']))
    epoch_start = 1
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, iter_per_epoch, model, optimizer, loss_fn_alex, epoch):
    model.train()
    pixel_loss = utils.Averager()
    lpips_loss = utils.Averager()

    loss_weight = config['loss_weight']
    pixel_weight = loss_weight['pixel']
    lpips_weight = loss_weight['lpips']
    l1 = nn.MSELoss(reduction='mean')
    data_norm = config['data_norm']
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, 1, 1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, 1, 1).cuda()

    iteration = 0
    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['lq']
        gt = batch['gt']

        pred = model(inp)
        if pixel_weight > 0:
            pixel_l = l1(pred, gt)
            pixel_loss.add(pixel_l.item())
        else:
            pixel_l = 0.0

        if lpips_weight > 0:
            lpips_l = loss_fn_alex(torch.clamp((pred-gt_sub) / gt_div, -1, 1), (batch['gt'].detach() - gt_sub) / gt_div)
            lpips_l = lpips_l.mean()
            lpips_loss.add(lpips_l.item())
        else:
            lpips_l = 0.0

        loss = pixel_l * pixel_weight + lpips_l * lpips_weight
        writer.add_scalars('loss', {'pixel_loss': pixel_l, 'lpips_loss': lpips_l}, (epoch - 1) * iter_per_epoch + iteration)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ret_loss = []
        ret_loss.extend((pixel_loss.item(), lpips_loss.item()))
        iteration += 1
    return ret_loss


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        class MyDataParallel(nn.DataParallel):
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)
        model = MyDataParallel(model)   # able to access custom methods
    config['num_gpu'] = n_gpus

    train_loader, iter_per_epoch = make_data_loader(config)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'lq': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')
    timer = utils.Timer()
    print("lr: {}".format(optimizer.param_groups[0]['lr']))
    for epoch in range(1, 400):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, iter_per_epoch, model, optimizer, loss_fn_alex, epoch)
        log_info.append("train: pix_l={:.4f}, lpips_l={:.4f}".format(train_loss[0], train_loss[1]))

        if lr_scheduler is not None:
            lr_scheduler.step()

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save({'model': model_spec}, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='config/train.yaml')
    parser.add_argument('--name', default='hfgat')
    parser.add_argument('--gpu', default='6')
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help='Distributed launcher requires.'
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./save', save_name)
    main(config, save_path)

