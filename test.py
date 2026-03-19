import os
import argparse
import torch
from torch.utils.data import DataLoader
import yaml
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import utils
import models
import datasets


def eval_psnr(loader, loss_fn_alex, model, data_norm=None, window_size=0, verbose=False, sample=0, detail=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'lq': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, 1, 1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, 1, 1).cuda()
    psnr_fn = utils.calc_psnr

    val_lpips = utils.Averager()
    val_psnr = utils.Averager()
    if detail:
        ssim_fn = utils.calculate_ssim
        val_ssim = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for idx, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['lq']
        _, _, _, h_old, w_old = inp.size()

        if window_size != 0:
            w_pad = (window_size - h_old % window_size) % window_size
            h_pad = (window_size - w_old % window_size) % window_size
            if h_pad > 0 or w_pad > 0:
                inp = torch.nn.functional.pad(inp, (0, h_pad, 0, w_pad))
        else:
            h_pad = 0
            w_pad = 0

        _, _, _, h_new, w_new = inp.size()
        torch.cuda.empty_cache()
        if w_old>2000:
            with torch.no_grad():
                pred_1 = model(inp[:, :, :, :h_new//2, :w_new//2].contiguous())
                pred_2 = model(inp[:, :, :, :h_new//2, w_new//2:].contiguous())
                pred_3 = model(inp[:, :, :, h_new//2:, :w_new//2].contiguous())
                pred_4 = model(inp[:, :, :, h_new//2:, w_new//2:].contiguous())
                pred_w = torch.cat([pred_1,pred_2], dim=3)
                pred_h = torch.cat([pred_3,pred_4], dim=3)
                pred = torch.cat([pred_w,pred_h], dim=2)
        else:
            with torch.no_grad():
                pred = model(inp[:,:,:,:])

        if h_pad > 0 or w_pad > 0:
            pred = pred[:, :, :h_old, :w_old].contiguous()

        if detail:
            res = ssim_fn(torch.clamp(pred, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255., batch['gt'].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.)
            val_ssim.add(res.item(), inp.shape[0])

            en_psnr = psnr_fn(pred, batch['gt'])
            val_psnr.add(en_psnr.item(), inp.shape[0])

            d = loss_fn_alex(torch.clamp((pred - gt_sub) / gt_div, -1, 1), (batch['gt'] - gt_sub) / gt_div)
            val_lpips.add(d.mean().detach(), inp.shape[0])

        if idx <sample:
            img = (pred[0].permute(1, 2, 0) * 255.).cpu().numpy()
            m = "%04d" % (idx+1)
            img = Image.fromarray(img.round().astype(np.uint8), mode='RGB')
            img.save(os.path.join(save_path, '{}.png'.format(m)))

        if verbose:
            pbar.set_description('LPIPS {:.4f}'.format(val_lpips.item()))

    if detail:
        result_dict = {'Enhanced PSNR': val_psnr.item(), 'Enhanced SSIM': val_ssim.item(),  'Enhanced LPIPS': val_lpips.item()}
        return result_dict
    else:
        return val_lpips.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/test.yaml')
    parser.add_argument('--model', default='QP42_HM.pth')
    parser.add_argument('--gpu',  default='7')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument("--detail", action='store_false')
    parser.add_argument('--name', type=str, default='hfgat')
    args = parser.parse_args()

    if args.name is None:
        save_path = './sample'
    else:
        save_path = os.path.join('./sample', args.name)

    if args.sample > 0 and not os.path.isdir(save_path):
        print("create sample directory {}".format(save_path))
        if not os.path.isdir('./sample'):
            os.mkdir('./sample')
        os.mkdir(save_path)
    torch.cuda.set_device(int(args.gpu))
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_spec = torch.load(args.model,map_location='cpu')
    model = models.make(model_spec['model'], load_sd=True).cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    spec = config['test_dataset']

    video_raw = sorted([d for d in Path(spec['dataset']['raw']).glob('*') if d.is_dir()])
    video_compress = sorted([d for d in Path(spec['dataset']['compress']).glob('*') if d.is_dir()])
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    for idx in range(len(video_compress)):
        spec['dataset']['args']['lq_path'] = video_compress[idx]
        spec['dataset']['args']['gr_path'] = video_raw[idx]
        dataset = datasets.make(spec['dataset'])
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

        loader = DataLoader(dataset, batch_size=spec['batch_size'],
            num_workers=1, pin_memory=True)

        res = eval_psnr(loader, loss_fn_alex, model,
            data_norm=config.get('data_norm'),
            window_size=16,
            verbose=True,
            sample=args.sample,
            detail=args.detail
            )
        if args.detail:
            print(video_compress[idx])
            for key, val in res.items():
                print(key, ": {:.8f}".format(val))
                if key == 'Enhanced PSNR':
                    avg_psnr = avg_psnr + val
                elif key == 'Enhanced SSIM':
                    avg_ssim = avg_ssim + val
                elif key == 'Enhanced LPIPS':
                    avg_lpips = avg_lpips + val
    print(avg_psnr, avg_ssim, avg_lpips)


