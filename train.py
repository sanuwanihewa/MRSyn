import argparse
import torch
import numpy as np

import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from dataset import CreateDatasetSynthesis

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos

    def p_sample(x_0_1, x_0_2, x_t, t):
        mean, _, log_var = q_posterior(x_0_1, x_0_2, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0_1, x_0_2, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator1, cond1, n_time, x_init, T, opt):
    x = x_init

    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)
            x_0_1 = generator1(torch.cat((x, cond1), axis=1), t_time, latent_z)

            x_new = sample_posterior(coefficients, x_0_1[:, [0], :], x, t)
            x = x_new.detach()

    return x


def uncer_loss(mean, var, label):
    loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
    loss2 = var
    loss = .5 * (loss1 + loss2)
    return loss.mean()


# %%
def train_syndiff(rank, gpu, args):
    from backbones.discriminator import Discriminator_small, Discriminator_large

    from backbones.ncsnpp_generator_adagn import NCSNpp

    from utils.EMA import EMA

    # rank = args.node_rank * args.num_process_per_node + gpu

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    dataset = CreateDatasetSynthesis(phase="train", input_path=args.input_path)
    dataset_val = CreateDatasetSynthesis(phase="test", input_path=args.input_path)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  sampler=val_sampler,
                                                  drop_last=True)

    val_l1_loss = np.zeros([2, args.num_epoch, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch, len(data_loader_val)])
    print('train data size:' + str(len(data_loader)))
    print('val data size:' + str(len(data_loader_val)))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # networks performing reverse denoising
    gen_diffusive = NCSNpp(args).to(device)

    # networks performing translation
    args.num_channels = 1

    disc_diffusive = Discriminator_large(nc=2, ngf=args.ngf,
                                         t_emb_dim=args.t_emb_dim,
                                         act=nn.LeakyReLU(0.2)).to(device)

    broadcast_params(gen_diffusive.parameters())

    broadcast_params(disc_diffusive.parameters())

    optimizer_disc_diffusive = optim.Adam(disc_diffusive.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizer_gen_diffusive = optim.Adam(gen_diffusive.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen_diffusive = EMA(optimizer_gen_diffusive, ema_decay=args.ema_decay)

    scheduler_gen_diffusive = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive, args.num_epoch,
                                                                         eta_min=1e-5)

    scheduler_disc_diffusive = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive, args.num_epoch,
                                                                          eta_min=1e-5)

    # ddp
    gen_diffusive = nn.parallel.DistributedDataParallel(gen_diffusive, device_ids=[gpu])

    disc_diffusive = nn.parallel.DistributedDataParallel(disc_diffusive, device_ids=[gpu])

    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive.load_state_dict(checkpoint['gen_diffusive_1_dict'])

        # load G

        optimizer_gen_diffusive.load_state_dict(checkpoint['optimizer_gen_diffusive'])
        scheduler_gen_diffusive.load_state_dict(checkpoint['scheduler_gen_diffusive'])

        # load D

        disc_diffusive.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive.load_state_dict(checkpoint['optimizer_disc_diffusive'])
        scheduler_disc_diffusive.load_state_dict(checkpoint['scheduler_disc_diffusive'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, args.num_epoch + 1):
        # train_sampler.set_epoch(epoch)

        for iteration, (x1, x2) in enumerate(data_loader):

            for p in disc_diffusive.parameters():
                p.requires_grad = True

            disc_diffusive.zero_grad()

            # sample from p(x_0)
            cond_data1 = x1.to(device, non_blocking=True)


            real_data = x2.to(device, non_blocking=True)

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train discriminator with real
            D_real = disc_diffusive(x_t, t, x_tp1.detach())

            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:

                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake

            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict_diff_g = gen_diffusive(
                torch.cat((x_tp1.detach(), cond_data1), axis=1), t, latent_z)

            # sampling q(x_t | x_0_predict, x_t+1)
            x2_pos_sample_g = sample_posterior(pos_coeff, x_0_predict_diff_g[:, [0], :], x_tp1, t)

            # D output for fake sample x_pos_sample
            output_g = disc_diffusive(x2_pos_sample_g, t, x_tp1.detach())

            errD2_fake2_g = (F.softplus(output_g)).mean()

            errD2_fake2_g.backward()

            optimizer_disc_diffusive.step()

            for p in disc_diffusive.parameters():
                p.requires_grad = False

            cond_data1 = x1.to(device, non_blocking=True)

            real_data = x2.to(device, non_blocking=True)

            gen_diffusive.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            # sample x_t and x_tp1
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict_diff_g = gen_diffusive(
                torch.cat((x_tp1.detach(), cond_data1), axis=1), t, latent_z)

            # sampling q(x_t | x_0_predict, x_t+1)
            x2_pos_sample_g = sample_posterior(pos_coeff, x_0_predict_diff_g[:, [0], :], x_tp1, t)

            output_g = disc_diffusive(x2_pos_sample_g, t, x_tp1.detach())

            errG = F.softplus(-output_g)
            errG = errG.mean()

            errG_adv = errG

            errG_L1 = F.l1_loss(x_0_predict_diff_g[:, [0], :], real_data)

            errG = errG_adv + (args.lambda_l1_loss * errG_L1)

            errG.backward()
            optimizer_gen_diffusive.step()

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{},  G-Adv: {}, G-Sum: {}'.format(epoch, iteration,
                                                                               errG_adv.item(), errG.item()))
            break

        if not args.no_lr_decay:
            scheduler_gen_diffusive.step()

            scheduler_disc_diffusive.step()

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x2_pos_sample_g,
                                             os.path.join(exp_path, 'xposg_epoch_{}.png'.format(epoch)),
                                             normalize=True)

            x_t = torch.randn_like(real_data)
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive, cond_data1,
                                             args.num_timesteps, x_t, T, args)

            fake_sample1 = torch.cat((real_data, fake_sample1), axis=-1)

            torchvision.utils.save_image(fake_sample1,
                                         os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)),
                                         normalize=True)

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'gen_diffusive_dict': gen_diffusive.state_dict(),
                               'optimizer_gen_diffusive': optimizer_gen_diffusive.state_dict(),
                               'scheduler_gen_diffusive': scheduler_gen_diffusive.state_dict(),

                               'disc_diffusive_dict': disc_diffusive.state_dict(),
                               'optimizer_disc_diffusive': optimizer_disc_diffusive.state_dict(),
                               'scheduler_disc_diffusive': scheduler_disc_diffusive.state_dict(),
                               }

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_gen_diffusive.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(gen_diffusive.state_dict(), os.path.join(exp_path, 'gen_diffusive_{}.pth'.format(epoch)))

                if args.use_ema:
                    optimizer_gen_diffusive.swap_parameters_with_ema(store_params_in_ema=True)

        for iteration, (cond, x_val) in enumerate(data_loader_val):
            real_data = x_val.to(device, non_blocking=True)
            source_data1 = cond.to(device, non_blocking=True)
           

            x1_t = torch.randn_like(real_data)
            # diffusion steps
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive, source_data1,
                                             args.num_timesteps, x1_t, T, args)

            fake_sample1 = to_range_0_1(fake_sample1);
            fake_sample1 = fake_sample1 / fake_sample1.mean()
            real_data = to_range_0_1(real_data);
            real_data = real_data / real_data.mean()

            fake_sample1 = fake_sample1.cpu().numpy()
            real_data = real_data.cpu().numpy()
            val_l1_loss[1, epoch, iteration] = abs(fake_sample1 - real_data).mean()

            val_psnr_values[1, epoch, iteration] = psnr(real_data, fake_sample1, data_range=real_data.max())

        print(np.nanmean(val_psnr_values[0, epoch, :]))
        print(np.nanmean(val_psnr_values[1, epoch, :]))
        np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
        np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='/data/shew0029/MedSyn/DATA/BRATS/')
    parser.add_argument('--output_path', help='/data/shew0029/MedSyn/VQGAN_3D/mrisyn/results')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5,
                        help='weightening of l1 loss part of diffusion ans cycle models')
    parser.add_argument('--lambda_mask_loss', type=float, default=0.01,
                        help='weightening of l1 loss part of diffusion ans cycle models')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--port_num', type=str, default='6091',
                        help='port selection for code')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train_syndiff, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:

        init_processes(0, size, train_syndiff, args)
