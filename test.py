import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import numpy as np

import torch.optim as optim
import torchvision
from backbones.ncsnpp_generator_adagn import NCSNpp

from dataset import CreateDatasetSynthesis

import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr


def mean_absolute_error(real_image, synthesized_image):
    # Compute the absolute difference between corresponding elements
    absolute_diff = np.abs(real_image - synthesized_image)

    # Calculate the mean of the absolute differences
    mae = np.mean(absolute_diff)

    return mae


def psnr_calc(fake, real):
    # Peak Signal to Noise Ratio
    psnr_val = psnr(real, fake, data_range=1.0)
    return psnr_val


def ssim_val(img1, img2):

    ssim_score = ssim(np.squeeze(img1), np.squeeze(img2), data_range=1.0)
    return ssim_score


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


def sample_posterior_combine(coefficients, x_0_1, x_0_2, x_t, t):
    def q_posterior(x_0_1, x_0_2, x_t, t):
        mean1 = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_1
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        mean2 = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_2
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        mean = (mean1 + mean2) / 2
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

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
            x_0_1 =generator1(torch.cat((x, cond1), axis=1), t_time, latent_z)
        
            x_new = sample_posterior(coefficients, x_0_1[:, [0], :], x, t)
            x = x_new.detach()

    return x


def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch, device='cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()


# %%
def sample_and_test(args):
    torch.manual_seed(42)
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen = args.which_epoch

    to_range_0_1 = lambda x: (x + 1.) / 2.

    # loading dataset
    phase = 'test'
    dataset = CreateDatasetSynthesis('test', args.input_path)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4)
    # Initializing and loading network
    gen_diffusive = NCSNpp(args).to(device)
   
    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir, exp)

    checkpoint_file = exp_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, gen_diffusive, 'gen_diffusive', epoch=str(epoch_chosen), device=device)
   
    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    save_dir = exp_path + "/generated_samples/epoch_{}".format(epoch_chosen)

    crop = transforms.CenterCrop((256, 256))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    loss1 = np.zeros((1, len(data_loader)))
    loss2 = np.zeros((1, len(data_loader)))
    loss3 = np.zeros((1, len(data_loader)))

    for iteration, (x1, y) in enumerate(data_loader):
        cond_data1 = x1.to(device, non_blocking=True)
        real_data = y.to(device, non_blocking=True)

        x1_t = torch.randn_like(real_data)

        fake_sample = sample_from_model(pos_coeff, gen_diffusive, cond_data1, 
                                         args.num_timesteps, x1_t, T, args)

        fake_sample = to_range_0_1(fake_sample);
        fake_sample = fake_sample / fake_sample.max()
        real_data = to_range_0_1(real_data);
        real_data = real_data / real_data.max()
        x1 = to_range_0_1(x1);
        x1 = x1 / x1.max()
       
        fake_sample = crop(fake_sample).cuda()
        real_data = crop(real_data).cuda()
        x1 = crop(x1).cuda()
       
      

        loss1[0, iteration] = psnr_calc(fake_sample.cpu().numpy(), real_data.cpu().numpy())
        loss2[0, iteration] = ssim_val(real_data.cpu().numpy(), fake_sample.cpu().numpy())
        loss3[0, iteration] = mean_absolute_error(real_data.cpu().numpy(), fake_sample.cpu().numpy())
        print(str(iteration))


        fake_sample = torch.cat((x1, fake_sample, real_data), axis=-1)
        torchvision.utils.save_image(fake_sample, '{}/{}_samples2_{}.jpg'.format(save_dir, phase, iteration),
                                     normalize=True)

    print(np.nanmean(loss1))
    print(np.nanstd(loss1))
    np.save('{}/psnr_values.npy'.format(save_dir), loss1)

    print(np.nanmean(loss2))
    print(np.nanstd(loss2))
    np.save('{}/ssim_values.npy'.format(save_dir), loss2)

    print(np.nanmean(loss3))
    print(np.nanstd(loss3))
    np.save('{}/mae_values.npy'.format(save_dir), loss3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int, default=1000)
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
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')

    # optimizaer parameters
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')

    parser.add_argument('--which_epoch', type=int, default=50)
    parser.add_argument('--gpu_chose', type=int, default=0)

    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast')
    args = parser.parse_args()

    sample_and_test(args)

