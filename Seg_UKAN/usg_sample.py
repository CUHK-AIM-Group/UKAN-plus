"""Sample images from trained USG models and compute FID/IS."""
import os, sys, argparse, yaml, torch, numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archs
from usg_train import GenDecoder, SimpleDDPM, USGModel, linear_beta_schedule


def build_model(config, device):
    embed_dims = [int(x) for x in config['embed_dims'].split(',')]
    grid_sizes = [int(x) for x in config['grid_sizes'].split(',')]
    seg_model = archs.UKANPP(
        num_classes=1, input_channels=3,
        embed_dims=embed_dims, grid_sizes=grid_sizes,
        fusion_type=config.get('fusion_type', 'softmax'),
        no_kan=config.get('no_kan', False))
    gen_decoder = GenDecoder(embed_dim=embed_dims[-1], img_channels=3, T=config.get('T', 1000))
    model = USGModel(seg_model, gen_decoder)
    return model


@torch.no_grad()
def p_sample_loop(model, ddpm, shape, device, T=1000):
    """DDPM reverse sampling loop."""
    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])

    B = shape[0]
    # Start from pure noise
    x_t = torch.randn(shape, device=device)

    # Need a dummy input for encoder — use x_t itself (noisy image as encoder input)
    for t_val in tqdm(reversed(range(T)), total=T, desc='Sampling'):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Use x_t as both encoder input and noisy image
        bottleneck, skips = model.encode(x_t)
        noise_pred = model.decode_gen(bottleneck, x_t, t)

        # DDPM reverse step
        beta_t = betas[t_val]
        alpha_t = alphas[t_val]
        alpha_bar_t = alphas_bar[t_val]

        # Mean
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coeff1 * (x_t - coeff2 * noise_pred)

        if t_val > 0:
            sigma = torch.sqrt(beta_t)
            x_t = mean + sigma * torch.randn_like(x_t)
        else:
            x_t = mean

    return x_t


def sample_and_save(model, ddpm, save_dir, n_samples=2048, batch_size=32,
                    img_size=256, device='cuda', T=1000):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0
    while count < n_samples:
        bs = min(batch_size, n_samples - count)
        imgs = p_sample_loop(model, ddpm, (bs, 3, img_size, img_size), device, T=T)
        imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        imgs = (imgs * 255).byte().cpu().numpy()
        for i in range(bs):
            img = imgs[i].transpose(1, 2, 0)  # CHW -> HWC
            Image.fromarray(img).save(os.path.join(save_dir, f'sample_{count}.png'))
            count += 1
    print(f"Saved {count} samples to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='Path to experiment output dir')
    parser.add_argument('--n_samples', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--T', type=int, default=1000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = os.path.join(args.exp_dir, 'config.yml')
    ckpt_path = os.path.join(args.exp_dir, 'model.pth')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = build_model(config, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    ddpm = SimpleDDPM(T=args.T, device=device)

    save_dir = os.path.join(args.exp_dir, 'gen_samples')
    sample_and_save(model, ddpm, save_dir, n_samples=args.n_samples,
                    batch_size=args.batch_size, img_size=args.img_size,
                    device=device, T=args.T)
