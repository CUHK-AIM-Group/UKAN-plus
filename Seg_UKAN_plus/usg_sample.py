"""Sample images from the generation branch of a trained USG model.

DDPM ancestral sampling at the generation size of the run (64x64, T = 1000).
The PNG samples are used for FID (pytorch-fid) and IS, see run_usg_fid.sh.

Usage:
    python usg_sample.py --exp_dir outputs/usg_joint_dynamic_busi_seed2981
"""
import os, sys, argparse, yaml, cv2, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usg_train import build_model, linear_beta_schedule


@torch.no_grad()
def p_sample_loop(model, shape, device, T=1000):
    """DDPM reverse sampling loop."""
    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    B = shape[0]
    # Start from pure noise
    x_t = torch.randn(shape, device=device)

    for t_val in tqdm(reversed(range(T)), total=T, desc='Sampling'):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise_pred = model(x_t, mode='gen', t=t)

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

    return x_t.clamp(-1, 1)


def sample_and_save(model, save_dir, n_samples=2048, batch_size=64,
                    img_size=64, device='cuda', T=1000):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0
    while count < n_samples:
        bs = min(batch_size, n_samples - count)
        imgs = p_sample_loop(model, (bs, 3, img_size, img_size), device, T=T)
        imgs = ((imgs + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for img in imgs:
            # Samples follow the BGR channel order of the training images
            cv2.imwrite(os.path.join(save_dir, f'sample_{count}.png'), img)
            count += 1
    print(f"Saved {count} samples to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='Path to experiment output dir')
    parser.add_argument('--out_dir', default=None, help='Sample directory (default: <exp_dir>/Gens)')
    parser.add_argument('--n_samples', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(os.path.join(args.exp_dir, 'config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config['mode'] == 'seg_only':
        raise ValueError('the run was trained without the generation branch')

    model = build_model(config)
    state = torch.load(os.path.join(args.exp_dir, 'model.pth'), map_location=args.device)
    model.load_state_dict(state)
    model = model.to(args.device)

    sample_and_save(model, args.out_dir or os.path.join(args.exp_dir, 'Gens'),
                    n_samples=args.n_samples, batch_size=args.batch_size,
                    img_size=config['gen_size'], device=args.device, T=config['T'])
