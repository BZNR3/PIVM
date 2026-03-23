import os
import argparse
import re
from collections import defaultdict
from glob import glob

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from utils import load_checkpoint, save_images, save_checkpoint, DDPMDataset, MaskDataset
from DDPM_model import DDPM, Discriminator

torch.manual_seed(1)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.fixed_noise = torch.randn(1, 3, img_size, img_size).to(device)

        self.beta = self.prepare_linear_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_linear_noise_schedule(self):
        return torch.abs(
            torch.cos(torch.linspace(0, torch.pi / 2, self.noise_steps)) * self.beta_end
            - (self.beta_end - self.beta_start)
        )
        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Forward diffusion: add Gaussian noise to x at timestep t.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps = torch.randn_like(x)
        noised = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return noised, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, images, prevs, labels, organs, n, counter):
        """
        Sample intermediate qualitative results during training.

        The model predicts the residual between image and label:
            residual = image - label

        During sampling, the reconstructed output is:
            generated = predicted_residual + label
        """
        images = images[:n]
        prevs = prevs[:n]
        labels = labels[:n]
        organs = organs[:n]

        model.eval()
        with torch.no_grad():
            x = torch.randn((images.shape[0], 1, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(images.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, organs, prevs)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1 / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )

        model.train()

        generated = x + labels

        generated_vis = (generated.clamp(-1, 1) + 1) / 2
        generated_vis = (generated_vis * 255).type(torch.uint8)

        image_vis = (images.clamp(-1, 1) + 1) / 2
        image_vis = (image_vis * 255).type(torch.uint8)

        label_vis = (labels.clamp(-1, 1) + 1) / 2
        label_vis = (label_vis * 255).type(torch.uint8)

        os.makedirs("results", exist_ok=True)
        save_images(generated_vis, os.path.join("results", f"{counter}_generated.png"))
        save_images(image_vis, os.path.join("results", f"{counter}_image.png"))
        save_images(label_vis, os.path.join("results", f"{counter}_label.png"))

    def generate(self, model, images, labels, organs, n, filenames):
        """
        Generate outputs for paired test samples.
        """
        images = images[:n]
        labels = labels[:n]
        organs = organs[:n]

        model.eval()
        with torch.no_grad():
            x = torch.randn((labels.shape[0], 1, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(labels.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, organs)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1 / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )

        generated = labels + x

        noise_vis = (x.clamp(-1, 1) + 1) / 2
        noise_vis = (noise_vis * 255).type(torch.uint8)

        image_vis = (images.clamp(-1, 1) + 1) / 2
        image_vis = (image_vis * 255).type(torch.uint8)

        label_vis = (labels.clamp(-1, 1) + 1) / 2
        label_vis = (label_vis * 255).type(torch.uint8)

        generated_vis = (generated.clamp(-1, 1) + 1) / 2
        generated_vis = (generated_vis * 255).type(torch.uint8)

        os.makedirs(os.path.join("results", "generated"), exist_ok=True)
        os.makedirs(os.path.join("results", "image"), exist_ok=True)
        os.makedirs(os.path.join("results", "label"), exist_ok=True)
        os.makedirs(os.path.join("results", "noise"), exist_ok=True)

        for image, noise, label, organ, gen, filename in zip(
            image_vis, noise_vis, label_vis, organs, generated_vis, filenames
        ):
            base_name = os.path.splitext(os.path.basename(filename))[0]
            save_images(gen, os.path.join("results", "generated", f"{base_name}.png"))
            save_images(image, os.path.join("results", "image", f"{base_name}.png"))
            save_images(label, os.path.join("results", "label", f"{base_name}.png"))
            save_images(noise, os.path.join("results", "noise", f"{base_name}.png"))

    def generate_sequence(self, model, initial_image, label_images, organ_images, save_dir, case_id=None):
        """
        Sequentially generate image-like slices conditioned on previous results.

        Args:
            model: trained DDPM model
            initial_image: first frame used as the initial condition
            label_images: tensor of label slices [N, C, H, W]
            organ_images: tensor of organ-mask slices [N, C, H, W]
            save_dir: directory to save generated results
            case_id: optional case identifier
        """
        model.eval()
        generated_images = [initial_image]

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "noise"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)

        with torch.no_grad():
            prev_image = initial_image.to(self.device)

            for idx, label in enumerate(tqdm(label_images, desc="Generating image sequence")):
                label = label.unsqueeze(0).to(self.device)
                organ = organ_images[idx].unsqueeze(0).to(self.device)

                x = torch.randn((1, 1, self.img_size, self.img_size)).to(self.device)

                for i in reversed(range(1, self.noise_steps)):
                    t = (torch.ones(1) * i).long().to(self.device)
                    predicted_noise = model(x, t, organ, prev_image)

                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]

                    noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                    x = (
                        (1 / torch.sqrt(alpha))
                        * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                        + torch.sqrt(beta) * noise
                    )

                generated_image = label + x

                final_image = (generated_image.clamp(-1, 1) + 1) / 2
                final_image = (final_image * 255).type(torch.uint8)

                x_vis = (x.clamp(-1, 1) + 1) / 2
                x_vis = (x_vis * 255).type(torch.uint8)

                label_vis = (label.clamp(-1, 1) + 1) / 2
                label_vis = (label_vis * 255).type(torch.uint8)

                if case_id is not None:
                    save_images(final_image, os.path.join(save_dir, f"{case_id}_{idx + 2}.png"))
                    save_images(x_vis, os.path.join(save_dir, "noise", f"{case_id}_{idx + 2}.png"))
                    save_images(label_vis, os.path.join(save_dir, "label", f"{case_id}_{idx + 2}.png"))
                else:
                    save_images(final_image, os.path.join(save_dir, f"{idx + 2}.png"))
                    save_images(x_vis, os.path.join(save_dir, "noise", f"{idx + 2}.png"))
                    save_images(label_vis, os.path.join(save_dir, "label", f"{idx + 2}.png"))

                prev_image = generated_image
                generated_images.append(generated_image.cpu())

        print(f"[Saved] Case {case_id}: {len(label_images)} slices saved to {save_dir}")
        return generated_images


def group_by_case(file_paths):
    """
    Group image paths by case ID extracted from filenames.

    Example:
        s0001_214_1.png -> case_id = "s0001_214"

    Returns:
        dict[str, list[str]]: mapping from case_id to a sorted list of paths
    """
    case_groups = defaultdict(list)

    for path in file_paths:
        filename = os.path.basename(path)
        match = re.match(r"(s\d+_\d+)_\d+\.png", filename)
        if match:
            case_id = match.group(1)
            case_groups[case_id].append(path)

    for case_id in case_groups:
        case_groups[case_id] = sorted(
            case_groups[case_id],
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
        )

    return case_groups


def test_seq(args):
    device = args.device

    ddpm = DDPM(img_channels=3, time_dim=args.emb_dim).to(device)
    diffusion = Diffusion(noise_steps=1000, img_size=args.image_size, device=device)

    if args.load_model:
        load_checkpoint(
            args.checkpoint_path,
            ddpm,
            None,
            None,
        )

    image_paths = sorted(glob(os.path.join(args.image_dir, "*.png")))
    label_paths = sorted(glob(os.path.join(args.label_dir, "*.png")))
    organ_paths = sorted(glob(os.path.join(args.organ_dir, "*.png")))

    image_groups = group_by_case(image_paths)
    label_groups = group_by_case(label_paths)
    organ_groups = group_by_case(organ_paths)

    resume_from = args.resume_from
    skip = resume_from is not None

    for case_id in image_groups.keys():
        if skip:
            if case_id != resume_from:
                print(f"Skipping case: {case_id}")
                continue
            else:
                skip = False
                print(f"Resuming from case: {case_id}")

        print(f"Generating sequential results for case: {case_id}")

        image_case = image_groups[case_id]
        label_case = label_groups[case_id]
        organ_case = organ_groups[case_id]

        dataset = DDPMDataset(
            image_paths=image_case,
            label_paths=label_case,
            organ_paths=organ_case,
            img_size=args.image_size
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "noise"), exist_ok=True)

        first_batch = next(iter(dataloader))

        if len(first_batch) == 5:
            _, initial_image, _, _, _ = first_batch
        else:
            raise ValueError(
                "Unexpected dataset output format in test_seq(). "
                "Expected a 5-element batch such as (image, prev, label, organ, filename)."
            )

        label_images = torch.stack([data[2].squeeze(0) for data in dataloader][1:], dim=0)
        organ_images = torch.stack([data[3].squeeze(0) for data in dataloader][1:], dim=0)

        diffusion.generate_sequence(
            model=ddpm,
            initial_image=initial_image,
            label_images=label_images,
            organ_images=organ_images,
            save_dir=save_dir,
            case_id=case_id
        )

        print(f"Case {case_id}: generation complete, results saved to {save_dir}")


def test(args):
    device = args.device

    image_paths = sorted(glob(os.path.join(args.image_dir, "*.png")), key=len)
    label_paths = sorted(glob(os.path.join(args.label_dir, "*.png")), key=len)
    organ_paths = sorted(glob(os.path.join(args.organ_dir, "*.png")), key=len)

    dataset = DDPMDataset(
        image_paths=image_paths,
        label_paths=label_paths,
        organ_paths=organ_paths,
        img_size=args.image_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    pbar = tqdm(dataloader)

    ddpm = DDPM(img_channels=2, time_dim=args.emb_dim).to(device)
    diffusion = Diffusion(noise_steps=1000, img_size=args.image_size, device=device)

    if args.load_model:
        load_checkpoint(
            args.checkpoint_path,
            ddpm,
            None,
            None,
        )

    for j, batch in enumerate(pbar):
        if len(batch) == 4:
            images, labels, organs, filenames = batch
        else:
            raise ValueError(
                "Unexpected dataset output format in test(). "
                "Expected (image, label, organ, filename)."
            )

        images = images.to(device)
        labels = labels.to(device)
        organs = organs.to(device)

        diffusion.generate(ddpm, images, labels, organs, args.batch_size, filenames)


def train(args):
    global counter
    device = args.device

    image_paths = glob(os.path.join(args.image_dir, "*.png"))
    label_paths = glob(os.path.join(args.label_dir, "*.png"))
    organ_paths = glob(os.path.join(args.organ_dir, "*.png"))

    filtered_image_paths = [
        path for path in image_paths
        if not path.endswith('_1.png')
    ]
    filtered_label_paths = [
        path for path in label_paths
        if not path.endswith('_1.png')
    ]
    filtered_organ_paths = [
        path for path in organ_paths
        if not path.endswith('_1.png')
    ]

    dataset = DDPMDataset(
        image_paths=filtered_image_paths,
        label_paths=filtered_label_paths,
        organ_paths=filtered_organ_paths,
        img_size=args.image_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    ddpm = DDPM(img_channels=3, time_dim=args.emb_dim).to(device)
    optimizer = optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            args.checkpoint_path,
            ddpm,
            optimizer,
            args.lr,
        )

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    min_avg_loss = float("inf")

    diffusion = Diffusion(noise_steps=1000, img_size=args.image_size, device=device)

    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        count1 = 0
        count2 = 0

        for i, (images, prevs, labels, organs, _) in enumerate(pbar):
            images = images.to(device)
            prevs = prevs.to(device)
            labels = labels.to(device)
            organs = organs.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            residual = images - labels
            x_t, noise = diffusion.noise_images(residual, t)

            for rep in range(5):
                if rep == 1:
                    count1 += 1

                predicted_noise = ddpm(x_t, t, organs, prevs)
                loss = mse(noise, predicted_noise) + l1(noise, predicted_noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if loss < min_avg_loss:
                    if rep == 4:
                        count2 += 1
                    break

            avg_loss += loss.item()
            pbar.set_postfix(
                epoch=epoch,
                AVG_MSE=avg_loss / (i + 1),
                count1=count1,
                count2=count2,
                MIN_MSE=min_avg_loss
            )

            interval = max(1, (len(dataloader) - 1) // 2)
            if i % interval == 0 and i != 0:
                diffusion.sample(ddpm, images, prevs, labels, organs, n=8, counter=counter)
                counter += 1

        epoch_avg_loss = avg_loss / len(dataloader)

        if min_avg_loss > epoch_avg_loss:
            min_avg_loss = epoch_avg_loss
            save_checkpoint(
                ddpm,
                optimizer,
                filename=os.path.join(args.checkpoint, f"ddpm{epoch}.pth.tar")
            )


if __name__ == '__main__':
    training = False
    counter = 0

    if training:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.load_model = True
        args.epochs = 500
        args.batch_size = 1
        args.emb_dim = 256 * 1
        args.image_size = 256 * 1
        args.num_workers = 4

        args.checkpoint = "results/checkpoints"
        args.checkpoint_path = os.path.join(args.checkpoint, "ddpm3.pth.tar")

        args.image_dir = "data/train/image"
        args.label_dir = "data/train/label"
        args.organ_dir = "data/train/organ"

        args.dataset_path = None
        args.generated_path = None
        args.output_dir = "results"
        args.device = "cuda"
        args.lr = 2e-5

        train(args)

    else:
        test_parser = argparse.ArgumentParser()
        test_args = test_parser.parse_args()

        test_args.load_model = True
        test_args.emb_dim = 256 * 1
        test_args.num_iters = 1000
        test_args.batch_size = 1
        test_args.image_size = 256 * 1
        test_args.num_workers = 4

        test_args.checkpoint = "results/checkpoints"
        test_args.checkpoint_path = os.path.join(test_args.checkpoint, "ddpm27.pth.tar")

        test_args.image_dir = "data/test/image"
        test_args.label_dir = "data/test/label"
        test_args.organ_dir = "data/test/organ"

        test_args.output_dir = "results/allresults"
        test_args.resume_from = None
        test_args.device = "cuda"

        test_seq(test_args)