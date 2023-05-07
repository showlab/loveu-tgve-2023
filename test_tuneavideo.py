from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid, ddim_inversion
from diffusers import DDIMScheduler

import os
import argparse
import torch
import decord
decord.bridge.set_bridge('torch')

from omegaconf import OmegaConf


def main(args):
    config = OmegaConf.load(args.config)
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    output_dir = config.output_dir
    pretrained_model_path = config.pretrained_model_path
    checkpoint_step = 300 # config.max_train_steps
    edited_prompts = config.validation_data.prompts
    height, width = config.train_data.height, config.train_data.width
    num_inference_steps = config.validation_data.num_inference_steps

    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder='unet').to(weight_dtype)
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, scheduler=scheduler, torch_dtype=weight_dtype).to("cuda")
    # load tuned parameters
    loaded_state_dict = torch.load(f"{output_dir}/checkpoints/tuneavideo-{checkpoint_step}.pth")
    new_state_dict = pipe.unet.state_dict()
    new_state_dict.update(loaded_state_dict)
    pipe.unet.load_state_dict(new_state_dict)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    inv_latents_path = f"{output_dir}/inv_latents/ddim_latent-{checkpoint_step}.pt"
    if not os.path.exists(inv_latents_path):
        raise FileNotFoundError(inv_latents_path)
    ddim_inv_latent = torch.load(inv_latents_path).to(weight_dtype)

    for edited_type, edited_prompt in edited_prompts.items():
        save_path = f"{output_dir}/results/{edited_type}/{edited_prompt}.gif"
        video = pipe(edited_prompt, latents=ddim_inv_latent, height=height, width=width, num_inference_steps=num_inference_steps,
                     video_length=ddim_inv_latent.shape[2], guidance_scale=args.cfg_scale).videos
        save_videos_grid(video, save_path)
        print(f"Saved output to {save_path}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--cfg_scale", type=float, default=12.5, help="classifier-free guidance scale")
    parser.add_argument("--fp16", action='store_true', help="use float16 for inference")
    args = parser.parse_args()

    main(args)