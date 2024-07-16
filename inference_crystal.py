import glob
import os
import warnings
from time import time

import cv2
import numpy as np
import torch
from liquidnoise import *
from PIL import Image

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

model = Model().ldm_stable
seed = 1


prompt = "Photo of a rock in a field of grass"
control_image = Image.open("control/control_image_seed_1.jpg")
batch_size = 1
num_inference_steps = 30
guidance_scale = 7.5
generator = torch.Generator().manual_seed(42)
controller = EmptyControl()
switch_percent = 70
control_func = partial(translate_entire_control, 8)
noise_func = translate_entire_noise
rolled_controls = [control_image] + [
    control_func(roll, control_image) for roll in range(15)
]


# Text input
text_input = model.tokenizer(
    prompt,
    padding="max_length",
    max_length=model.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = model.tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

# context = [uncond_embeddings, text_embeddings]
context = torch.cat([uncond_embeddings, text_embeddings])

model.scheduler.set_timesteps(30)
latents = torch.randn(1, 4, 64, 64).to(model.device)

# ControlNet
control = control_image
# diffusion_step(model, controller, latents, curr_control, context, t, guidance_scale, generator)
# def diffusion_step(model, controller, latents, control, context, t, guidance_scale, generator=None):
width, height = control.size
control_images = torch.cat(
    [
        model.prepare_image(
            image=c,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=torch.device("cuda:0"),
            dtype=model.controlnet.dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        )
        for c in rolled_controls
    ]
)

t_switch = int(num_inference_steps * 0.70)
# t_switch = num_inference_steps
print("using t_switch", t_switch)
t_pre_switch = [model.scheduler.timesteps[e] for e in range(t_switch)]
t_post_switch = [
    model.scheduler.timesteps[e] for e in range(t_switch, num_inference_steps)
]
print("t_pre_switch", t_pre_switch)
print("t_post_switch", t_post_switch)
start = time()
for t in tqdm(t_pre_switch):
    down_block_res_samples, mid_block_res_sample = model.controlnet(
        latents,
        t,
        encoder_hidden_states=context[1:],
        controlnet_cond=control_images[0],
        conditioning_scale=model.cond_scale,
        guess_mode=False,
        return_dict=False,
    )
    pair = model.unet(
        latents.repeat(2, 1, 1, 1),
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    noise_pred_uncond, noise_prediction_text = pair.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents, generator=generator)[
        "prev_sample"
    ]
    if controller is not None:
        latents = controller.step_callback(latents)


latents = latents.repeat(16, 1, 1, 1)
latents = torch.stack([torch.roll(latents[i], -i, 2) for i in range(16)])
# context = context.repeat_interleave(16,0)
context = context.repeat(16, 1, 1)

for t in tqdm(t_post_switch):
    down_block_res_samples, mid_block_res_sample = model.controlnet(
        latents,
        t,
        encoder_hidden_states=context[0::2],
        controlnet_cond=control_images,
        conditioning_scale=model.cond_scale,
        guess_mode=False,
        return_dict=False,
    )
    down_block_res_samples = [e.repeat(2, 1, 1, 1) for e in down_block_res_samples]
    mid_block_res_sample = mid_block_res_sample.repeat(2, 1, 1, 1)
    # down_block_res_samples = [e.repeat_interleave(2,0) for e in down_block_res_samples]
    # mid_block_res_sample = mid_block_res_sample.repeat_interleave(2,0)

    pair = model.unet(
        latents.repeat(2, 1, 1, 1),
        t,
        encoder_hidden_states=context,
        # latents.repeat_interleave(2,0), t, encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    noise_pred_uncond, noise_prediction_text = pair.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents, generator=generator)[
        "prev_sample"
    ]
    if controller is not None:
        latents = controller.step_callback(latents)
end = time()
print("Time taken", end - start)
print("Time per frame", (end - start) / 16)
# image = latent2image(model.vae, latents)
# view_image(image[0])
# view_image(image[-1])
