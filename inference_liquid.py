import glob
import os
import warnings
from time import time

import cv2
import numpy as np
import torch
from liquidnoise import *
from PIL import Image


def batched_interpret_optical_flow_col(steps, fmap):
    N = len(steps)
    fmap = np.uint8(np.array(fmap))
    hsv = torch.tensor(cv2.cvtColor(fmap, cv2.COLOR_RGB2HSV)).float().cuda()
    hsv[:, :, 0] *= 2

    distances = torch.zeros((N, 512, 512, 2)).cuda()
    hue = hsv[:, :, 0]
    red_mask = (315 < hue) | (hue <= 45)
    cyan_mask = (135 < hue) & (hue <= 225)
    magenta_mask = (225 < hue) & (hue <= 315)
    lime_mask = (45 < hue) & (hue <= 135)
    periodic_mask = hsv[:, :, 2] < 255

    for i, step in enumerate(steps):
        if step == 0:
            continue
        distances[i, :, :, 1] = torch.where(
            red_mask, 4 * step * hsv[:, :, 1] / 100, distances[i, :, :, 1]
        )
        distances[i, :, :, 1] = torch.where(
            cyan_mask, -4 * step * hsv[:, :, 1] / 100, distances[i, :, :, 1]
        )
        distances[i, :, :, 0] = torch.where(
            magenta_mask, -4 * step * hsv[:, :, 1] / 100, distances[i, :, :, 0]
        )
        distances[i, :, :, 0] = torch.where(
            lime_mask, 4 * step * hsv[:, :, 1] / 100, distances[i, :, :, 0]
        )
        distances[i, :, :, 0] = torch.where(
            periodic_mask,
            (distances[i, :, :, 0] / step) * torch.sin(30 * step / hsv[:, :, 2]),
            distances[i, :, :, 0],
        )
        distances[i, :, :, 1] = torch.where(
            periodic_mask,
            (distances[i, :, :, 1] / step) * torch.sin(30 * step / hsv[:, :, 2]),
            distances[i, :, :, 1],
        )

    return torch.round(distances).int()


def batched_compute_flow(
    model, fmap, wrap, switch_percent, roll, x_t_orig, noising=False
):
    distances = batched_interpret_optical_flow_col(roll, fmap)
    x_t = x_t_orig.clone()
    scale_factor = -2 * (switch_percent / 100) + 3

    if not noising:
        x_t /= scale_factor

    x_t *= 1 / 0.18215
    x_t_decoded = model.vae.decode(x_t)["sample"].clone()
    x_t_decoded = torch.cat(
        [
            flow_mapping(distances[i], x_t_decoded[i : i + 1], wrap)
            for i in range(len(roll))
        ]
    )

    if noising:
        x_t_decoded += 0.1 * torch.randn_like(x_t_decoded)

    x_t_decoded = model.vae.encode(x_t_decoded)["latent_dist"].mean.detach()
    x_t_decoded *= 0.18215

    if noising:
        x_t_decoded += 0.05 * torch.randn_like(x_t_decoded)

    x_t_decoded = normalize(x_t_orig, x_t_decoded)
    return x_t_decoded


warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

model = Model().ldm_stable
seed = 1

prompt = "Acacia trees in the savannah, long grass, sky with small clouds"
control_image = Image.open("control/sway_control.png").convert("RGB")
ctrl_flow = Image.open("control/sway_flow.png").convert("RGB")
batch_size = 1
num_inference_steps = 30
guidance_scale = 7.5
generator = torch.Generator().manual_seed(seed)
controller = EmptyControl()
wrap = False
switch_percent = 70
control_func = partial(control_flow, ctrl_flow, wrap)
noise_func = partial(
    batched_compute_flow, model, ctrl_flow, wrap, switch_percent, noising=False
)
rolled_controls = [control_func(roll, control_image) for roll in range(16)]
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

context = torch.cat([uncond_embeddings, text_embeddings])

model.scheduler.set_timesteps(30)
latents = torch.randn(1, 4, 64, 64).to(model.device)

# ControlNet
control = control_image
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

t_switch = int(num_inference_steps * switch_percent / 100)
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
latents = batched_compute_flow(
    model,
    ctrl_flow,
    wrap,
    switch_percent,
    noising=False,
    roll=list(range(16)),
    x_t_orig=latents,
)
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
