# MIT License

# Copyright (c) 2024 Muhammas Haaris Khan (https://github.com/Strikewind)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from scipy import interpolate
from PIL import Image, ImageFilter
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ptp_utils_new as ptp_utils
from ptp_utils_new import *

torch.cuda.empty_cache()
device = torch.device('cuda:0')
DISPLAY = True
NUM_DIFFUSION_STEPS = 20
GUIDANCE_SCALE = 8.0
COND_SCALE = 1.0
SEED = 1

class Model():

	def __init__(self):
		self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
		self.ldm_stable = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True, controlnet=self.controlnet)
		self.ldm_stable.scheduler = DDIMScheduler(
			beta_start=0.00085,
			beta_end=0.012,
			beta_schedule="scaled_linear",
			steps_offset=1,
			clip_sample=False)
		self.ldm_stable.to(device)
		self.ldm_stable.cond_scale = COND_SCALE


def run_and_display(prompt, model, controller=EmptyControl(), latent=None, control=None, 
					num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, seed=SEED):
	# Run diffusion on one prompt and display the result. ControlNet optional.
    if type(prompt) is str:
        prompt = [prompt]
    generator=torch.Generator().manual_seed(seed)
    images, _, x_t = ptp_utils.text2image_ldm_stable(model=model, prompt=prompt, controller=controller, latent=latent, control=control, 
													 num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
    ptp_utils.view_images(images)
    torch.cuda.empty_cache()
    return images, x_t

##########################################################################################################################
#-------------------------------------------------Plotting and Latents---------------------------------------------------#
##########################################################################################################################


def show_histogram(x_t):
	# Useful for analysing the distribution of latents (debug purposes)
	if torch.is_tensor(x_t):
		x_t = x_t.cpu().numpy()
	plt.hist(x_t.flatten(), bins=100)
	plt.xlim(-4.5, 4.5)
	print("mean", x_t.flatten().mean())
	print("std", x_t.flatten().std())
	plt.show()

def q_q_plot(x_t):
	# Useful for analysing the distribution of latents (debug purposes)
	if torch.is_tensor(x_t):
		x_t = x_t.cpu().numpy()
	sorted_x_t = np.sort(x_t.flatten())
	gaussian_quantiles = sorted(np.random.normal(0, 1, x_t.size).flatten())
	plt.figure(figsize=(6, 6))
	plt.scatter(gaussian_quantiles, sorted_x_t, s=0.2)
	plt.xlabel('Theoretical Quantiles')
	plt.ylabel('Ordered Values')
	plt.ylim([-4.5,4.5])
	plt.xlim([-4.5,4.5])
	plt.title('Q-Q Plot')
	plt.grid(True)
	plt.show()

def gif_to_wide_image(gif_path, space_between=20):
	# splits a gif into a wide image sequence
	gif = imageio.mimread(gif_path)
	frame_width, frame_height = gif[0].shape[1], gif[0].shape[0]
	total_width = len(gif) * frame_width + (len(gif) - 1) * space_between
	wide_image = Image.new('RGB', (total_width, frame_height), 'white')
	for i, frame in enumerate(gif):
		frame_image = Image.fromarray(frame)
		wide_image.paste(frame_image, (i * (frame_width + space_between), 0))
	display(wide_image)

def view_image(image):
	if type(image) is torch.Tensor:
		image = image.cpu().numpy()
	if type(image) is np.ndarray:
		if image.ndim == 4: # Expecting image straight from CUDA with batch dimension
			image = image[0]
		image = image[0].astype(np.uint8)
		image = Image.fromarray(image)
	display(image)

def view_noise(noise):
	# Takes 3 out of 4 channels of noise and displays it. Scaling is arbitrary.
	noise2 = noise.detach().clone().cpu()
	noise2 = (noise2 / 4 + 0.5).clamp(0, 1)
	noise2 = noise2.cpu().permute(0, 2, 3, 1).numpy()
	noise2 = (noise2 * 255).astype(np.uint8)
	noise2 = Image.fromarray(noise2[0]).resize((256, 256), Image.NEAREST).convert('RGB')
	display(noise2)

def return_noise(noise, channel=None):
	noise2 = noise.detach().clone().cpu()
	noise2 = (noise2 / 4 + 0.5).clamp(0, 1)
	noise2 = noise2.cpu().permute(0, 2, 3, 1).numpy()
	noise2 = (noise2 * 255).astype(np.uint8)
	if channel is not None:
		for i in range(4):
			if i != channel:
				noise2[:, :, :, i] = noise2[:, :, :, channel]
	noise2 = Image.fromarray(noise2[0]).resize((256, 256), Image.NEAREST).convert('RGB')
	return noise2

def view_noise_channels(noise):
	# View all 4 channels of noise in greyscale
	noise2 = noise.detach().clone().cpu()
	noise2 = (noise2 / 4 + 0.5).clamp(0, 1)
	noise2 = noise2.cpu().permute(0, 2, 3, 1).numpy()
	noise2 = (noise2 * 255).astype(np.uint8)
	channels = np.concatenate([noise2[0, :, :, i] for i in range(4)], axis=1)
	channels = Image.fromarray(channels).resize((1024, 256), Image.NEAREST).convert('RGB')
	display(channels)

def latents_to_rgb(latents):
	# Adapted from https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#direct-conversion-of-sdxl-latents-to-rgb-with-a-linear-approximation
	weights = ( # found using VAE_weights_finder.py
		(43.8964,  16.3552, -35.4404, -21.6123),
		(29.6584,  44.5785,  32.1088, -29.1559),
		(36.2716,   5.5332,  28.2693, -82.5393)
		)
	biases = (123.5403, 111.4831,  98.5240)
	weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
	biases_tensor = torch.tensor(biases, dtype=latents.dtype).to(latents.device)
	rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
	image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
	image_array = image_array.transpose(1, 2, 0)
	return Image.fromarray(image_array).resize((256, 256), Image.NEAREST).convert('RGB')

def create_latent(model, seed=SEED):
	generator=torch.Generator().manual_seed(seed)
	latent, _ = ptp_utils.init_latent(None, model, 512, 512, generator)
	return latent.detach().cuda()

def discrete_latent(model, seed=SEED):
	# Snap latent to a point in VAE latent space. Useful for Experimentation.
	x_t = create_latent(seed=seed)
	x_t = 1/0.18215 * x_t
	with torch.no_grad():
		img = model.vae.decode(x_t)['sample']
		x_t_2 = model.vae.encode(img)['latent_dist'].mean.detach().cpu()
	x_t_2 = x_t_2 * 0.18215
	return x_t_2

def img2noise(model, image, percent, generator=None, noise=None, mult=1):
	# Adds some noise to an image and converts it to a latent. mult=10 for correct amount of noise according to scheduler.
	if noise is None:
		if generator is None:
			generator=torch.Generator().manual_seed(10)
		noise = torch.randn((1,4,64,64), generator=generator).cuda()
	latent = ptp_utils.image2latent(model.vae, np.array(image))
	timesteps = torch.LongTensor([max(1,int((100-(percent))*mult))]).cuda()
	noisy_latent = model.scheduler.add_noise(latent, noise, timesteps)
	return noisy_latent

def normalize(x_t_orig, x_t_new):
	# Fix noise statistics to match original image
	orig_std = x_t_orig.std(axis=(2,3), keepdims=True)
	orig_mean = x_t_orig.mean(axis=(2,3), keepdims=True)
	x_t_new = (x_t_new / x_t_new.std(axis=(2,3), keepdims=True)) * orig_std
	x_t_new = (x_t_new - x_t_new.mean(axis=(2,3), keepdims=True)) + orig_mean
	x_t_new = torch.sinh(torch.arcsinh(x_t_new) * 0.99) # kurtosis transform
	return x_t_new

def overlay_noise(x_t_modified, x_t_rolled, mask_rolled):
	# Expects noise mask with white and black sections
	result = np.where(mask_rolled == 255, np.array(x_t_rolled), np.array(x_t_modified))
	return torch.from_numpy(result)

def alpha_blend(images, masks=None):
	# Stack image layers which have transparency
	# If no masks, use image alpha for masks
	if type(images[0]) is not Image.Image:
		images = [Image.fromarray(image) for image in images]
	images = [image.convert('RGBA') for image in images]
	if masks is None:
		masks = images
	background = images[0]
	for i in range(1, len(images)):
		background.paste(images[i], (0, 0), masks[i])
	background = background.convert('RGB')
	return background


##########################################################################################################################
#--------------------------------------------------Transformations-------------------------------------------------------#
##########################################################################################################################

def translate_entire_noise(roll, x_t):
	# Default pan. 1 roll of latent array is 8 pixels in image space
	x_t_modified = x_t.detach().clone().cpu()
	x_t_modified = torch.from_numpy(np.roll(x_t_modified, -roll, axis=3)).cuda()
	return x_t_modified

def translate_entire_control(step, roll, control_image):
	# Default pan. Step is usually 1 or 8
	modified_control_image = np.array(control_image)
	modified_control_image = np.roll(modified_control_image, -roll*step, axis=1)
	modified_control_image = Image.fromarray(modified_control_image)
	return modified_control_image

def affine_parallax_noise(mask, roll, x_t):
	# Custom rolling for parallax effect
	x_t_modified = x_t.detach().clone().cpu()
	for row in range(64):
		dist = int((row+12) * -(2.5*roll) / 32)
		x_t_modified[:, :, row, :] = torch.from_numpy(np.roll(x_t_modified[:, :, row, :], dist, axis=2))
	x_t_modified.cuda()
	if mask is not None:
		x_t_rolled = x_t.detach().clone().cpu()
		x_t_rolled = torch.from_numpy(np.roll(x_t_rolled, -4*roll, axis=3))
		mask_rolled = np.roll(np.array(mask), -4*roll, axis=1)
		x_t_modified = overlay_noise(x_t_modified, x_t_rolled, mask_rolled).cuda()
	return x_t_modified

def affine_parallax_noise_v(roll, x_t):
	# Custom rolling for parallax effect (vista)
	x_t_modified = x_t.detach().clone().cpu()
	for row in range(64):
		dist = -1*roll if row < 28 else -2*roll if row < 35 else -3*roll if row < 44 else -11*roll
		modified_row = np.roll(x_t_modified[:, :, row, :], dist, axis=2)
		if row >= 45:
			modified_row = 0.5*modified_row + 0.5*np.roll(modified_row, 1, axis=2)
		if torch.is_tensor(modified_row):
			x_t_modified[:, :, row, :] = modified_row
		else:
			x_t_modified[:, :, row, :] = torch.from_numpy(modified_row)
	x_t_modified.cuda()
	return x_t_modified

def stereo_rotate_control(roll, control_image):
	# ControlNet slightly lags 1 pixel behind each time to "reveal" more of the moving object
	modified_control_image = np.array(control_image)
	modified_control_image = np.roll(modified_control_image, -4*roll*7, axis=1)
	modified_control_image = Image.fromarray(modified_control_image)
	return modified_control_image

def perspective_control_v(roll, control_image):
	# Custom rolling for parallax effect (vista)
	modified_control_image = np.array(control_image)
	for row in range(512):
		dist = -1*roll if row < 28*8 else -2*roll if row < 35*8 else -3*roll if row < 44*8 else -11*roll
		modified_control_image[row, :, :] = np.roll(modified_control_image[row, :, :], dist*8, axis=0)
	modified_control_image = Image.fromarray(modified_control_image)
	return modified_control_image

def flip_horizontal_noise(x_t):
	x_t_modified = x_t.detach().clone().cpu().numpy()
	x_t_modified2 = x_t_modified.copy()[:, :, :, ::-1]
	x_t_modified2 = torch.from_numpy(x_t_modified2.copy()).cuda()
	return x_t_modified2

def upscale_noise(method, roll, x_t):
	# Resizing using PIL for interpolation
	x_t_modified = x_t.detach().clone().cpu()[0]
	x_t_modified = x_t_modified / 5
	x_t_modified = (x_t_modified + 0.5).clamp(0, 1)
	x_t_modified = x_t_modified.cpu().numpy()
	x_t_modified = (x_t_modified * 255).astype(np.uint8)
	x_t_modified_PIL = [Image.fromarray(x_t_modified[i]).convert('L') for i in range(4)]
	x_t_modified_PIL = [x_t_modified_PIL[i].resize((512, 512), method) for i in range(4)]
	display(Image.fromarray(np.array([x_t_modified_PIL[i] for i in range(3)]).transpose(1, 2, 0)))
	x_t_modified = np.array([np.array(im) for im in x_t_modified_PIL])
	x_t_modified = np.roll(x_t_modified, -roll, axis=2)
	x_t_modified = x_t_modified[:, 4::8, 4::8] # sample every 8th pixel starting from midpoints
	x_t_modified = torch.from_numpy(x_t_modified).unsqueeze(0).float()
	x_t_modified = (x_t_modified / 255 - 0.5).cuda()
	x_t_modified = x_t_modified * 5
	x_t_modified = normalize(x_t, x_t_modified) # increase variance
	return x_t_modified

def grid_interp_noise(method, roll, x_t):
	# Resizing using numpy for interpolation
	x_t_modified = x_t.detach().clone().cpu().numpy()[0]
	interpolators = [interpolate.RegularGridInterpolator((np.arange(0,512,8), np.arange(0,512,8)), 
						x_t_modified[i, :, :], bounds_error=False, fill_value=None, method=method) for i in range(4)]
	X, Y = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
	x_t_modified = np.array([interpolators[i]((X, Y)) for i in range(4)])
	x_t_rolled = np.roll(x_t_modified, -roll, axis=2)
	x_t_modified = x_t_rolled[:, ::8, ::8] #sample every 8th pixel
	x_t_modified = torch.from_numpy(x_t_modified).unsqueeze(0).float().cuda()
	x_t_modified = normalize(x_t, x_t_modified) # increase variance
	return x_t_modified

def vae_interp(model, roll, x_t):
	# VAE upscale for panning only
	view_noise(x_t)
	r = roll % 8
	snap = int(np.floor((roll+3.5)/8))
	if r > 4: r = r - 8
	dist = abs(r)
	x_t_snap = x_t.detach().clone().cpu()
	x_t_snap = torch.from_numpy(np.roll(x_t_snap, -snap, axis=3)).cuda()
	x_t_snap = 1/0.18215 * x_t_snap
	with torch.no_grad():
		decoded_x_t = model.vae.decode(x_t_snap.cuda())['sample']
		decoded_x_t_rolled = np.roll(np.array(decoded_x_t.cpu()), -1*r, axis=3)
		x_t_2 = model.vae.encode(torch.from_numpy(decoded_x_t_rolled).cuda())['latent_dist'].mean.detach()
	x_t_2 = x_t_2 * 0.18215
	x_t_2 = normalize(x_t, x_t_2)
	scale1 = 1*(1-dist/45)
	x_t_2 = x_t_2 * scale1
	view_noise(x_t_2)
	return x_t_2

def VAE_zoom(model, switch_percent, roll, x_t):
	# VAE upscale for zooming only
	view_noise(x_t)
	zoom_size = 512 + 2*roll
	left_top = (zoom_size - 512)//2
	right_bottom = (zoom_size + 512)//2
	x_t_2 = x_t.detach().clone().cpu()
	scale_factor = -2*(switch_percent/100) + 3
	x_t_2 = x_t_2 / scale_factor
	x_t_2 = 1/0.18215 * x_t_2
	with torch.no_grad():
		decoded_x_t = model.vae.decode(x_t_2.cuda())['sample']
		decoded_x_t_PIL = [Image.fromarray(np.array(decoded_x_t.cpu()[0][i])) for i in range(3)]
		decoded_x_t_zoomed = [decoded_x_t_PIL[i].resize((zoom_size, zoom_size), Image.BICUBIC) for i in range(3)]
		decoded_x_t_zoomed = [decoded_x_t_zoomed[i].crop((left_top, left_top, right_bottom, right_bottom)) for i in range(3)]
		decoded_x_t_zoomed = [np.array(decoded_x_t_zoomed[i]) for i in range(3)]
		decoded_x_t_zoomed = np.array(decoded_x_t_zoomed)
		x_t_modified = model.vae.encode(torch.from_numpy(decoded_x_t_zoomed).unsqueeze(0).cuda())['latent_dist'].mean.detach()
	x_t_modified = x_t_modified * 0.18215
	view_image(ptp_utils.latent2image(model.vae, x_t_modified))
	x_t_modified = normalize(x_t, x_t_modified)
	return x_t_modified

def VAE_rotate(model, switch_percent, roll, x_t):
	# VAE upscale for rotating only
	view_noise(x_t)
	x_t_2 = x_t.detach().clone().cpu()
	scale_factor = -2*(switch_percent/100) + 3
	x_t_2 = x_t_2 / scale_factor
	x_t_2 = 1/0.18215 * x_t_2
	with torch.no_grad():
		decoded_x_t = model.vae.decode(x_t_2.cuda())['sample']
		decoded_x_t_PIL = [Image.fromarray(np.array(decoded_x_t.cpu()[0][i])) for i in range(3)]
		decoded_x_t_rotated = [decoded_x_t_PIL[i].rotate(-roll) for i in range(3)]
		decoded_x_t_rotated = [np.array(decoded_x_t_rotated[i]) for i in range(3)]
		decoded_x_t_rotated = np.array(decoded_x_t_rotated)
		x_t_modified = model.vae.encode(torch.from_numpy(decoded_x_t_rotated).unsqueeze(0).cuda())['latent_dist'].mean.detach()
	x_t_modified = x_t_modified * 0.18215
	view_image(ptp_utils.latent2image(model.vae, x_t_modified))
	x_t_modified = normalize(x_t, x_t_modified)
	return x_t_modified


##########################################################################################################################
#--------------------------------------------------------Flow------------------------------------------------------------#
##########################################################################################################################

def interpret_optical_flow_col(step, map):
	# Reads optical flow map into distances (parallellized for speed)
	if step == 0:
		return torch.zeros((512,512,2)).int()
	map = np.uint8(np.array(map))
	hsv = torch.tensor(cv2.cvtColor(map, cv2.COLOR_RGB2HSV)).float().cuda()
	hsv[:,:,0] = hsv[:,:,0] * 2
	distances = torch.zeros((512,512,2)).cuda() #x, y
	red_mask = (315 < hsv[:,:,0]) | (hsv[:,:,0] <= 45)
	cyan_mask = (135 < hsv[:,:,0]) & (hsv[:,:,0] <= 225)
	magenta_mask = (225 < hsv[:,:,0]) & (hsv[:,:,0] <= 315)
	lime_mask = (45 < hsv[:,:,0]) & (hsv[:,:,0] <= 135)
	periodic_mask = hsv[:,:,2] < 255
	distances[:, :, 1] = torch.where(red_mask, 4*step * hsv[:,:,1]/100, distances[:, :, 1])
	distances[:, :, 1] = torch.where(cyan_mask, -4*step * hsv[:,:,1]/100, distances[:, :, 1])
	distances[:, :, 0] = torch.where(magenta_mask, -4*step * hsv[:,:,1]/100, distances[:, :, 0])
	distances[:, :, 0] = torch.where(lime_mask, 4*step * hsv[:,:,1]/100, distances[:, :, 0])
	distances[:, :, 0] = torch.where(periodic_mask, (distances[:, :, 0]/step) * torch.sin(30*step/hsv[:,:,2]), distances[:, :, 0])
	distances[:, :, 1] = torch.where(periodic_mask, (distances[:, :, 1]/step) * torch.sin(30*step/hsv[:,:,2]), distances[:, :, 1])
	return torch.round(distances).int()

def flow_mapping(distances, image, wrap):
	# Applies optical flow to image (parallellized for speed)
	i_idx, j_idx = torch.meshgrid(torch.arange(512).cuda(), torch.arange(512).cuda())
	if wrap:
		new_i = (i_idx + distances[:,:,0]) % 512
		new_j = (j_idx + distances[:,:,1]) % 512
	else:
		new_i = torch.clamp(i_idx + distances[:,:,0], 0, 511)
		new_j = torch.clamp(j_idx + distances[:,:,1], 0, 511)
	image[:, :, new_i, new_j] = image[:, :, i_idx, j_idx]
	return image

def compute_flow(model, map, wrap, switch_percent, roll, x_t, noising=False):
	# Flow method for liquid noise. Can use scaling or noising
	with torch.no_grad():
		distances = interpret_optical_flow_col(roll, map).cuda()
		x_t_2 = x_t.detach().clone()
		scale_factor = -2*(switch_percent/100) + 3
		if not noising: x_t_2 = x_t_2 / scale_factor # scale variance
		x_t_2 = 1/0.18215 * x_t_2
		x_t_decoded = model.vae.decode((x_t_2).cuda())['sample']
		x_t_decoded_new = x_t_decoded.clone()
		x_t_decoded_new = flow_mapping(distances, x_t_decoded_new, wrap)
		if noising: x_t_decoded_new = x_t_decoded_new + 0.1*torch.randn_like(x_t_decoded_new) # noising method
		x_t_modified = model.vae.encode(x_t_decoded_new)['latent_dist'].mean.detach()
		x_t_modified = x_t_modified * 0.18215
		# view_image(ptp_utils.latent2image(model.vae, x_t_modified))
		if noising: x_t_modified = x_t_modified + 0.05*torch.randn_like(x_t_modified) # noising method
		x_t_modified = normalize(x_t, x_t_modified)
		return x_t_modified
	
def combine_images_flow(model, roll, images, flows, switch_percent, wrap, generator, noise_mult=1):
	# Img2Vid with layers (flow)
	assert(len(images)==len(flows)), "Number of images and flows must be equal"
	noise = torch.randn((1,4,64,64), generator=generator).cuda()
	noise_decoded = torch.from_numpy(ptp_utils.latent2image(model.vae, noise)).cuda().permute(0, 3, 1, 2)
	image_layers = []
	noise_layers = []
	for i in range(len(images)): # Process image and noise stacks separated first
		distances = interpret_optical_flow_col(roll, flows[i]).cuda()
		image = torch.from_numpy(np.array(images[i])).permute(2, 0, 1).unsqueeze(0).cuda()
		modified_image = flow_mapping(distances, image.clone(), wrap)
		image_layers.append(Image.fromarray(modified_image[0].cpu().numpy().transpose(1, 2, 0)))
		modified_noise = flow_mapping(distances, noise_decoded.clone(), wrap)
		noise_layers.append(Image.fromarray(modified_noise[0].cpu().numpy().transpose(1, 2, 0)))
	
	stacked_img = alpha_blend(image_layers)
	stacked_noise = alpha_blend(noise_layers, masks=image_layers)
	noise_encoded = ptp_utils.image2latent(model.vae, np.array(stacked_noise))
	final_latent = img2noise(stacked_img, switch_percent, noise=noise_encoded, mult=noise_mult) # Diffuse final stacks
	return final_latent.cuda()


##########################################################################################################################
#------------------------------------------------------Upscaling---------------------------------------------------------#
##########################################################################################################################

def upscale_noise_tracking(model, image, coords, tiles, mult=2, seed=SEED, seamless=True, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE):
	# Seamless upscale meathod using noise tracking
	print(coords)
	denoise_amt = 0.05
	switch_percent = (1-denoise_amt)*100
	generator = torch.Generator().manual_seed(seed)
	coords_latent = (np.ceil(coords[0]*mult / 8).astype(int), np.ceil(coords[1]*mult / 8).astype(int))
	noise = torch.randn((1,4,64,64), generator=generator).cuda()
	if seamless:
		noise = noise.repeat(1,1,tiles[0],tiles[1]) # Tiled noise plane to sample from
		noise_crop = noise[:, :, coords_latent[0]:coords_latent[0]+64, coords_latent[1]:coords_latent[1]+64]
	else:
		noise_crop = noise
	small_crop = np.ceil(512/mult).astype(int)
	image_crop = image.crop((coords[0], coords[1], coords[0]+small_crop, coords[1]+small_crop))
	image_resized = image_crop.resize((512, 512), Image.BILINEAR)
	image_resized = Image.blend(image_resized, image_resized.filter(ImageFilter.EDGE_ENHANCE), 0.5)
	# display(image_resized)
	noisy_latent = img2noise(model, image_resized, switch_percent, noise=noise_crop)
	denoised_image, _, _ = ptp_utils.text2image_ldm_stable(model, [""], latent=noisy_latent, precalced=True, 
														switch_percent=switch_percent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
	clear_output()
	# view_image(denoised_image)
	return denoised_image[0]

def stitch(images, width, height):
	# Make large image from smaller sections
	sq_size = max(width, height)
	stitch = Image.new('RGB', (sq_size, sq_size))
	small_width, small_height, _ = images[0][0].shape
	for i in range(len(images)):
		for j in range(len(images[0])):
			image = Image.fromarray(np.array(images[i][j]))
			stitch.paste(image, (small_width*i, small_height*j))
	stitch = stitch.crop((0, 0, width, height))
	return stitch

def correct_size(stitched, mult, tiles):
	# Handle rounding errors for non-typical upscale factors (e.g. 3×)
	width, height = stitched.size
	width_error = tiles[0] * (np.ceil(512/mult) - 512/mult)
	new_width = int(width - width_error) - 1
	height_error = tiles[1] * (np.ceil(512/mult) - 512/mult)
	new_height = int(height - height_error) - 1
	corrected = stitched.crop((0, 0, new_width, new_height))
	corrected = corrected.resize((width, height), Image.BICUBIC)
	return corrected

def upscale_and_stitch(model, image, mult=2, seed=SEED, seamless=True):
	width, height = image.size
	jump = np.ceil(512/mult).astype(int)
	tiles = (np.ceil(mult*width/512).astype(int), np.ceil(mult*height/512).astype(int))
	images = [[upscale_noise_tracking(model, image, coords=(jump*i,jump*j), 
					tiles=tiles, mult=mult, seed=seed, seamless=seamless) for j in range(tiles[1])] for i in range(tiles[0])]
	stitched = stitch(images, width*mult, height*mult)
	if mult % 2 != 0:
		stitched = correct_size(stitched, mult, tiles)
	return stitched

def magnifying_glass(model, image, coords, mult=2, seed=SEED, seamless=True):
	# Arbitrary upscaling using noise tracking
	width, height = image.size
	tiles = (np.ceil(mult*width/512).astype(int), np.ceil(mult*height/512).astype(int))
	result = upscale_noise_tracking(model, image, coords, tiles=tiles, mult=mult, seed=seed, seamless=seamless)
	result = Image.fromarray(np.array(result))
	return result

##########################################################################################################################
#------------------------------------------------------Relighting--------------------------------------------------------#
##########################################################################################################################

def combine_latents(model, seed_bg, seed_fg, prompt, mask, control, percent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE):
	# Paste one noise onto another using a mask
	if type(prompt) is str:
		prompt = [prompt]
	x_t_1 = create_latent(model, seed=seed_bg)
	x_t_2 = create_latent(model, seed=seed_fg)
	_, precalced_1, _ = ptp_utils.text2image_ldm_stable(model, prompt, controller=EmptyControl(), latent=x_t_1, switch_percent=percent, control=control, 
													 precalced=False, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=None)
	_, precalced_2, _ = ptp_utils.text2image_ldm_stable(model, prompt, controller=EmptyControl(), latent=x_t_2, switch_percent=percent, control=control, 
													 precalced=False, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=None)
	x_t = overlay_noise(precalced_1.cpu().numpy(), precalced_2.cpu().numpy(), np.array(mask)).cuda()
	image, _, _ = ptp_utils.text2image_ldm_stable(model, prompt, controller=EmptyControl(), latent=x_t, switch_percent=percent, control=control, 
											   precalced=True, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=None)
	return image

##########################################################################################################################
#-------------------------------------------------Video Style Transfer---------------------------------------------------#
##########################################################################################################################

def convert_to_img_seq(path, change_fps=False):
	if path[-4:] == '.gif':
		gif = imageio.mimread(path)
		images = [Image.fromarray(frame).convert('RGB') for frame in gif]
	elif path[-4:] == '.mp4':
		video = cv2.VideoCapture(path)
		images = []
		count = 0
		while True:
			ret, frame = video.read()
			if not ret:
				break
			count += 1
			if change_fps and count % 2 == 0: #reduce frame rate
				continue
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = Image.fromarray(frame).convert('RGB')
			images.append(frame)
	images_cropped = []
	for frame in images:
		width, height = frame.size
		sq_size = min(width, height) #crop to square
		left = (width - sq_size) // 2
		top = (height - sq_size) // 2
		right = (width + sq_size) // 2
		bottom = (height + sq_size) // 2
		frame = frame.crop((left, top, right, bottom))
		frame = frame.resize((512, 512), Image.BICUBIC)
		images_cropped.append(frame)
	return images_cropped

def extract_optical_flow(images):
	# CV2 optical flow instead of custom. Incompatible with flow maps
	flows = []
	for i in range(1, len(images)):
		prev = cv2.cvtColor(np.array(images[i-1]), cv2.COLOR_RGB2GRAY)
		curr = cv2.cvtColor(np.array(images[i]), cv2.COLOR_RGB2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flows.append(flow)
	return flows

def apply_optical_flow(image, flow):
	# CV2 optical flow instead of custom. Incompatible with flow maps
	w, h = image.size
	flow = cv2.resize(flow, (w, h))
	flow = -flow
	flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w))).reshape(h, w, 2)
	flow_map = flow_map + flow
	map_x = flow_map[..., 0].astype(np.float32)
	map_y = flow_map[..., 1].astype(np.float32)
	image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	new_image_cv2 = cv2.remap(image_cv2, map_x, map_y, cv2.INTER_LINEAR)
	new_image = Image.fromarray(cv2.cvtColor(new_image_cv2, cv2.COLOR_BGR2RGB))
	return new_image

def video_style_transfer(model, prompt, path, percent, seed=SEED, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE):
	# Style transfer using noise tracking to reduce distortion
	if type(prompt) is str:
		prompt = [prompt]
	images = convert_to_img_seq(path, change_fps=True)
	flows = extract_optical_flow(images)
	generator = torch.Generator().manual_seed(seed)
	noise = torch.randn((1,4,64,64), generator=generator).cuda()
	curr_noise = Image.fromarray(ptp_utils.latent2image(model.vae, noise)[0])
	final_frames = []
	for i in range(1, len(images)):
		curr_noise = apply_optical_flow(curr_noise, flows[i-1])
		# final_frames.append(np.array(curr_noise))
		encoded_noise = ptp_utils.image2latent(model.vae, np.array(curr_noise))
		latent = img2noise(model, images[i], percent, noise=encoded_noise, mult=10)
		denoised_image, _, _ = ptp_utils.text2image_ldm_stable(model, prompt, latent=latent, precalced=True, switch_percent=percent, 
														 num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
		clear_output(wait=True)
		view_image(denoised_image)
		view_noise(latent)
		print("frame:", i, "of", len(images))
		final_frames.append(denoised_image[0])
	
	gif_PIL = [Image.fromarray(np.uint8(img)).resize((512, 512), Image.NEAREST) for img in final_frames]
	id = np.random.randint(0,10000)
	duration = 80
	gif_PIL[0].save(f'output/transfer{id}.gif', format='GIF', append_images=gif_PIL[1:], save_all=True, duration=duration, loop=0)
	print(f"saved as output/transfer{id}.gif")
		

##########################################################################################################################
#------------------------------------------------------Main--------------------------------------------------------------#
##########################################################################################################################

def create_animation(model, prompt, control=None, flows=None, mask=None, images=None, start=0, stop=16, switch_percent=70, 
					wrap=True, noise_mult=1, controller=EmptyControl(), noise_anim="flow", control_anim="None", 
					num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, cond_scale=COND_SCALE, 
					flip=False, fps=12.5, seed=SEED, save=True):
	
	# Call this function to create a video. Don't pass in images for prompt2vid. 
	# Pass one image and flow map for img2vid. Pass a list of images and flows for layered images.
	# Most of the time, use "flow" for noise_anim (liquid noise)
	# noise_anim: pan, parallax, parallaxV, upscale, grid_interp, vae_interp, zoom, rotate, flow, noising
	# control_anim: pan_small, pan, perspective, perspectiveV

	torch.cuda.empty_cache()
	model.cond_scale = cond_scale
	generator=torch.Generator().manual_seed(seed)
	if type(prompt) is str:
		prompt = [prompt]
	gif_images = []
	step = 1

	if images is None: #prompt-to-video
		x_t = create_latent(model, seed=seed)
		precalced_latent = None
	elif type(images) is not list: #single image-to-video
		x_t = None
		precalced_latent = img2noise(model, images, switch_percent, generator=generator, mult=noise_mult)
	else:	#layered images-to-video
		x_t = None
		precalced_latent = img2noise(model, alpha_blend(images), switch_percent, generator=generator, mult=noise_mult)
	# display(control)
		
	if flip:
		x_t = flip_horizontal_noise(x_t)
		if control is not None: control = control.transpose(Image.FLIP_LEFT_RIGHT)
		if mask is not None: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
		start = -start
		stop = -stop
		step = -1
	
	if flows is None:
		if noise_anim == "pan":
				noise_func = translate_entire_noise
		elif noise_anim == "parallax":
			noise_func = partial(affine_parallax_noise, mask)
		elif noise_anim == "parallaxV":
			noise_func = affine_parallax_noise_v
		elif noise_anim == "upscale":
			noise_func = partial(upscale_noise, Image.LANCZOS)
		elif noise_anim == "grid_interp":
			noise_func = partial(grid_interp_noise, "linear")
		elif noise_anim == "vae_interp":
			noise_func = partial(vae_interp, model)
		elif noise_anim == "zoom":
			noise_func = partial(VAE_zoom, model, switch_percent)
		elif noise_anim == "rotate":
			noise_func = partial(VAE_rotate, model, switch_percent)
		else:
			noise_func = None
		if control_anim == "pan_small":
			control_func = partial(translate_entire_control, 1)
		elif control_anim == "pan":
			control_func = partial(translate_entire_control, 8)
		elif control_anim == "perspective":
			control_func = stereo_rotate_control
		elif control_anim == "perspectiveV":
			control_func = perspective_control_v
		else:
			control_func = None
	else:
		if noise_anim == "noising":
			noise_func = partial(compute_flow, model, flows, wrap, switch_percent, noising=True)
		else:
			noise_func = partial(compute_flow, model, flows, wrap, switch_percent, noising=False)
		control_func = None

	for roll in range(start, stop, step):
		if precalced_latent is None:
			image, precalced_latent, final_latent = ptp_utils.text2image_ldm_stable(model=model, prompt=prompt, controller=controller, latent=x_t, precalced=False, control=control, 
															switch_percent=switch_percent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
		else:
			if type(images) is not list:
				if control is not None and control_func is not None:
					curr_control = control_func(roll, control)
				else:
					curr_control = None
				if noise_func is not None:
					modified_latent = noise_func(roll, precalced_latent).cuda()
			else: # layer images
				modified_latent = combine_images_flow(model, roll, images, flows, switch_percent, wrap, generator, noise_mult)
				curr_control = None
			image, _, final_latent = ptp_utils.text2image_ldm_stable(model=model, prompt=prompt, controller=controller, latent=modified_latent, precalced=True, control=curr_control, 
															switch_percent=switch_percent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
		clear_output(wait=True)
		view_image(image)
		if control_func is not None:
			display(control_func(roll, control))
		# view_noise(noise_func(roll, x_t))
		view_noise(final_latent)
		print("frame: ", roll)
		gif_images.append(image[0])
		torch.cuda.empty_cache()

	if save:
		gif_PIL = [Image.fromarray(np.uint8(img)).resize((512, 512), Image.NEAREST) for img in gif_images]
		id = np.random.randint(0,10000)
		duration = int(1000/fps)
		gif_PIL[0].save(f'output/roll{id}.gif', format='GIF', append_images=gif_PIL[1:], save_all=True, duration=duration, loop=0)
		print(f"saved as output/roll{id}.gif")
	else:
		return gif_images

##########################################################################################################################
#------------------------------------------------------Eval--------------------------------------------------------------#
##########################################################################################################################

def sobel_edge_detection(image, kernel, blur):
    if image.ndim == 3: # greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (blur, blur), 0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    magnitudes = np.sqrt(sobelx**2 + sobely**2)
    orientations = np.arctan2(sobely, sobelx)
    return magnitudes, orientations

def weighted_orientations(magnitudes, orientations):
    magnitudes = np.where(abs(orientations)<np.pi/2, magnitudes, 0)
    weighted_orientations = magnitudes * np.exp(1j * orientations)
    average_directions = np.angle(np.sum(weighted_orientations, axis=(1)))

    # plt.imshow(np.abs(weighted_orientations))
    # plt.colorbar()
    # plt.show()

    # plt.imshow(np.angle(weighted_orientations), vmin=-np.pi, vmax=np.pi, cmap='hsv')
    # plt.colorbar()
    # plt.show()
    # plt.plot(average_directions)
    # plt.title("Average Directions")
    # plt.xlabel("frame")
    # plt.ylabel("angle (radians)")
    # plt.ylim(-np.pi/2, np.pi/2)
    # plt.show()
    return average_directions

def analyze_smoothness(average_directions):
    second_derivative = np.diff(average_directions, n=2)
    smoothness = np.std(second_derivative)
    smoothness = np.exp(-smoothness)
    
    # plt.plot(second_derivative)
    # plt.title("Second Derivative")
    # plt.xlabel("frame")
    # plt.ylim(-3, 3)
    # plt.show()
    return smoothness

def temporal_smoothness_metric(x_t_slices, kernel=3, blur=3):
    if type(x_t_slices) is not np.ndarray:
        x_t_slices = np.array(x_t_slices)
    magnitudes, orientations = sobel_edge_detection(x_t_slices, kernel=kernel, blur=blur)
    average_directions = weighted_orientations(magnitudes, orientations)
    smoothness = analyze_smoothness(average_directions)
    return smoothness

def X_T_slices(path, line=350, direction="horizontal", thickness=1):
	images = convert_to_img_seq(path)
	images = images[:16]
	total_width = len(images) * thickness
	X_T_array = np.zeros((512, total_width, 3))
	for i, frame in enumerate(images):
		frame_image = np.array(frame)
		slice = frame_image[line] if direction == "horizontal" else frame_image[:, line]
		slice = np.repeat(slice[:, np.newaxis, :], thickness, axis=1)
		X_T_array[:, i*thickness:(i+1)*thickness, :] = slice
	X_T_img = Image.fromarray(X_T_array.transpose(1, 0, 2).astype(np.uint8))
	return X_T_img

def MSE_over_frames(gif_images, speed=8, flow=None):
	gif_images = [np.array(image).astype(int) for image in gif_images]
	original = gif_images[0]
	MSEs = []
	for i in range(1, len(gif_images)):
		if flow:
			distances = interpret_optical_flow_col(i, flow)
			original_rolled = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).cuda()
			original_rolled = flow_mapping(distances, original_rolled.clone(), wrap=True)[0].cpu().numpy().transpose(1, 2, 0)
		else:
			original_rolled = np.roll(original, -i*speed, axis=1)
		SE = (original_rolled - gif_images[i])**2
		if i%2:
			display(Image.fromarray(SE.astype(np.uint8)))
		MSE = np.mean(SE)
		MSEs.append(MSE)
	average_MSE = np.mean(MSEs)
	return average_MSE

def MSE_upscale(model, path, x_coords, y_coords, mult=2, seamless=True):
	image = Image.open(path).convert('RGB')
	stitched = upscale_and_stitch(model, image, mult=mult, seamless=seamless)
	view_image(stitched)
	MSEs = []
	for x in x_coords:
		for y in y_coords:
			magnified = magnifying_glass(model, image, coords=(x, y), mult=mult, seamless=seamless)
			magnified = np.array(magnified).astype(int)
			cropped = stitched.crop((x*mult, y*mult, (x*mult)+512, (y*mult)+512))
			cropped = np.array(cropped).astype(int)
			SE = (cropped - magnified)**2
			display(Image.fromarray(SE.astype(np.uint8)))
			MSE = np.mean(SE)
			MSEs.append(MSE)
	average_MSE = np.mean(MSEs)
	return average_MSE