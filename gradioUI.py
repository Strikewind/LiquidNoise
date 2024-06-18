import gradio as gr
from liquidnoise import *
import warnings
warnings.filterwarnings('ignore')

model = Model()
ldm_stable = model.ldm_stable


white_image = Image.new("RGB", (512, 512), (255, 255, 255))
curr_upscale = Image.new("RGB", (512, 512), (255, 255, 255))
brush1 = gr.Brush(colors=["#04FA07", "#06E6E6", "#0907E6", "#787846"], default_color="#04FA07", default_size=30)
brush2 = gr.Brush(colors=["#99FFF6", "#D1A3FF", "#FF908E", "#BEFF89"], default_color="#99FFF6", default_size=20)

def main1(prompt, duration, switch_percent, seed, diffusion_steps, cfg, fps, noise_anim, control_anim, control, cond_scale, progress=gr.Progress()):
	if not np.all(np.array(control["composite"]) == 255):
		control_image = control["composite"].convert("RGB")
	else:
		control_image = None

	gif_path = create_animation(ldm_stable, prompt, stop=duration, switch_percent=switch_percent, fps=fps, seed=seed, control=control_image,
						num_inference_steps=diffusion_steps, guidance_scale=cfg, cond_scale=cond_scale, save=True,
						noise_anim=noise_anim, control_anim=control_anim, progress_callback=progress)
	return gif_path

def main2(prompt, duration, switch_percent, seed, diffusion_steps, cfg, fps, noise_anim, control_anim, control, cond_scale, flow, wrap, progress=gr.Progress()):
	if not np.all(np.array(control["composite"]) == 255):
		control_image = control["composite"].convert("RGB")
	else:
		control_image = None

	gif_path = create_animation(ldm_stable, prompt, stop=duration, switch_percent=switch_percent, fps=fps, seed=seed, control=control_image,
						flows=flow["composite"].convert("RGB"), num_inference_steps=diffusion_steps, guidance_scale=cfg, cond_scale=cond_scale, save=True,
						noise_anim=noise_anim, control_anim=control_anim, wrap=wrap, progress_callback=progress)
	return gif_path


def main3(I1, I2, I3, F1, F2, F3, prompt, duration, switch_percent, seed, diffusion_steps, cfg, fps, noise_anim, wrap, progress=gr.Progress()):
	
	flow = [F1, F2, F3]
	flow_images = [flow[i]["composite"].convert("RGB") for i in range(len(flow))]

	images = [I1, I2, I3]
	for i in reversed(range(len(images))):
		if images[i] is None:
			images.pop(i)
			flow_images.pop(i)
			
	if len(images) == 1:
		images = images[0]
		flow_images = flow_images[0]

	gif_path = create_animation(ldm_stable, prompt, stop=duration, switch_percent=switch_percent, fps=fps, seed=seed,
						flows=flow_images, images=images, num_inference_steps=diffusion_steps, guidance_scale=cfg, save=True,
						noise_anim=noise_anim, wrap=wrap, progress_callback=progress)
	return gif_path

def vid2vid(input_video, prompt, tracking, switch_percent, seed, diffusion_steps, cfg, progress=gr.Progress()):
	video_path = video_style_transfer(ldm_stable, prompt, tracking=tracking, path=input_video, percent=switch_percent, seed=seed, 
								   num_inference_steps=diffusion_steps, guidance_scale=cfg, progress_callback=progress)
	return video_path

def upscale_region(input_img, seamless, seed, upscale_factor, evt: gr.SelectData):
	global curr_upscale
	orig_width, orig_height = input_img.size
	new_width, new_height = orig_width * upscale_factor, orig_height * upscale_factor
	if curr_upscale.size != (new_width, new_height):
		curr_upscale = Image.new("RGB", (new_width, new_height), (0, 0, 0))
	halfway = int(256/upscale_factor)
	coords = (max(0, evt.index[0]-halfway), max(0, evt.index[1]-halfway))
	coords = (min(coords[0], orig_width-halfway), min(coords[1], orig_height-halfway))
	print(coords)####

	img = magnifying_glass(ldm_stable, input_img, coords, upscale_factor, seed, seamless)
	curr_upscale.paste(img, (coords[0] * upscale_factor, coords[1] * upscale_factor))
	return curr_upscale


with gr.Blocks(theme=gr.themes.Default(), title="Liquid Noise") as demo:
	inputs1 = []
	with gr.Tab("Noise Crystallization"):
		gr.Markdown("# Noise Crystallization")
		gr.Markdown("## Simple consistent animation with large movements. Segmentation map optional.")
		with gr.Row():
			with gr.Column():
					inputs1.append(gr.Textbox(label="Prompt", value="A beautiful landscape"))
					with gr.Row():
						inputs1.append(gr.Slider(minimum=1, maximum=128, value=8, label="Duration"))
						inputs1.append(gr.Slider(minimum=0, maximum=100, value=70, label="Switch Percent"))
						inputs1.append(gr.Slider(minimum=0, maximum=100, value=1, label="Seed"))
					with gr.Row():
						inputs1.append(gr.Slider(minimum=15, maximum=30, value=20, label="Diffusion Steps"))
						inputs1.append(gr.Slider(minimum=5, maximum=12, value=8, label="CFG Scale"))
						inputs1.append(gr.Slider(minimum=5, maximum=30, value=12.5, label="Frames per Second"))
					with gr.Row():
						inputs1.append(gr.Dropdown(label="Noise Animation", choices=["pan", "parallax", "parallaxV"], value="pan")),
						inputs1.append(gr.Dropdown(label="Control Animation", choices=["None", "pan", "perspective", "perspectiveV"], value="None")),
					gr.Markdown("### Choose object colours from [ADE20K dataset](https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8), or upload image.")
					inputs1.append(gr.ImageEditor(label="Segmentation Map", type="pil", value=white_image, canvas_size=[512,512], brush=brush1, image_mode="RGB"))
					inputs1.append(gr.Slider(minimum=0.0, maximum=2.0, value=1.0, label="Conditioning Strength"))
			with gr.Column():
				output1 = gr.Image(label="Output")
				gr.Markdown("### All outputs are saved in the /output folder.")
				button1 = gr.Button("Generate")

		button1.click(main1, inputs1, output1)

	inputs2 = []
	with gr.Tab("Liquid Noise"):
		gr.Markdown("# Liquid Noise")
		gr.Markdown("## Fluid animation with small movements driven by flow map. Segmentation map optional.")
		with gr.Row():
			with gr.Column():
					inputs2.append(gr.Textbox(label="Prompt", value="A beautiful landscape"))
					with gr.Row():
						inputs2.append(gr.Slider(minimum=1, maximum=128, value=8, label="Duration"))
						inputs2.append(gr.Slider(minimum=0, maximum=100, value=70, label="Switch Percent"))
						inputs2.append(gr.Slider(minimum=0, maximum=100, value=1, label="Seed"))
					with gr.Row():
						inputs2.append(gr.Slider(minimum=15, maximum=30, value=20, label="Diffusion Steps"))
						inputs2.append(gr.Slider(minimum=5, maximum=12, value=8, label="CFG Scale"))
						inputs2.append(gr.Slider(minimum=5, maximum=30, value=12.5, label="Frames per Second"))
					with gr.Row():
						inputs2.append(gr.Dropdown(label="Noise Animation", choices=["flow", "noising"], value="flow"))
						inputs2.append(gr.Dropdown(label="Control Animation", choices=["None", "flow", "pan_small"], value="None"))
					with gr.Row():
						with gr.Column():
							gr.Markdown("### Choose object colours from [ADE20K dataset](https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?gid=0#gid=0), or upload image.")
							brush = gr.Brush(colors=["#FFFFFF", "#99FFF6", "#D1A3FF", "#FF908E", "#BEFF89"], default_color="#99FFF6")
							inputs2.append(gr.ImageEditor(label="Segmentation Map", type="pil", value=white_image, canvas_size=[512,512], brush=brush1, image_mode="RGB"))
							inputs2.append(gr.Slider(minimum=0.0, maximum=2.0, value=1.0, label="Conditioning Strength"))
						with gr.Column():
							gr.Markdown("### Flow map is mandatory for this method (draw or upload).")
							inputs2.append(gr.ImageEditor(label="Flow Map", type="pil", value=white_image, canvas_size=[512,512], brush=brush2, image_mode="RGB"))
							inputs2.append(gr.Checkbox(label="Wrap edges", value=True))
							gr.Image(label="Colour guide", value=Image.open("control/flow_guide.png"))
			with gr.Column():
				output2 = gr.Image(label="Output")
				gr.Markdown("### All outputs are saved in the /output folder.")
				button2 = gr.Button("Generate")

		button2.click(main2, inputs2, output2)

	inputs3 = []
	with gr.Tab("Img2Vid"):
		gr.Markdown("# Image-to-Video")
		gr.Markdown("## Liquid noise animation applied to single image or multiple images.")
		with gr.Row():
			with gr.Column():
				with gr.Row():
					with gr.Column():
						with gr.Tab("L1"):
							inputs3.append(gr.Image(label="Input Image 1 (Mandatory)", type="pil", image_mode="RGBA"))
						with gr.Tab("L2"):
							inputs3.append(gr.Image(label="Input Image 2 (Optional)", type="pil", image_mode="RGBA"))
						with gr.Tab("L3"):
							inputs3.append(gr.Image(label="Input Image 3 (Optional)", type="pil", image_mode="RGBA"))
					with gr.Column():
						with gr.Tab("F1"):
							inputs3.append(gr.ImageEditor(label="Flow Map 1 (Mandatory)", type="pil", value=white_image, canvas_size=[512,512], brush=brush2, image_mode="RGB"))
						with gr.Tab("F2"):
							inputs3.append(gr.ImageEditor(label="Flow Map 2 (Optional)", type="pil", value=white_image, canvas_size=[512,512], brush=brush2, image_mode="RGB"))
						with gr.Tab("F3"):
							inputs3.append(gr.ImageEditor(label="Flow Map 3 (Optional)", type="pil", value=white_image, canvas_size=[512,512], brush=brush2, image_mode="RGB"))
				inputs3.append(gr.Textbox(label="Prompt", value="A beautiful landscape"))
				with gr.Row():
					with gr.Column():
						inputs3.append(gr.Slider(minimum=1, maximum=128, value=8, label="Duration"))
						inputs3.append(gr.Slider(minimum=0, maximum=100, value=70, label="Switch Percent"))
						inputs3.append(gr.Slider(minimum=0, maximum=100, value=1, label="Seed"))
						inputs3.append(gr.Slider(minimum=15, maximum=30, value=20, label="Diffusion Steps"))
						inputs3.append(gr.Slider(minimum=5, maximum=12, value=8, label="CFG Scale"))
						inputs3.append(gr.Slider(minimum=5, maximum=30, value=12.5, label="Frames per Second"))
						inputs3.append(gr.Dropdown(label="Noise Animation", choices=["flow", "noising"], value="flow"))
					with gr.Column():
						gr.Image(label="Colour guide", value=Image.open("control/flow_guide.png"))
						inputs3.append(gr.Checkbox(label="Wrap edges", value=True))
			with gr.Column():
				output3 = gr.Image(label="Output")
				gr.Markdown("### All outputs are saved in the /output folder.")
				button3 = gr.Button("Generate")
		
		button3.click(main3, inputs3, output3)

	inputs4 = []
	with gr.Tab("Vid2Vid"):
		gr.Markdown("# Video-to-Video Style Transfer with Noise Tracking")
		gr.Markdown("## Improved vid2vid using liquid noise to reduce distortions. Record or upload video.")
		with gr.Row():
			with gr.Column():
					inputs4.append(gr.Video(label="Input Video", format="mp4"))
					inputs4.append(gr.Textbox(label="Prompt", value="Woman walking in Tokyo, pixar"))
					inputs4.append(gr.Checkbox(label="Noise tracking", value=True))
					with gr.Row():
						inputs4.append(gr.Slider(minimum=0, maximum=100, value=70, label="Switch Percent"))
						inputs4.append(gr.Slider(minimum=0, maximum=100, value=1, label="Seed"))
					with gr.Row():
						inputs4.append(gr.Slider(minimum=15, maximum=30, value=20, label="Diffusion Steps"))
						inputs4.append(gr.Slider(minimum=5, maximum=12, value=8, label="CFG Scale"))
			with gr.Column():
				output4 = gr.Image(label="Output")
				gr.Markdown("### All outputs are saved in the /output folder.")
				button4 = gr.Button("Generate")

		button4.click(vid2vid, inputs4, output4)

	inputs5 = []
	with gr.Tab("Seamless Upscale"):
		gr.Markdown("# Seamless Upscaling with Noise Tracking")
		gr.Markdown("## Improved upscaling of arbitrary regions (without overlap discrepancies) via noise tracking.")
		with gr.Row():
			with gr.Column():
					input_img = gr.Image(label="Input Image", type="pil", image_mode="RGB")
					inputs5.append(input_img)
					gr.Markdown("### Click image to upscale region.")
					inputs5.append(gr.Checkbox(label="Noise Tracking", value=True))
					inputs5.append(gr.Slider(minimum=0, maximum=100, value=1, label="Seed"))
					inputs5.append(gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Upscale Factor"))
			with gr.Column(min_width=1024):
				output5 = gr.Image(label="Output")

		input_img.select(upscale_region, inputs5, output5)

demo.launch(share=True)