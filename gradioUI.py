import gradio as gr
from liquidnoise import *

model = Model()
ldm_stable = model.ldm_stable

def noise_crys(prompt, duration, switch_percent, seed, diffusion_steps, cfg, fps, noise_anim, control_anim, control, cond_scale):
	if not control["composite"]:
		control_image = None
	else:
		control_image = control["composite"]
	gif_path = create_animation(ldm_stable, prompt, stop=duration, switch_percent=switch_percent, fps=fps, seed=seed, control=control_image,
						num_inference_steps=diffusion_steps, guidance_scale=cfg, cond_scale=cond_scale, save=True,
						noise_anim=noise_anim, control_anim=control_anim)
	return gif_path


with gr.Blocks(theme=gr.themes.Default(), title="Liquid Noise") as demo:
	inputs = []
	with gr.Tab("Noise Crystallization"):
		gr.Markdown("# Noise Crystallization")
		with gr.Row():
			with gr.Column():
					inputs.append(gr.Textbox(label="Prompt")),
					with gr.Row():
						inputs.append(gr.Slider(minimum=1, maximum=128, value=8, label="Duration")),
						inputs.append(gr.Slider(minimum=0, maximum=100, value=70, label="Switch Percent")),
						inputs.append(gr.Slider(minimum=0, maximum=100, value=1, label="Seed")),
					with gr.Row():
						inputs.append(gr.Slider(minimum=15, maximum=30, value=20, label="Diffusion Steps")),
						inputs.append(gr.Slider(minimum=5, maximum=12, value=8, label="CFG Scale")),
						inputs.append(gr.Slider(minimum=5, maximum=30, value=12.5, label="Frames per Second")),
					with gr.Row():
						inputs.append(gr.Dropdown(label="Noise Animation", choices=["pan", "parallax", "parallaxV"], value="pan")),
						inputs.append(gr.Dropdown(label="Control Animation", choices=["None", "pan", "perspective", "perspectiveV"], value="None")),
					gr.Markdown("## Choose object colours from [ADE20K dataset](https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?gid=0#gid=0)")
					inputs.append(gr.ImageEditor(label="Draw Segmentation Map", type="pil", height=512, width=512, canvas_size=[512,512], image_mode="RGB"))
					inputs.append(gr.Slider(minimum=0.0, maximum=2.0, value=1.0, label="Conditioning Strength"))
			with gr.Column():
				output = gr.Image(label="Output")
				gr.Markdown("### All outputs are saved in the /output folder.")
				button = gr.Button("Generate")

		button.click(noise_crys, inputs, output)

demo.launch(share=True)