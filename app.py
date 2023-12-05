import streamlit as st
import os
import torch
from IPython.display import display

os.chdir(r'C:\Users\VJ\Desktop\3D-Generation\text_to_3D\shap-e')

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

def generate_3d(prompt_, size_, render_mode_ ='nerf'): # you can change render_mode_ to 'stf'or 'nerf'

    batch_size = 1
    guidance_scale = 15.0
    prompt = prompt_

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0)
    
    render_mode = render_mode_
    size = size_ # size = size of the renders --> higher values take longer to render
    cameras = create_pan_cameras(size, device)

    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        display(gif_widget(images))

    return latents, cameras

# def display(render_mode ='nerf'):  # you can change render_mode_ to 'stf'or 'nerf'
#     for i, latent in enumerate(latents):
#         images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
#         display(gif_widget(images))

def save_model():
    # Saving the latents as meshes.
    for i, latent in enumerate(latents):
      with open(f'{prompt}_{i}.ply', 'wb') as f:
        decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)



st.title('Text to 3D')
prompt = st.text_input('Enter text prompt')

# batch_size = 4 # batch_size = size of the renders --> higher values take longer to render
# guidance_scale = 15.0

# latents = sample_latents(
#     batch_size=batch_size,
#     model=model,
#     diffusion=diffusion,
#     guidance_scale=guidance_scale,
#     model_kwargs=dict(texts=[prompt] * batch_size),
#     progress=True,
#     clip_denoised=True,
#     use_fp16=True,
#     use_karras=True,
#     karras_steps=64,
#     sigma_min=1e-3,
#     sigma_max=160,
#     s_churn=0,
# )

# cameras = create_pan_cameras(guidance_scale, device)
if st.button('Generate'):
    latents, cameras = generate_3d(prompt, 100, 'nerf')

st.button('Download (.ply)',on_click=save_model)

