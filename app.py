import streamlit as st
import os
import torch
from IPython.display import display
from PIL import Image
from io import BytesIO
from pythreejs import *
from IPython.display import display
from traitlets import link

# os.chdir(r'C:\Users\VJ\Desktop\3D-Generation\text_to_3D\shap-e')
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Assuming the necessary imports and definitions are available
# for sample_latents, create_pan_cameras, decode_latent_images, gif_widget, and decode_latent_mesh
def load_models_and_config():
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    return xm, model, diffusion

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

    # # Saving the latents as meshes.
    # for i, latent in enumerate(latents):
    #   with open(f'{prompt}_{i}.ply', 'wb') as f:
    #     decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)
    for i, latent in enumerate(latents):
        with open(f'{prompt}_{i}.ply', 'wb') as f:
            mesh = decode_latent_mesh(xm, latent).tri_mesh()
            mesh.write_ply(f)
            
            # Create a 3D plot
            f = Figure()
            plot_mesh = SurfaceGeometry(vertices=mesh.vertices, faces=mesh.faces)
            mesh = Mesh(geometry=plot_mesh, material=MeshLambertMaterial(color='red'))
            f.scence.add(mesh)
            f.write_html(f'{prompt}_{i}.html')


os.chdir(r'c:\Users\VJ\Desktop\3D-Generation\text_to_3D\shap-e')
xm, model, diffusion = load_models_and_config()

prompt = st.text_input('Enter a text prompt', "a plane looking like an apple ")
size = st.number_input('Enter the size', min_value=1, max_value=100, value=50)
render_mode = st.selectbox('Select render mode', ('nerf', 'stf'))

if st.button('Generate'):
    generate_3d(prompt, size, render_mode)
    # Display the 3D plots
    for i in range(len(latents)):
        with open(f'{prompt}_{i}.html', 'r') as f:
            html = f.read()
        st.components.v1.html(html, width=800, height=600)
