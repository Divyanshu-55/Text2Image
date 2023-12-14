from flask import Flask, jsonify, request
from pathlib import Path
import sys
import torch
import os
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
# import streamlit as st

model_path = '800'             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = None
         
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

app = Flask(__name__)
  
@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
  
        data = "Text2Image"
        return jsonify({'service': data})
  

@app.route("/generate", methods=["POST"])
def generate():

    prompt = request.form['prompt']
    negative_prompt = request.form['Negative prompt']
    num_samples = request.form['No. of samples']

    guidance_scale = 7.5
    num_inference_steps = 24
    height = 512
    width = 512

    g_cuda = torch.Generator(device='cuda')
    seed = 52362
    g_cuda.manual_seed(seed)

    # commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")

    with autocast("cpu"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=int(num_samples),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images
    
    Output = {}

    import base64
    for i, img in enumerate(images):
        img.save('dog' + str(i) + '.jpg')
        with open('/content/dog' + str(i) + '.jpg', "rb") as image2string: 
            converted_string = base64.b64encode(image2string.read())
        converted_string = converted_string.decode("utf-8")
        Output["Image" + str(i)] = converted_string
    
    return jsonify(Output)


  
# driver function
if __name__ == '__main__':
  
    app.run(debug = True)
