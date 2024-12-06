<!-- <div align="center">
  <h1>Negative Token Merging: Image-based Adversarial Feature Guidance</h1>
  <p>Official Implementation for our paper: Negative Token Merging: Image-based Adversarial Feature Guidance
 </p>
</div>
<br> -->

<div align="center">
  
## Negative Token Merging: Image-based Adversarial Feature Guidance
  [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)](https://negtome.github.io/)

[[Paper](https://negtome.github.io/docs/negtome.pdf)] &emsp; [[Project Page](https://negtome.github.io/)] &emsp;  [[🤗 Huggingface Demo Flux ](https://a62ec09c8038aea4ee.gradio.live/)] [[🤗 Huggingface Demo SDXL](https://4e116ddf3ab78d0b07.gradio.live/)]
<!-- [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://negtome.github.io/)  -->
<br>
</div>


---


<!-- ## Negative Token Merging: Image-based Adversarial Feature Guidance -->

Official Implementation for our paper: 
**Neg**ative **To**ken **Me**rging: Image-based Adversarial Feature Guidance


### What is NegToMe?

Using a negative prompt for avoiding generation of undesired concepts has emerged as a widely adopted approach. However, capturing complex visual concepts using text-alone is often not feasible (e.g, child in park in Fig. below) and can be insufficient (e.g., for removing copyrighted characters).

We propose NegToMe, which proposes to perform adversarial guidance directly using images (as opposed to text alone). The key idea is simple: even if describing the undesired concepts is not effective or feasible in text-alone
(e.g.: child in park} for figure below), we can directly use the visual features from a reference image in order to adversarially guide the generation process.


<div align="center">
  <img src="./docs/method-overview-v1.jpg" alt="Description of Image">
</div>

## News and Updates
**[2024.12.4]** Initial Release with Gradio Demo.


## Examples and Applications
Simply adjusting the used reference image, allows for a range of custom applications with NegToMe.

#### Increasing Output Diversity
> Using NegToMe across different outputs improves output diversity (by guiding visual features of each image away from others during reverse diffusion)
<div align="center">
  <img src="./docs/diversity-sdxl-v2.jpg" alt="Description of Image">
</div>


#### Copyright Mitigation
> When using copyrighted retrieval database (RAG) as reference, NegToMe allows for better reduction of visual similarity to cpyrighted images.
<div align="center">
  <img src="./docs/copyright-v1.jpg" alt="Description of Image">
</div>


#### Improving Output Aesthetics and Details
> Simply using a blurry / poor quality reference leads to improved output aesthetics and details without requiring any finetuning. (by guiding away from poor / blurry features)
<div align="center">
  <img src="./docs/output-quality-v2.jpg" alt="Description of Image">
</div>


#### Adversarial Style Guidance
> Using NegToMe w.r.t to a style reference image helps exclude certain artistic elements while still obtaining the desired output content. 
<div align="center">
  <img src="./docs/style-guidance-v1.jpg" alt="Description of Image">
</div>



## TODO / Updates
- [x] Initial Code Release
- [x] Implementation with Flux
- [x] Implementation with SDXL
- [x] Source code for gradio demo
- [ ] Masked Adversarial Guidance
- [ ] Low VRAM Gradio Demo
- [ ] Benchmark Release for Diversity and Copyright Mitigation


## Setup
To set up our environment, please run:

```
conda create -n negtome python=3.11 -y
conda activate negtome
pip install -r requirements.txt
```

<!-- ```
conda env create --name negtome --file=environment/environment.yml
``` -->

## Usage
NegToMe can be incorportated in just few lines of code in most state-of-the-art diffusion models. Currently we provide three ways of using NegToMe:


### Directly Using NegToMe Pipeline
First simply load the pipeline:
```python
from src.negtome.pipeline_negtome_flux import FluxNegToMePipeline
pipe = FluxNegToMePipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
```

Inference with and w/o negtome can then be run as,
```python
import torch
import time

negtome_args = {
    'use_negtome': False,
    'merging_alpha': 0.9,
    'merging_threshold': 0.65, 
    'merging_t_start': 1000, 
    'merging_t_end': 900,
    'num_joint_blocks': -1, # number of joint transformer blocks (flux) to apply negtome
    'num_single_blocks': -1, # number of single transformer blocks (flux) to apply negtome
}

# input prompt 
prompt = "a high resolution photo of a person"
print (f"using prompt: {prompt}")

# hyperparameters
seed = 0 
num_inference_steps = 25
num_images_per_prompt = 4 # generate 4 images across the batch
height = width = 768 

# generate both with and w/o negtome
inference_times = []
for use_negtome in [False, True]:
    print(f"\nuse_negtome: {use_negtome}")
    generator = torch.Generator(pipe.device).manual_seed(seed)
    
    # Measure time
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
        use_negtome=use_negtome,
        negtome_args=negtome_args,
    ).images
    elapsed_time = time.time() - start_time
    inference_times.append(elapsed_time)
    
    print(f"use_negtome: {use_negtome}\nTime taken: {elapsed_time:.2f} seconds")
    display(display_alongside_batch(images, resize_dims=512))

# Calculate percentage increase
percentage_increase = ((inference_times[1] - inference_times[0]) / inference_times[0]) * 100
print(f"\nPercentage increase in inference time with negtome: {percentage_increase:.2f}%")
```


### Use the Jupyter Notebook
Example usage in ```notebooks/demo-negtome-flux.ipynb``` and ```notebooks/demo-negtome-sdxl.ipynb```

### Start a local gradio demo
Run the following command:
```
python gradio_app_negtome.py
```



## Citation
If you find our work useful, please consider citing:
```
@article{singh2024negtome,
  title={Negative Token Merging: Image-based Adversarial Feature Guidance},
  author={Singh, Jaskirat and Li, Lindsey and Shi, Weijia and Krishna, Ranjay and Choi, Yejin and 
    Wei, Pang and Gould, Stephen and Zheng, Liang and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv}, 
  url={https://arxiv.org/abs/2408},
  year={2024}
}
```

---

