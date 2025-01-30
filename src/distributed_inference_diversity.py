# Originally by jiwooya1000, put together together by sayakpaul.
# Documentation: https://huggingface.co/docs/diffusers/main/en/training/distributed_inference

"""
Run: 

accelerate launch distributed_inference_diffusers.py --batch_size 8

# Enable memory optimizations for large models like SD3
accelerate launch distributed_inference_diffusers.py --batch_size 8 --low_mem=1
"""

from diffusers import DiffusionPipeline
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch
import time
import os
import fire

import sys
src_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(src_folder))
from src.utils import patch_file_open, pilimg_from_base64, file_exists

from src.negtome.pipeline_negtome_sdxl import StableDiffusionXLNegToMePipeline
from diffusers.utils import logging
import pickle

START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

####################################################
# inputs
####################################################
final_cat_list = ['bird', 'mammal', 'person', 'man', 'woman', 'child', 'dog', 'cat', 'dragon', 'boat', 'building', 'bus', 'car', 'airplane', 'fish', 'reptile', 'insect/bug', 'snake', 'shirt', 'dress']

default_copycat_args = {
    ####################################################
    # crossframe token merging 
    ####################################################
    'use_negtome': False,
    'merging_alpha': 0.9,#-2.,
    'merging_alpha_neg': -0.7,#-0.5,
    'merging_threshold': 0.7,
    'merging_dropout': 0.,
    'merging_t_start': 1000,
    'merging_t_end': 600,
    'min_selfattn_dim': 32, # min dimension of the layers for applying tokern merging
    'blocks': ['up_blocks','down_blocks', 'mid_block'][:1], # blocks for which token merging is applied e.g. only up_blocks
    # 'merge_negative_pass': True,
    # 'merge_cross_attn': True,
    ####################################################
    # rag asset args
    ####################################################
    'use_rag_assets': False,
    'noise_rag': False,
    'rag_assets': None,#rag_assets[2:],#[:-1][::-1],
    ####################################################
    # self-attention from rag anchor to the neg image
    ####################################################
    'use_neg_selfattn': False,
    'neg_selfattn_dropout': 0.6,
    'neg_selfattn_tstart': 750,
    'neg_selfattn_tend': 700,
}

cads_args = {
    "use_cads": False,            # The pipeline code uses either `use_cads` or cads_args["use_cads"]
    "tau1": 0.6,                 # t-end
    "tau2": 0.9,                 # t-start
    "noise_scale": 0.25,         # Scale of noise injected
    "mixing_factor": 1.0,        # 1.0 => fully re-normalize after adding noise, 0.0 => no renormalization
    "rescale": True,             # Whether to do that renormalization
}

####################################################
# utils
####################################################
def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches

def save_to_pkl(output_file, save_dir, prompts, images, args):
    """Save the collected data to a .pkl file.

    Args:
        output_file (str): The path where the .pkl file will be saved.
        prompts (list): List of prompts used for image generation.
        images (list): List of generated images.
        args (dict): Dictionary of arguments used for image generation.
    """
    data = {
        'prompts': prompts,
        'images': images,
        'args': args
    }

    with patch_file_open(os.path.join(save_dir, output_file) , 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_file}")

####################################################
# main generation fn
####################################################
def main(category_list=final_cat_list, prompt_prefixs=['a hyper-realistic digital painting of a'], prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, guidance_scale=5.0, use_negtome=False, 
                 merging_alpha=0.9, merging_threshold=0.7, merging_t_start=1000, merging_t_end=600,
                 height=1024, width=1024, num_images_per_prompt=4, seed=0, num_inference_steps=50, neg_prompt=None,
                 batch_size=1, output_file="output.pkl", save_dir='/data/home/jaskirats/project/alphagen/t2i_infringe/data/diversity-sdxl',
                 low_mem=False, overwrite=False, use_cads=False):

    output_file = f'guidance{guidance_scale}_seed{seed}_{output_file}'
    
    # Define negative prompt
    if neg_prompt is None:
        neg_prompt = ("naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, "
                      "poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, "
                      "distorted hands, amputation")

    # Update copycat_args using given arguments
    copycat_args = default_copycat_args.copy()
    copycat_args.update({
        'use_negtome': use_negtome,
        'merging_alpha': merging_alpha,
        'merging_threshold': merging_threshold,
        'merging_t_start': merging_t_start,
        'merging_t_end': merging_t_end,
    })
    model_id = "SG161222/RealVisXL_V4.0"
    save_dir = os.path.join(save_dir, model_id.replace("/", "_"))
    # save_dir = save_dir # + f"_{START_TIME}"
    if use_negtome:
        output_file = f'alpha{abs(merging_alpha)}_t{merging_threshold}_start{merging_t_start}_end{merging_t_end}_{output_file}'
    elif use_cads:
        output_file = f'cads_tau1{cads_args["tau1"]}_tau2{cads_args["tau2"]}_noise{cads_args["noise_scale"]}_mix{cads_args["mixing_factor"]}_rescale{cads_args["rescale"]}_{output_file}'

    if file_exists(os.path.join(save_dir, output_file)) and not overwrite:
        print(f"File {output_file} already exists in {save_dir}. Skipping...")
        return
    # Load the pipeline and move it to the specific GPU
    pipeline = StableDiffusionXLNegToMePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ) 
    

    # Step 1: Prepare prompts
    prompts_ = [f'{prompt_prefix} {x}. {prompt_suffix}' for x in category_list for prompt_prefix in prompt_prefixs]
    data_loader = get_batches(items=prompts_, batch_size=batch_size)

    from accelerate import PartialState 

    distributed_state = Accelerator()
    # distributed_state = PartialState()
    # print distributed_state
    print(distributed_state.device.index)

    if low_mem:
        pipeline.enable_model_cpu_offload(gpu_id=distributed_state.device.index)
    else:
        pipeline = pipeline.to(distributed_state.device)

    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            # os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    count = 0
    overall_prompts = []
    overall_images = []
    for _, prompts_raw in tqdm(enumerate(data_loader), total=len(data_loader)):
        with distributed_state.split_between_processes(prompts_raw) as prompts:
            # Set the seed for reproducibility
            # generator = torch.manual_seed(seed)
            if distributed_state.is_main_process:
                print (prompts)
            else:
                logging.set_verbosity_error()
                logging.disable_progress_bar()

            # Generate images using the pipeline
            images = pipeline(
                prompt=prompts[0], 
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompt, 
                num_images_per_prompt=num_images_per_prompt, 
                # generator=generator,
                seed=seed,
                num_inference_steps=num_inference_steps, 
                use_negtome=use_negtome,
                negtome_args=copycat_args,
                use_cads=use_cads,
                cads_args=cads_args,
            ).images
            print(len(images))
            images = [images]
            input_prompts = prompts

        distributed_state.wait_for_everyone()

        images = gather_object(images)
        print(len(images), len(images[0]))
        input_prompts = gather_object(input_prompts)
        overall_prompts.extend(input_prompts)
        overall_images.extend(images)
        torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        # Save the results to a .pkl file
        args = {
            'category_list': category_list,
            'prompt_prefix': prompt_prefixs,
            'prompt_suffix': prompt_suffix,
            'gpu_idxs': gpu_idxs,
            'default_copycat_args': default_copycat_args,
            'use_crossframe_token_merging': use_negtome,
            'merging_alpha': merging_alpha,
            'merging_threshold': merging_threshold,
            'merging_t_start': merging_t_start,
            'merging_t_end': merging_t_end,
            'height': height,
            'width': width,
            'num_images_per_prompt': num_images_per_prompt,
            'seed': seed,
            'num_inference_steps': num_inference_steps,
            'neg_prompt': neg_prompt,
            'use_cads': use_cads,
            'cads_args': cads_args,
        }
        
        save_to_pkl(output_file, save_dir, overall_prompts[:len(prompts_)], overall_images[:len(prompts_)], args)

    if distributed_state.is_main_process:
        print(f">>> Image Generation Finished. Saved in {save_dir}")


def sample_for_fid_eval(prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, guidance_scale=5.0, use_negtome=False, 
                 merging_alpha=0.9, merging_threshold=0.7, merging_t_start=1000, merging_t_end=600,
                 height=1024, width=1024, num_images_per_prompt=4, seed_start=0, seed_end=10, num_inference_steps=50, neg_prompt=None,
                 batch_size=2, save_dir='copycat/output_diversity',
                 low_mem=False, overwrite=False, use_cads=False):
    category_list = [
        'bird', 'mammal', 'animal', 'fish',  'building', 'bus', 'car', 'airplane', 'insect/bug', 'dog', 'cat',  'dragon', 'boat', 'bridge', 

        'person', 'woman', 'man', 'child','shirt', 'dress'
    ]
    promtp_prefixs = [
        'a photo of a ',
        'a good photo of a ',
        # 'a high quality image of a ',
        # 'a high resolution cinematic photo of a ',
        # 'a hyper-realistic digital painting of a '
        'a high contrast photo of a  ',
        'a photo of the ',
        'a high contrast photo of the ',
        'a good photo of the ',
        'a high quality image of a ',
    ]

    # output_file = "non_human_output.pkl"
    output_file = "output.pkl"
    for seed in range(seed_start, seed_end):
        main(category_list=category_list, prompt_prefixs=promtp_prefixs, prompt_suffix=prompt_suffix, gpu_idxs=gpu_idxs, default_copycat_args=default_copycat_args, guidance_scale=guidance_scale, use_negtome=use_negtome, 
             merging_alpha=merging_alpha, merging_threshold=merging_threshold, merging_t_start=merging_t_start, merging_t_end=merging_t_end,
             height=height, width=width, num_images_per_prompt=num_images_per_prompt, seed=seed, num_inference_steps=num_inference_steps, neg_prompt=neg_prompt,
             batch_size=batch_size, save_dir=save_dir, low_mem=low_mem, output_file=output_file, overwrite=overwrite, use_cads=use_cads)


def main_per_prompts(category_list=final_cat_list, prompt_prefixs=['a hyper-realistic digital painting of a'], prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, guidance_scale=5.0, use_negtome=False, 
                 merging_alpha=0.9, merging_threshold=0.7, merging_t_start=1000, merging_t_end=600,
                 height=1024, width=1024, num_images_per_prompt=4, seed=0, num_inference_steps=50, neg_prompt=None,
                 batch_size=1, output_file="output.pkl", save_dir='/data/home/jaskirats/project/alphagen/t2i_infringe/data/diversity-sdxl',
                 low_mem=False, overwrite=False, use_cads=False, group_prompts=1):

    output_file = f'guidance{guidance_scale}_seed{seed}_{output_file}'
    
    # Define negative prompt
    if neg_prompt is None:
        neg_prompt = ("naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, "
                      "poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, "
                      "distorted hands, amputation")

   # Update copycat_args using given arguments
    copycat_args = default_copycat_args.copy()
    copycat_args.update({
        'use_negtome': use_negtome,
        'merging_alpha': merging_alpha,
        'merging_threshold': merging_threshold,
        'merging_t_start': merging_t_start,
        'merging_t_end': merging_t_end,
    })
    model_id = "SG161222/RealVisXL_V4.0"
    save_dir = os.path.join(save_dir, model_id.replace("/", "_"))
    if use_negtome:
        output_file = f'alpha{abs(merging_alpha)}_t{merging_threshold}_start{merging_t_start}_end{merging_t_end}_{output_file}'
    
    elif use_cads:
        output_file = f'cads_tau1{cads_args["tau1"]}_tau2{cads_args["tau2"]}_noise{cads_args["noise_scale"]}_mix{cads_args["mixing_factor"]}_rescale{cads_args["rescale"]}_{output_file}'

    if file_exists(os.path.join(save_dir, output_file)) and not overwrite:
        print(f"File {output_file} already exists in {save_dir}. Skipping...")
        return
    # Load the pipeline and move it to the specific GPU
    pipeline = StableDiffusionXLNegToMePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ) 

    # Step 1: Prepare prompts
    prompts_ = [f'{prompt_prefix} {x}. {prompt_suffix}' for x in category_list for prompt_prefix in prompt_prefixs]
    data_loader = get_batches(items=prompts_, batch_size=batch_size*group_prompts)

    distributed_state = Accelerator()
    print(distributed_state.device.index)
    if low_mem:
        pipeline.enable_model_cpu_offload(gpu_id=distributed_state.device.index)
    else:
        pipeline = pipeline.to(distributed_state.device)

    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    count = 0
    overall_prompts = []
    overall_images = []
    for _, prompts_raw in tqdm(enumerate(data_loader), total=len(data_loader)):
        print(len(prompts_raw))
        with distributed_state.split_between_processes(prompts_raw) as prompts:
            # Set the seed for reproducibility
            # generator = torch.manual_seed(seed)
            if distributed_state.is_main_process:
                print (prompts)
            # else:
            #     logging.set_verbosity_error()
            #     logging.disable_progress_bar()


            # hyperparmeters
            generator = torch.Generator(pipeline.device).manual_seed(seed)

            print(prompts[0])
            # Generate images using the pipeline
            images = pipeline(
                prompt=prompts[0], 
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompt, 
                num_images_per_prompt=num_images_per_prompt // group_prompts, 
                # generator=generator,
                seed=seed,
                num_inference_steps=num_inference_steps, 
                use_negtome=use_negtome,
                negtome_args=copycat_args,
                use_cads=use_cads,
                cads_args=cads_args,
            ).images
            print(len(images))
            images = [images]
            input_prompts = prompts

        distributed_state.wait_for_everyone()

        images = gather_object(images)
        input_prompts = gather_object(input_prompts)
        overall_prompts.extend(input_prompts)
        overall_images.extend(images)

    if distributed_state.is_main_process:
        # Save the results to a .pkl file
        args = {
            'category_list': category_list,
            'prompt_prefix': prompt_prefixs,
            'prompt_suffix': prompt_suffix,
            'gpu_idxs': gpu_idxs,
            'default_copycat_args': default_copycat_args,
            'use_crossframe_token_merging': use_negtome,
            'merging_alpha': merging_alpha,
            'merging_threshold': merging_threshold,
            'merging_t_start': merging_t_start,
            'merging_t_end': merging_t_end,
            'height': height,
            'width': width,
            'num_images_per_prompt': num_images_per_prompt,
            'seed': seed,
            'num_inference_steps': num_inference_steps,
            'neg_prompt': neg_prompt,
            'use_cads': use_cads,
            'cads_args': cads_args,
        }
        print(len(overall_prompts), len(overall_images), len(overall_images[0]))
        save_to_pkl(output_file, save_dir, overall_prompts[:len(prompts_)], overall_images[:len(prompts_)], args)

    if distributed_state.is_main_process:
        print(f">>> Image Generation Finished. Saved in {save_dir}")


def sample_coco_prompt(prompt_file, prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, use_negtome=False, 
                 merging_alpha=0.9, merging_threshold=0.7, merging_t_start=1000, merging_t_end=600,
                 height=1024, width=1024, num_images_per_prompt=4, seed_start=0, seed_end=3, num_inference_steps=50, neg_prompt=None,
                 batch_size=2, save_dir='copycat/output_diversity',
                 low_mem=False, overwrite=False, use_cads=False, group_prompts=1, debug=False):
    prompts = []
    with patch_file_open(prompt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line.endswith("."):
                line = line[:-1]
            prompts.append(line)
    
    promtp_prefixs = ['']

    prompt_file_name = os.path.basename(prompt_file)
    print("num prompts", len(prompts))
    if debug:
        prompts = prompts[:4]
        prompt_file_name += "_debug"
        seed_end = seed_start + 1
        print("debugging", len(prompts))
    save_dir = os.path.join(save_dir, prompt_file_name)


    output_file = "output.pkl"
    for seed in range(seed_start, seed_end):
        main_per_prompts(group_prompts=group_prompts, category_list=prompts, prompt_prefixs=promtp_prefixs, prompt_suffix=prompt_suffix, gpu_idxs=gpu_idxs, default_copycat_args=default_copycat_args, use_negtome=use_negtome, 
             merging_alpha=merging_alpha, merging_threshold=merging_threshold, merging_t_start=merging_t_start, merging_t_end=merging_t_end,
             height=height, width=width, num_images_per_prompt=num_images_per_prompt, seed=seed, num_inference_steps=num_inference_steps, neg_prompt=neg_prompt,
             batch_size=batch_size, save_dir=save_dir, low_mem=low_mem, output_file=output_file, overwrite=overwrite, use_cads=use_cads)



def sample_for_tau(prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, guidance_scale=5.0, use_negtome=True, 
                 merging_alpha=0.9, merging_t_start=1000, merging_t_end=600,
                 height=1024, width=1024, num_images_per_prompt=4, seed_start=0, seed_end=5, num_inference_steps=50, neg_prompt=None,
                 batch_size=2, save_dir='copycat/output_diversity',
                 low_mem=False, overwrite=False, use_cads=False):
    category_list = [
        'bird', 'mammal', 'animal', 'fish',  'building', 'bus', 'car', 'airplane', 'insect/bug', 'dog', 'cat',  'dragon', 'boat', 'bridge', 

        'person', 'woman', 'man', 'child','shirt', 'dress'
    ]
    promtp_prefixs = [
        # 'a photo of a ',
        # 'a good photo of a ',
        # 'a high quality image of a ',
        # 'a high resolution cinematic photo of a ',
        'a hyper-realistic digital painting of a '
        # 'a high contrast photo of a  ',
        # 'a photo of the ',
        # 'a high contrast photo of the ',
        # 'a good photo of the ',
        # 'a high quality image of a ',
    ]

    # output_file = "non_human_output.pkl"
    output_file = "output.pkl"
    for merging_threshold in [1.0, 0.7, 0.8, 0.5]:
        for seed in range(seed_start, seed_end):
            main(category_list=category_list, prompt_prefixs=promtp_prefixs, prompt_suffix=prompt_suffix, gpu_idxs=gpu_idxs, default_copycat_args=default_copycat_args, guidance_scale=guidance_scale, use_negtome=use_negtome, 
                merging_alpha=merging_alpha, merging_threshold=merging_threshold, merging_t_start=merging_t_start, merging_t_end=merging_t_end,
                height=height, width=width, num_images_per_prompt=num_images_per_prompt, seed=seed, num_inference_steps=num_inference_steps, neg_prompt=neg_prompt,
                batch_size=batch_size, save_dir=save_dir, low_mem=low_mem, output_file=output_file, overwrite=overwrite, use_cads=use_cads)

if __name__ == "__main__":
    fire.Fire()
    