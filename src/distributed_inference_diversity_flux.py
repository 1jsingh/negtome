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

# from .pipeline_negtome_sdxl import StableDiffusionXLCopyCatPipeline
from src.negtome.pipeline_negtome_flux import FluxNegToMePipeline
from diffusers.utils import logging
import pickle

START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

####################################################
# inputs
####################################################
final_cat_list = ['animal', 'bird', 'mammal', 'person', 'man', 'woman', 'child', 'dog', 'cat', 'dragon', 'boat', 'building', 'bus', 'car', 'airplane', 'fish', 'bridge', 'insect/bug', 'snake', 'shirt', 'dress']

default_copycat_args = {
    ####################################################
    # timestep
    ####################################################
    'timestep': 0, # placeholder
    'time_idx_store': -1, # for storing hidden_states
    ####################################################
    # timestep
    ####################################################
    'use_crossframe_attn': False,
    'crossframe_attn_tstart': 1000,
    'crossframe_attn_tend': 800,
    'num_lang_tokens': 512,
    ####################################################
    # crossframe token merging 
    ####################################################
    'use_crossframe_token_merging': False,
    'merging_alpha': -1.9,#-2.,
    'merging_threshold': 0.65,
    'merging_dropout': 0.,
    'merging_t_start': 1000,
    'merging_t_end': 900,
    ####################################################
    # rag asset args
    ####################################################
    'use_rag_assets': False,
    'noise_rag': True,
    'rag_assets': None,#[:-1][::-1],
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
def main(category_list=final_cat_list, prompt_prefixs=['a high resolution photo of a'], prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, use_negtome=False, 
                 merging_alpha=-1.9, merging_threshold=0.65, merging_t_start=1000, merging_t_end=900, num_joint_blocks = -1, num_single_blocks = -1,
                 height=768, width=768, num_images_per_prompt=4, seed=0, num_inference_steps=25, neg_prompt=None, guidance_scale=3.5, 
                 batch_size=2, output_file="output.pkl", save_dir='/data/home/jaskirats/project/alphagen/t2i_infringe/data/diversity-flux',
                 low_mem=False, overwrite=False):

    output_file = f'guidance{guidance_scale}_seed{seed}_{output_file}'
    
    # Define negative prompt
    if neg_prompt is None:
        neg_prompt = ("naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, "
                      "poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, "
                      "distorted hands, amputation")

    # Update copycat_args using given arguments
    copycat_args = default_copycat_args.copy()
    copycat_args.update({
        'use_crossframe_token_merging': use_negtome,
        'merging_alpha': merging_alpha,
        'merging_threshold': merging_threshold,
        'merging_t_start': merging_t_start,
        'merging_t_end': merging_t_end,
    })
    model_id = "black-forest-labs/FLUX.1-dev"
    try:
        model_path = os.path.join("models", model_id)
        from src.utils import get_list_of_files_to_prepare
        to_prepare = get_list_of_files_to_prepare(model_path)
        from azfuse import File
        File.prepare(to_prepare)
        model_id = model_path
    except ImportError:
        pass

    save_dir = os.path.join(save_dir, model_id.replace("/", "_"))
    if use_negtome:
        output_file = f'alpha{abs(merging_alpha)}_t{merging_threshold}_start{merging_t_start}_end{merging_t_end}_{output_file}'

    if file_exists(os.path.join(save_dir, output_file)) and not overwrite:
        print(f"File {output_file} already exists in {save_dir}. Skipping...")
        return

    # Load the pipeline and move it to the specific GPU
    pipeline = FluxNegToMePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Step 1: Prepare prompts
    prompts_ = [f'{prompt_prefix} {x}. {prompt_suffix}' for x in category_list for prompt_prefix in prompt_prefixs]
    data_loader = get_batches(items=prompts_, batch_size=batch_size)

    distributed_state = Accelerator()
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
        with distributed_state.split_between_processes(prompts_raw) as prompts:
            # Set the seed for reproducibility
            # generator = torch.manual_seed(seed)
            if distributed_state.is_main_process:
                print (prompts)
            else:
                logging.set_verbosity_error()
                logging.disable_progress_bar()


            # hyperparmeters
            generator = torch.Generator(pipeline.device).manual_seed(seed)

            images = pipeline(
                prompt=prompts[0],
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator = generator,
                num_images_per_prompt=num_images_per_prompt,
                crossframe_args=copycat_args,
                num_joint_blocks = num_joint_blocks, # use all joint transformer blocks
                num_single_blocks = num_single_blocks, # use all single transformer blocks
            ).images
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
            'num_single_blocks': num_single_blocks,
            'num_joint_blocks': num_joint_blocks,
        }
        
        save_to_pkl(output_file, save_dir, overall_prompts[:len(prompts_)], overall_images[:len(prompts_)], args)

    if distributed_state.is_main_process:
        print(f">>> Image Generation Finished. Saved in {save_dir}")


def sample_for_fid_eval(prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, use_negtome=False, 
                 merging_alpha=-1.9, merging_threshold=0.65, merging_t_start=1000, merging_t_end=900, num_joint_blocks = -1, num_single_blocks=-1,
                 height=768, width=768, num_images_per_prompt=4, seed_start=0, seed_end=10, num_inference_steps=25, neg_prompt=None, guidance_scale=3.5, 
                 batch_size=2, save_dir='copycat/output_diversity',
                 low_mem=False, overwrite=False):
    category_list = [
        'bird', 'mammal', 'animal', 'fish',  'insect/bug', 'dog', 'cat',  
        'person', 'woman', 'man', 'child',
        'building', 'bus', 'car', 'airplane', 'boat', 'bridge', 'shirt', 'dress',
        'dragon',
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

    output_file = "output.pkl"
    for seed in range(seed_start, seed_end):
        main(category_list=category_list, prompt_prefixs=promtp_prefixs, prompt_suffix=prompt_suffix, gpu_idxs=gpu_idxs, default_copycat_args=default_copycat_args, guidance_scale=guidance_scale, use_negtome=use_negtome, 
             merging_alpha=merging_alpha, merging_threshold=merging_threshold, merging_t_start=merging_t_start, merging_t_end=merging_t_end,
             height=height, width=width, num_images_per_prompt=num_images_per_prompt, seed=seed, num_inference_steps=num_inference_steps, neg_prompt=neg_prompt,
             batch_size=batch_size, save_dir=save_dir, low_mem=low_mem, output_file=output_file, overwrite=overwrite, num_joint_blocks=num_joint_blocks, num_single_blocks=num_single_blocks)


def main_long_prompts(category_list=final_cat_list, prompt_prefixs=['a high resolution photo of a'], prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, use_negtome=False, 
                 merging_alpha=-1.9, merging_threshold=0.65, merging_t_start=1000, merging_t_end=900, num_joint_blocks = -1, num_single_blocks = -1,
                 height=768, width=768, num_images_per_prompt=4, seed=0, num_inference_steps=25, neg_prompt=None, guidance_scale=3.5, 
                 batch_size=2, output_file="output.pkl", save_dir='/data/home/jaskirats/project/alphagen/t2i_infringe/data/diversity-flux',
                 low_mem=False, overwrite=False, group_prompts=1):

    output_file = f'guidance{guidance_scale}_seed{seed}_{output_file}'
    
    # Define negative prompt
    if neg_prompt is None:
        neg_prompt = ("naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, "
                      "poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, "
                      "distorted hands, amputation")

    # Update copycat_args using given arguments
    copycat_args = default_copycat_args.copy()
    copycat_args.update({
        'use_crossframe_token_merging': use_negtome,
        'merging_alpha': merging_alpha,
        'merging_threshold': merging_threshold,
        'merging_t_start': merging_t_start,
        'merging_t_end': merging_t_end,
    })
    model_id = "black-forest-labs/FLUX.1-dev"
    try:
        model_path = os.path.join("models", model_id)
        from src.utils import get_list_of_files_to_prepare
        to_prepare = get_list_of_files_to_prepare(model_path)
        from azfuse import File
        File.prepare(to_prepare)
        model_id = model_path
    except ImportError:
        pass

    save_dir = os.path.join(save_dir, model_id.replace("/", "_"))
    if use_negtome:
        output_file = f'alpha{abs(merging_alpha)}_t{merging_threshold}_start{merging_t_start}_end{merging_t_end}_{output_file}'

    if file_exists(os.path.join(save_dir, output_file)) and not overwrite:
        print(f"File {output_file} already exists in {save_dir}. Skipping...")
        return

    # Load the pipeline and move it to the specific GPU
    pipeline = FluxNegToMePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Step 1: Prepare prompts
    prompts_ = [f'{prompt_prefix} {x}. {prompt_suffix}' for x in category_list for prompt_prefix in prompt_prefixs]
    data_loader = get_batches(items=prompts_, batch_size=batch_size*group_prompts)

    distributed_state = Accelerator()
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
        with distributed_state.split_between_processes(prompts_raw) as prompts:
            # Set the seed for reproducibility
            # generator = torch.manual_seed(seed)
            if distributed_state.is_main_process:
                print (prompts)
            else:
                logging.set_verbosity_error()
                logging.disable_progress_bar()


            # hyperparmeters
            generator = torch.Generator(pipeline.device).manual_seed(seed)

            images = pipeline(
                prompt=prompts,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator = generator,
                num_images_per_prompt=num_images_per_prompt // group_prompts,
                crossframe_args=copycat_args,
                num_joint_blocks = num_joint_blocks, # use all joint transformer blocks
                num_single_blocks = num_single_blocks, # use all single transformer blocks
            ).images
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
            'num_single_blocks': num_single_blocks,
            'num_joint_blocks': num_joint_blocks,
        }
        
        save_to_pkl(output_file, save_dir, overall_prompts[:len(prompts_)], overall_images[:len(prompts_)], args)

    if distributed_state.is_main_process:
        print(f">>> Image Generation Finished. Saved in {save_dir}")


def sample_long_prompt(prompt_file, prompt_suffix='', gpu_idxs=[0], default_copycat_args=default_copycat_args, use_negtome=False, 
                 merging_alpha=-0.9, merging_threshold=0.65, merging_t_start=1000, merging_t_end=950, num_joint_blocks = -1, num_single_blocks=-1,
                 height=768, width=768, num_images_per_prompt=4, seed_start=0, seed_end=8, num_inference_steps=25, neg_prompt=None, guidance_scale=3.5, 
                 batch_size=2, save_dir='copycat/output_diversity',
                 low_mem=False, overwrite=False, group_prompts=1):
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
    save_dir = os.path.join(save_dir, prompt_file_name)

    output_file = "output.pkl"
    for seed in range(seed_start, seed_end):
        main_long_prompts(group_prompts=group_prompts, category_list=prompts, prompt_prefixs=promtp_prefixs, prompt_suffix=prompt_suffix, gpu_idxs=gpu_idxs, default_copycat_args=default_copycat_args, guidance_scale=guidance_scale, use_negtome=use_negtome, 
             merging_alpha=merging_alpha, merging_threshold=merging_threshold, merging_t_start=merging_t_start, merging_t_end=merging_t_end,
             height=height, width=width, num_images_per_prompt=num_images_per_prompt, seed=seed, num_inference_steps=num_inference_steps, neg_prompt=neg_prompt,
             batch_size=batch_size, save_dir=save_dir, low_mem=low_mem, output_file=output_file, overwrite=overwrite, num_joint_blocks=num_joint_blocks, num_single_blocks=num_single_blocks)


if __name__ == "__main__":
    fire.Fire()
