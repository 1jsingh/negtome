import os
import pickle
import json
import time
from tqdm import tqdm
import torch
import fire

from collections import defaultdict
from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers.utils import logging

# If your pipeline is in src/negtome/pipeline_negtome_sdxl.py, adapt the import:
from src.negtome.pipeline_negtome_sdxl import StableDiffusionXLNegToMePipeline

###############################################################################
# Default: categories, prefixes, negative prompt, CADS args
###############################################################################
CATEGORY_LIST = [
    'bird', 'mammal', 'animal', 'fish', 'building', 'bus', 'car', 'airplane',
    'insect/bug', 'dog', 'cat', 'dragon', 'boat', 'bridge',
    'person', 'woman', 'man', 'child', 'shirt', 'dress'
]
PROMPT_PREFIXES = [
    'a photo of a',
    'a high resolution cinematic photo of a',
    'a hyper-realistic digital painting of a',
    'a watercolor illustration of a',
    'an oil painting of a'
]

DEFAULT_NEG_PROMPT = (
    "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, "
    "disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, "
    "watermarks, oversaturated, distorted hands, amputation"
)

# Example CADS args (if you need them; set `use_cads=True` to enable)
CADS_ARGS = {
    "use_cads": True,       # Enable or disable CADS
    "tau1": 0.6,            # t-end
    "tau2": 0.9,            # t-start
    "noise_scale": 0.25,    # Scale of noise injected
    "mixing_factor": 1.0,   # 1.0 => fully re-normalize after adding noise
    "rescale": True,        # Whether to re-normalize the latents after mixing
}


###############################################################################
# Helpers
###############################################################################
def get_prompts_from_cat_and_prefix(category_list, prompt_prefixes):
    """
    Build a list of prompts by combining each prefix with each category.
    """
    prompts = []
    for prefix in prompt_prefixes:
        for cat in category_list:
            # e.g., "a photo of a bird"
            prompt = f"{prefix} {cat}"
            prompts.append(prompt.strip())
    return prompts

def get_prompts_from_coco_captions(coco_caption_file, max_num_captions=5):
    """
    Load captions from a COCO-style annotations file (e.g. captions_val2017.json).
    Return a list of text prompts, up to `max_num_captions` per image.
    """
    with open(coco_caption_file, 'r') as f:
        coco_anns = json.load(f)

    # We'll gather captions by image_id
    id2captions = defaultdict(list)
    for ann in coco_anns["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        id2captions[image_id].append(caption)

    # Flatten into a list of prompts, up to max_num_captions per image
    all_prompts = []
    for image_id, caps in id2captions.items():
        # limit each image to `max_num_captions`
        all_prompts.extend(caps[:max_num_captions])
    return all_prompts

def get_batches(items, batch_size):
    """
    Given a list of items, yield sublists (batches) of size `batch_size`.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

def save_results_to_pkl(results, save_path):
    """
    Saves a dictionary `results` to a pickle file at `save_path`.
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")


###############################################################################
# Main distributed function
###############################################################################
def main(
    # -- Prompt-building modes
    category_list=CATEGORY_LIST,
    prompt_prefixes=PROMPT_PREFIXES,
    coco_captions=None,     # path to a COCO caption file (e.g. captions_val2017.json)
    max_num_captions=5,     # how many captions per image to use if coco_captions is given
    max_num_prompts=100,

    # -- Generation config
    neg_prompt=DEFAULT_NEG_PROMPT,
    batch_size=2,
    num_images_per_prompt=4,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=0,
    height=512,
    width=512,
    model_id="SG161222/RealVisXL_V4.0",

    # -- Output
    output_pkl="./output/distributed_results.pkl",

    # -- CADS
    use_cads=False,
    tau1=0.6,
    tau2=0.9,
    noise_scale=0.25,
    mixing_factor=1.0,
    rescale=True,

    # -- NegToMe
    use_negtome=False,
    merging_alpha=1.9,
    merging_threshold=0.65,
    merging_t_start=1000,
    merging_t_end=900,
):
    """
    Distributed inference script (run with `accelerate launch`).

    Examples:
      1) Category + prefix mode (default):
         accelerate launch distributed_inference_script.py --batch_size=4

      2) COCO caption mode:
         accelerate launch distributed_inference_script.py --coco_captions=./data/coco2017_val/annotations/captions_val2017.json \
                                                          --max_num_captions=5 \
                                                          --batch_size=2
    """
    # Build the final prompt list
    if coco_captions is not None:
        # Use COCO captions
        all_prompts = get_prompts_from_coco_captions(coco_captions, max_num_captions=max_num_captions)
        print(f"[COCO CAPTIONS MODE] Found {len(all_prompts)} captions (max {max_num_captions} per image).")
    else:
        # Category + prefix combos
        all_prompts = get_prompts_from_cat_and_prefix(category_list, prompt_prefixes)
        print(f"[CATEGORY+PREFIX MODE] Built {len(all_prompts)} prompts from categories + prefixes.")

    import random
    random.shuffle(all_prompts)
    all_prompts  = all_prompts[:max_num_prompts]

    # Initialize Accelerator for distributed usage
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Running on device: {device}")
        print(f"Will generate {len(all_prompts)} prompts x {num_images_per_prompt} images each = "
              f"{len(all_prompts)*num_images_per_prompt} total images.")

    # Prepare pipeline
    pipeline = StableDiffusionXLNegToMePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipeline.to(device)

    # If needed, enable CPU offload:
    # pipeline.enable_model_cpu_offload(gpu_id=device.index)

    # Prepare CADS args
    cads_args = {
        "use_cads": use_cads,
        "tau1": tau1,
        "tau2": tau2,
        "noise_scale": noise_scale,
        "mixing_factor": mixing_factor,
        "rescale": rescale,
    }

    # Prepare NegToMe args
    negtome_args = {
        'use_negtome': use_negtome,
        'merging_alpha': merging_alpha,
        'merging_threshold': merging_threshold,
        'merging_dropout': 0.0,
        'merging_t_start': merging_t_start,
        'merging_t_end': merging_t_end,
    }

    # We'll store final results here on the main process:
    all_collected_prompts = []
    all_collected_images = []

    # Build data loader
    prompt_loader = get_batches(all_prompts, batch_size=batch_size)

    # For reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)

    # MAIN LOOP
    for batch in tqdm(prompt_loader, disable=not accelerator.is_main_process):
        # Split this batch among multiple processes
        with accelerator.split_between_processes(batch) as local_prompts:
            if not accelerator.is_main_process:
                logging.set_verbosity_error()
                logging.disable_progress_bar()

            if len(local_prompts) == 0:
                continue

            # Generate images with your pipeline
            output = pipeline(
                prompt=local_prompts,
                negative_prompt=neg_prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,

                # custom arguments
                use_cads=use_cads,
                cads_args=cads_args,
                use_negtome=use_negtome,
                negtome_args=negtome_args,
            )
            # pipeline output is typically with .images
            images = output.images  # this is a list, length = len(local_prompts) * num_images_per_prompt

        # Synchronize across processes
        accelerator.wait_for_everyone()

        # Gather images + prompts from all ranks
        gathered_images = gather_object(images)
        gathered_prompts = gather_object(local_prompts)

        # Append to the global lists (on main process)
        all_collected_images.extend(gathered_images)
        all_collected_prompts.extend(gathered_prompts)

        # Free up GPU memory
        torch.cuda.empty_cache()

    ############################################################################
    # Save results (only on main process)
    ############################################################################
    if accelerator.is_main_process:
        total_prompts = len(all_collected_prompts)
        print(f"\nCompleted generation of {total_prompts} prompts, each prompt has {num_images_per_prompt} images.")
        # Store everything in a dict
        results = {
            "prompts": all_collected_prompts,
            "images": all_collected_images,
            "args": {
                # which mode was used
                "coco_captions": coco_captions,
                "max_num_captions": max_num_captions,
                "category_list": category_list,
                "prompt_prefixes": prompt_prefixes,

                # generation params
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "seed": seed,
                "batch_size": batch_size,
                "neg_prompt": neg_prompt,

                # cads + negtome
                "use_cads": use_cads,
                "cads_args": cads_args,
                "use_negtome": use_negtome,
                "negtome_args": negtome_args,

                "model_id": model_id,
            }
        }
        save_results_to_pkl(results, output_pkl)
        print(f"Done. Results saved to {output_pkl}.")


if __name__ == "__main__":
    fire.Fire(main)