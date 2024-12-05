import gradio as gr
import torch
from src.negtome.pipeline_negtome_sdxl import StableDiffusionXLNegToMePipeline

##############################################
# Load the pipeline
##############################################
pipe = StableDiffusionXLNegToMePipeline.from_pretrained("SG161222/RealVisXL_V4.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

##############################################
# generate_images function
##############################################
def generate_images(
    prompt,
    merging_alpha=1.9,
    merging_threshold=0.7,
    merging_t_start=1000,
    merging_t_end=800,
    seed=0,
    num_inference_steps=50,
    num_images_per_prompt=4,
    height=1024,
    width=1024,
    use_negtome=False,
    neg_prompt="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted",
):
    # Convert parameters to appropriate types
    merging_t_start = int(merging_t_start)
    merging_t_end = int(merging_t_end)
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    num_images_per_prompt = int(num_images_per_prompt)
    height = int(height)
    width = int(width)
    
    negtome_args = {
        'use_negtome': use_negtome,
        'merging_alpha': merging_alpha,
        'merging_threshold': merging_threshold, 
        'merging_t_start': merging_t_start, 
        'merging_t_end': merging_t_end,
        'blocks': ['up_blocks'],
    }
    
    # Set up the random generator with the provided seed
    generator = torch.Generator(pipe.device).manual_seed(seed)
    
    # Run inference
    with torch.no_grad():
        output = pipe(
        prompt=prompt,
        guidance_scale=5.0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=neg_prompt,
        use_negtome=use_negtome,
        negtome_args=negtome_args,
    )
    
    # Clear the cache to avoid memory issues
    torch.cuda.empty_cache()
    
    # Return the images
    return output.images

##############################################
# Define the Gradio interface using Blocks
##############################################
with gr.Blocks() as demo:
    gr.Markdown("# NegToMe Demo: Increasing Output Diversity (SDXL) 🚀")
    gr.Markdown("🔥🔥 NegToMe helps significantly increase output diversity by guiding features of each image away from each other. 🔥🔥")
    gr.Markdown("🤗 [[Project Page](https://negtome.github.io/)] [[Paper](https://negtome.github.io/docs/negtome.pdf)]    [[GitHub](https://github.com/1jsingh/negtome)] 🤗")
    
    with gr.Row():
        # Adjust the width of the prompt textbox
        prompt = gr.Textbox(
            lines=2,
            label="Prompt",
            value="a 3D animation of a cute cat",
            elem_id="prompt_textbox"
        )
    
    # Apply custom CSS to adjust widths and sizes
    demo.css = """
    #prompt_textbox textarea {
        width: 100% !important;
    }
    """
    merging_alpha = gr.Slider(minimum=-1., maximum=3., step=0.1, value=1.5, label="Merging Alpha (controls diversity: higher alpha pushes images further apart)")
    merging_threshold = gr.Slider(minimum=0.5, maximum=1, step=0.05, value=0.7, label="Merging Threshold (controls which features are pushed apart: higher threshold preserves original features more)")


    with gr.Accordion("Advanced Settings", open=False):
        merging_t_start = gr.Slider(minimum=950, maximum=1000, step=10., value=950, label="Merging t_start")
        merging_t_end = gr.Slider(minimum=700, maximum=950, step=10., value=800, label="Merging t_end")
        seed = gr.Number(value=0, label="Seed", precision=0)
        num_inference_steps = gr.Slider(minimum=25, maximum=100, step=5, value=50, label="Number of Inference Steps")
        num_images_per_prompt = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Number of Images per Prompt")
        neg_prompt = gr.Textbox(lines=2, label="Negative Prompt", value="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation", elem_id="neg_prompt_textbox")
            
    with gr.Row():
        gen_without_button = gr.Button("Generate Images without NegToMe")
        gen_with_button = gr.Button("Generate Images with NegToMe")
        run_both_button = gr.Button("Run Both", elem_id="run_both_button")
    
    # Apply custom CSS to style the "Run Both" button
    demo.css += """
    #run_both_button button {
        background-color: #FF0000 !important;
        color: #FFFFFF !important;
    }
    """
    
    # Adjust the galleries to display images
    output_gallery_without = gr.Gallery(label="Generated Images without NegToMe", elem_id="output_gallery_without", columns=4)
    output_gallery_with = gr.Gallery(label="Generated Images with NegToMe", elem_id="output_gallery_with", columns=4)

    def submit_fn_without(
        prompt,
        merging_alpha,
        merging_threshold,
        merging_t_start,
        merging_t_end,
        seed,
        num_inference_steps,
        num_images_per_prompt,
        neg_prompt,
    ):
        images = generate_images(
            prompt,
            merging_alpha,
            merging_threshold,
            merging_t_start,
            merging_t_end,
            seed,
            num_inference_steps,
            num_images_per_prompt,
            use_negtome=False,
            neg_prompt=neg_prompt,
        )
        return images

    def submit_fn_with(
        prompt,
        merging_alpha,
        merging_threshold,
        merging_t_start,
        merging_t_end,
        seed,
        num_inference_steps,
        num_images_per_prompt,
        neg_prompt,
    ):
        images = generate_images(
            prompt,
            merging_alpha,
            merging_threshold,
            merging_t_start,
            merging_t_end,
            seed,
            num_inference_steps,
            num_images_per_prompt,
            use_negtome=True,
            neg_prompt=neg_prompt,
        )
        return images

    inputs = [
        prompt,
        merging_alpha,
        merging_threshold,
        merging_t_start,
        merging_t_end,
        seed,
        num_inference_steps,
        num_images_per_prompt,
        neg_prompt,
    ]
    
    gen_without_button.click(
        fn=submit_fn_without,
        inputs=inputs,
        outputs=output_gallery_without
    )

    gen_with_button.click(
        fn=submit_fn_with,
        inputs=inputs,
        outputs=output_gallery_with
    )

    # Use .then() to chain the functions for sequential execution
    run_both_button.click(
        fn=submit_fn_without,
        inputs=inputs,
        outputs=output_gallery_without
    ).then(
        fn=submit_fn_with,
        inputs=inputs,
        outputs=output_gallery_with
    )

    # Examples
    examples = [
        ["a hyper-realistic digital painting of a woman", 1.9],
        ["a high resolution photo of a child", 1.9],
        ["a hyper-realistic digital painting of a dragon", 1.9],
        ["a watercolor illustration of a animal", 1.9],
        ["a hyper-realistic digital painting of a building", .9],
        ["a 3D animation of a cute cat", 1.5],
        ["a hyper-realistic digital painting of a dress", 1.5],
        ["A panda leaping up to catch a dumpling mid-air with its paws, surrounded by bamboo and a few steaming baskets of dumplings  on the ground", 1.5],
        ["A majestic unicorn with a shimmering golden horn, galloping across a crystal-clear lake reflecting a vibrant sunset, surrounded by glowing fireflies and blossoming lotus flowers, while distant mountains fade into mist.", 1.5],
        ["A wise old wizard with a long flowing beard and a star-studded robe, holding a glowing staff topped with a crystal orb, standing on a cliff edge overlooking a stormy sea, with runes glowing faintly in the air around him.", 1.9],
        ["A powerful wizard standing atop a mountain peak, their hands raised as they summon a thunderstorm, with crackling lightning bolts shooting from their fingertips and dark clouds swirling ominously overhead.", 1.5],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[prompt, merging_alpha],
    )
    
# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()