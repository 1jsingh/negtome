import gradio as gr
import torch
from src.negtome.pipeline_negtome_flux import FluxNegToMePipeline

##############################################
# Load the pipeline
##############################################
pipe = FluxNegToMePipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

##############################################
# generate_images function
##############################################
def generate_images(
    prompt,
    merging_alpha=0.9,
    merging_threshold=0.65,
    merging_t_start=1000,
    merging_t_end=900,
    num_joint_blocks=-1,
    num_single_blocks=-1,
    seed=0,
    num_inference_steps=25,
    num_images_per_prompt=4,
    height=768,
    width=768,
    use_negtome=False
):
    # Convert parameters to appropriate types
    merging_t_start = int(merging_t_start)
    merging_t_end = int(merging_t_end)
    num_joint_blocks = int(num_joint_blocks)
    num_single_blocks = int(num_single_blocks)
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
        'num_joint_blocks': num_joint_blocks, 
        'num_single_blocks': num_single_blocks,
    }
    
    # Set up the random generator with the provided seed
    generator = torch.Generator(pipe.device).manual_seed(seed)
    
    # Run inference
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
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
    gr.Markdown("# NegToMe Demo: Increasing Output Diversity (Flux) 🚀")
    gr.Markdown("🔥🔥 NegToMe helps significantly increase output diversity by guiding features of each image away from each other. 🔥🔥")
    gr.Markdown("🤗 [[Project Page](https://negtome.github.io/)] [[Paper](https://negtome.github.io/docs/negtome.pdf)]    [[GitHub](https://github.com/1jsingh/negtome)] 🤗")
    
    with gr.Row():
        # Adjust the width of the prompt textbox
        prompt = gr.Textbox(
            lines=2,
            label="Prompt",
            value="a high resolution photo of a person",
            elem_id="prompt_textbox"
        )
    
    # Apply custom CSS to adjust widths and sizes
    demo.css = """
    #prompt_textbox textarea {
        width: 100% !important;
    }
    """
    merging_alpha = gr.Slider(minimum=-1., maximum=3., step=0.1, value=.9, label="Merging Alpha (controls diversity: higher alpha pushes images further apart)")
    merging_threshold = gr.Slider(minimum=0.5, maximum=1, step=0.05, value=0.65, label="Merging Threshold (controls which features are pushed apart: higher threshold preserves original features more)")


    with gr.Accordion("Advanced Settings", open=False):
        merging_t_start = gr.Slider(minimum=950, maximum=1000, step=10., value=1000, label="Merging t_start")
        merging_t_end = gr.Slider(minimum=850, maximum=950, step=10., value=900, label="Merging t_end")
        num_joint_blocks= gr.Slider(minimum=-1, maximum=19, step=1, value=-1, label="Number of Joint Transformer Blocks")
        num_single_blocks= gr.Slider(minimum=-1, maximum=38, step=1, value=-1, label="Number of Single Transformer Blocks")
        seed = gr.Number(value=0, label="Seed", precision=0)
        num_inference_steps = gr.Slider(minimum=25, maximum=50, step=1, value=25, label="Number of Inference Steps")
        num_images_per_prompt = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Number of Images per Prompt")
    
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
        num_joint_blocks,
        num_single_blocks,
        seed,
        num_inference_steps,
        num_images_per_prompt,
    ):
        images = generate_images(
            prompt,
            merging_alpha,
            merging_threshold,
            merging_t_start,
            merging_t_end,
            num_joint_blocks,
            num_single_blocks,
            seed,
            num_inference_steps,
            num_images_per_prompt,
            use_negtome=False
        )
        return images

    def submit_fn_with(
        prompt,
        merging_alpha,
        merging_threshold,
        merging_t_start,
        merging_t_end,
        num_joint_blocks,
        num_single_blocks,
        seed,
        num_inference_steps,
        num_images_per_prompt,
    ):
        images = generate_images(
            prompt,
            merging_alpha,
            merging_threshold,
            merging_t_start,
            merging_t_end,
            num_joint_blocks,
            num_single_blocks,
            seed,
            num_inference_steps,
            num_images_per_prompt,
            use_negtome=True
        )
        return images

    inputs = [
        prompt,
        merging_alpha,
        merging_threshold,
        merging_t_start,
        merging_t_end,
        num_joint_blocks,
        num_single_blocks,
        seed,
        num_inference_steps,
        num_images_per_prompt,
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
        ["a high resolution photo of a person", 0.9],
        ["an artistic painting of a sunset over the mountains", 0.9],
        ["a high-resolution photo of a fish", 0.9],
        ["A wizard stepping through a swirling, glowing portal, his robes billowing around him as he holds a staff that hums with energy. The portal casts a soft, otherworldly light, illuminating his determined expression and the dark, ancient ruins surrounding him.", 0.9],
        ["a hyper-realistic digital painting of a child", 0.9],
        ["a black and white photo of a cat", 0.9],
        ["a tiger chasing a person in a forest", 0.9],
        ["a beautiful landscape painting of a waterfall", 0.9],
        ["A panda leaping up to catch a dumpling mid-air with its paws, surrounded by bamboo and a few steaming baskets of dumplings  on the ground", 0.9],
        ["A lively golden retriever chasing a bright blue butterfly through a vibrant garden filled with colorful flowers, with sunlight streaming through the leaves and a soft breeze rustling the petals", 0.9],
        ["An energetic orange tabby cat pouncing on a rolling blue ball in a cozy living room, with a soft rug beneath and sunlight streaming through the window, casting warm patterns on the floor.", 0.9],
        ["a watercolor illustration of a city skyline at night", 0.9],
        ["A towering elemental creature made of flowing water, in the middle of a serene meadow, its form constantly shifting and glistening under a soft, diffuse light. The surrounding grass gently bends as a light mist rises around it.", 0.9],


    ]
    
    gr.Examples(
        examples=examples,
        inputs=[prompt, merging_alpha],
    )
    
# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()