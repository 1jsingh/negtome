import fire
import torch
import pickle
import numpy as np
import tqdm
from PIL import Image
from dreamsim import dreamsim
import t2v_metrics
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
import torch.nn as nn
import torch.nn.functional as F
import clip
import pytorch_lightning as pl
import os
import sys
src_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(src_folder))
from src.utils import patch_file_open, map2tmp, file_exists

# Helper function to load pickle data
def load_pickle_data(pkl_path):
    with patch_file_open(pkl_path, 'rb') as f:
        return pickle.load(f)

# Dreamsim Score calculation class
class DreamsimScore():
    def __init__(self, dreamsim_type="default", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        if dreamsim_type == "default":
            self.model, self.processor = dreamsim(pretrained=True, device=device)
        else:
            self.model, self.processor = dreamsim(pretrained=True, device=device, dreamsim_type=dreamsim_type)

    @torch.no_grad()
    def get_image_features(self, images_or_paths, norm=True):
        if not isinstance(images_or_paths, list):
            images_or_paths = [images_or_paths]
        if isinstance(images_or_paths[0], str):
            images = []
            for image_or_path in images_or_paths:
                img = Image.open(image_or_path).convert('RGB')
                images.append(img)
        else:
            images = images_or_paths
        inputs = torch.stack([self.processor(image).to(self.device).squeeze(0) for image in images])
        image_features = self.model.embed(inputs)
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

def get_im2img_sim(dreamsim_score, images):
    features = dreamsim_score.get_image_features(images)
    cosine_sim_matrix = torch.mm(features, features.T)
    lower_triangular = torch.tril(cosine_sim_matrix, diagonal=-1)
    num_pairs = (len(images) * (len(images) - 1)) / 2
    avg_similarity = lower_triangular.sum() / num_pairs
    return avg_similarity

# write the save as multiprocess
def save_image(img, idx):
    with patch_file_open(f"/tmp/{idx}.jpg", "wb") as f:
        img.save(f, format='JPEG')
    return f"/tmp/{idx}.jpg"

# VQAScore calculation
def compute_vqa_score(data):
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')
    img2prompt_sim = []
    total_entries = len(data['prompts'])
    for idx in tqdm.tqdm(range(total_entries), desc="Computing VQAScore"):
        prompt = data['prompts'][idx]
        images = data['images'][idx]
        img_paths = []
        
        import multiprocessing as mp
        with mp.Pool(mp.cpu_count()) as pool:
            img_paths = pool.starmap(save_image, [(img, idx) for idx, img in enumerate(images)])

        scores = clip_flant5_score(images=img_paths, texts=[prompt])
        img2prompt_sim.extend(scores.tolist())
    avg_vqa_score = np.mean(img2prompt_sim)
    return avg_vqa_score

def load_clip_model():
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def calculate_clip_score(model, processor, img, text):
    inputs = processor(text=text, images=img, return_tensors="pt", padding=True, max_length=77, truncation=True)
    inputs = {name: tensor.to(device=model.device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    # get the average value 
    score = logits_per_image.mean().item()
    return round(float(score), 4)

def clip_score(images, prompts):
    all_scores = []
    from functools import partial
    from tqdm import tqdm

    model, processor = load_clip_model()
    clip_score_fn = partial(calculate_clip_score, model, processor)
    for prompt, image in tqdm(zip(prompts, images)):
        # print(prompt)
        score = clip_score_fn(image, [prompt]*len(image))
        all_scores.append(score)
    return np.mean(all_scores)

# Aesthetic Score calculation
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def compute_aes_score(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(768)
    # Load the pretrained aesthetic model (update the path accordingly)
    s = torch.load("src/copycat/eval/models/sac+logos+ava1-l14-linearMSE.pth")
    model.load_state_dict(s)
    model.to(device)
    model.eval()
    aes_model, preprocess = clip.load("ViT-L/14", device=device)
    aes_scores = []
    for images in tqdm.tqdm(data['images'], desc="Computing Aesthetic Scores"):
        for img in images:
            image = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = aes_model.encode_image(image)
                im_emb_arr = normalized(image_features.cpu().detach().numpy())
                prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
            aes_scores.append(prediction.item())
    avg_aes_score = np.mean(aes_scores)
    return avg_aes_score

# Inception Score calculation
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size, f"Batch size {batch_size} should be less than the number of images {N}"
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))
    for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Computing Inception Score")):
        batch = batch.type(dtype)
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [entropy(pyx, py) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))
    return np.sum(split_scores), np.std(split_scores)

class PILImageDataset(Dataset):
    def __init__(self, pil_images, transform=None):
        self.pil_images = pil_images
        self.transform = transform

    def __getitem__(self, index):
        img = self.pil_images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.pil_images)

def compute_inception_score(data):
    pil_images = [img for images in data['images'] for img in images]
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = PILImageDataset(pil_images, transform=transform)
    inc_score, inc_std = inception_score(dataset, cuda=torch.cuda.is_available(), batch_size=32, resize=False, splits=10)
    return inc_score, inc_std

# Main function to compute selected metrics
def compute_metrics(data_path, scores='all'):
    """
    Compute selected metrics for images in the provided data path.

    Args:
        data_path (str): Path to the .pkl data file.
        scores (str or list): Metrics to compute. Options are 'all', or any combination of
                              ['dreamsim', 'vqa', 'aes', 'inception'].
    """
    # Load data
    data = load_pickle_data(data_path)

    # Parse the scores argument
    if scores == 'all':
        scores_to_compute = ['dreamsim', 'vqa', 'aes', 'inception']
    elif isinstance(scores, str):
        scores_to_compute = [scores.lower()]
    elif isinstance(scores, list):
        scores_to_compute = [s.lower() for s in scores]
    else:
        raise ValueError("Invalid value for 'scores' argument.")

    # Initialize results dictionary
    results = {}

    # Compute Dreamsim Score
    if 'dreamsim' in scores_to_compute:
        dreamsim_score_obj = DreamsimScore(device="cuda" if torch.cuda.is_available() else "cpu")
        img2img_sim = []
        for images in tqdm.tqdm(data['images'], desc="Computing Dreamsim Scores"):
            sim = get_im2img_sim(dreamsim_score_obj, images).item()
            img2img_sim.append(sim)
        avg_dreamsim_score = np.mean(img2img_sim)
        results['Dreamsim Score'] = avg_dreamsim_score

    # Compute VQAScore
    if 'vqa' in scores_to_compute:
        avg_vqa_score = compute_vqa_score(data)
        results['VQAScore'] = avg_vqa_score

    # Compute Aesthetic Score
    if 'aes' in scores_to_compute:
        avg_aes_score = compute_aes_score(data)
        results['Aesthetic Score'] = avg_aes_score

    # Compute Inception Score
    if 'inception' in scores_to_compute:
        inc_score, inc_std = compute_inception_score(data)
        results['Inception Score'] = f"{inc_score} ± {inc_std}"

    # Display results
    print(f"\nMetrics for {data_path}:")
    for metric, value in results.items():
        print(f"{metric}: {value}")


def load_gen_images(gen_folder='copycat/output_diversity', model_id="SG161222/RealVisXL_V4.0", baseline=True, categories=['bird', 'mammal', 'animal', 'boat', 'building', 'bus', 'car', 'airplane', 'fish', 'insect/bug', 'dog', 'cat',  'dragon', 'bridge', 'person', 'woman', 'man', 'child','shirt', 'dress'], guidance_scale=5.0, merge_categories=False, tau=0.7, t_start=1000, t_end=900, cads=False):
    from collections import defaultdict
    from src.utils import get_list_of_files_to_prepare
    import re
    import pickle
    gen_folder = os.path.join(gen_folder, model_id.replace("/", "_"))

    file_list = get_list_of_files_to_prepare(gen_folder)
    # print(file_list)
    # search for output files following the pattern using regex
    # define the regex pattern using guidance scale
    if baseline:
        # baseline
        pattern = re.compile(fr"guidance{guidance_scale}_seed(\d+)_output.pkl")
    elif cads:
        pattern = re.compile(fr"cads_tau1(\d+).(\d+)_tau2(\d+).(\d+)_noise(\d+).(\d+)_mix(\d+).(\d+)_rescaleTrue_guidance{guidance_scale}_seed(\d+)_output.pkl")
    else:
        # alpha and t can be float
        pattern = re.compile(fr"alpha(\d+).(\d+)_t{tau}_start{t_start}_end{t_end}_guidance{guidance_scale}_seed(\d+)_output.pkl")
    
    gen_images = defaultdict(list)
    merged_data = defaultdict(list)
    max_dreamsim_scores = defaultdict(list)
    average_dreamsim_scores = defaultdict(list)
    entropy_scores = defaultdict(lambda: defaultdict(list))

    for file in file_list:
        filename = os.path.basename(file)
        match = pattern.match(filename) # exact match
        if match and filename.endswith(".pkl"):
            print(f"Processing {filename}")
            output_filename = filename
            with patch_file_open(file, "rb") as f:
                loaded_data = pickle.load(f)
            prompts = loaded_data["prompts"]
            images = loaded_data["images"]

            if len(prompts) != len(images):
                num_prompts_per_image_row = len(images[0])
                # images = images[len(prompts)//num_prompts_per_image_row]
                num_prompts_per_image_row_last = len(images[-1])
                if num_prompts_per_image_row_last != num_prompts_per_image_row:
                    # number of rows mismatch
                    number_of_rows = (len(images) - len(prompts)//num_prompts_per_image_row) * (num_prompts_per_image_row //num_prompts_per_image_row_last)
                    # merge the last few rows, so that each row has the number of images as num_prompts_per_image_row
                    first_row_images = images[:len(images) - number_of_rows]
                    last_row_images = images[len(images) - number_of_rows:]
                    merged_images = []
                    for i in range(0, len(last_row_images), num_prompts_per_image_row_last):
                        merged_row = []
                        for j in range(i, i+num_prompts_per_image_row_last):
                            merged_row.extend(last_row_images[j])
                        merged_images.append(merged_row)

                    images = first_row_images + merged_images
                assert len(prompts) == num_prompts_per_image_row * len(images), "Number of prompts and images mismatch, {} != {}".format(len(prompts), num_prompts_per_image_row*len(images))
                #flatten the images
                flatten_images = [[img] for imgs in images for img in imgs]
                images = flatten_images
            
            for prompt, image in zip(prompts, images):
                # check if the prompt contains the category
                if merge_categories:
                    for category in categories:
                        if category in prompt:
                            gen_images[category].extend(image)
                            break
                else:
                    # if "dragon" not in prompt:
                    gen_images[prompt].extend(image)
            
            if "test_dreamsim_scores" in loaded_data:
                dreamsim_scores = loaded_data["test_dreamsim_scores"]
                for item in dreamsim_scores:
                    print(item)
                    for i, score in enumerate(item):
                        max_dreamsim_scores[i].append(score["max_score"])
                        average_dreamsim_scores[i].append(score["average_score"])
                for prompt, score_item in zip(prompts, dreamsim_scores):
                    for i, score in enumerate(score_item):
                        entropy_scores[i][prompt].append(score["character"])
            
                
    for prompt, images in gen_images.items():
        merged_data["prompts"].append(prompt)
        merged_data["images"].append(images)
    if not merge_categories:
        gen_images = merged_data 

    if "test_dreamsim_scores" in loaded_data:
        for i in max_dreamsim_scores.keys():
            print(f"Max Dreamsim Score {i}: {np.mean(max_dreamsim_scores[i])}")
            print(f"Average Dreamsim Score {i}: {np.mean(average_dreamsim_scores[i])}")
            entropy = 0
            for prompt, characters in entropy_scores[i].items():                
                characters_assets = [
                    'Ariel',
                    'Astro Boy',
                    'Batman',
                    'Black Panther',
                    'Bulbasaur',
                    'Buzz Lightyear',
                    'Captain America',
                    'Chun-Li',
                    'Cinderella',
                    'CupHead',
                    'Donald Duck',
                    'Doraemon',
                    'Elsa',
                    'Goofy',
                    'Groot',
                    'Hulk',
                    'Iron Man',
                    'Judy Hopps',
                    'Kirby',
                    'Kung Fu Panda',
                    'Lightning McQueen',
                    'Link',
                    'Maleficent',
                    'Mario',
                    'Mickey Mouse',
                    'Mike Wazowski',
                    'Monkey D. Luffy',
                    'Mr. Incredible',
                    'Naruto',
                    'Nemo',
                    'Olaf',
                    'Pac-Man',
                    'Peter Pan',
                    'Piglet',
                    'Pikachu',
                    'Princess Jasmine',
                    'Puss in boots',
                    'Rapunzel',
                    'Snow White',
                    'Sonic The Hedgehog',
                    'Spider-Man',
                    'SpongeBob SquarePants',
                    'Squirtle',
                    'Thanos',
                    'Thor',
                    'Tinker Bell',
                    'Wall-E',
                    'Winnie-the-Pooh',
                    'Woody',
                    'Yoda',
                    'Superman',
                    'Jack Sparrow']
                # count the characters
                char_count = [0] * len(characters_assets)
                for char in characters:
                    char_count[characters_assets.index(char)] += 1
                # calculate the entropy
                curr_entropy = 0
                for count in char_count:
                    if count == 0:
                        continue
                    prob = count / sum(char_count)
                    curr_entropy -= prob * np.log(prob)
                entropy += curr_entropy
            print(f"Entropy {i}: {entropy}") 
    return gen_images, output_filename


def compute_metrics_output_diversity(gen_folder, model_id="SG161222/RealVisXL_V4.0", guidance_scale=5.0, baseline=False, scores='all', tau=0.7, t_start=1000, t_end=900, cads=False):
    """
    Compute selected metrics for images in the provided data path.

    Args:
        data_path (str): Path to the .pkl data file.
        scores (str or list): Metrics to compute. Options are 'all', or any combination of
                              ['dreamsim', 'vqa', 'aes', 'inception'].
    """
    # Load data
    # data = load_pickle_data(data_path)

    # Parse the scores argument
    if scores == 'all':
        scores_to_compute = [
            # 'dreamsim',                
            'vqa', 
            # 'aes', 
            # 'inception',
            # 'clip'
            ]
    elif isinstance(scores, str):
        scores_to_compute = [scores.lower()]
    elif isinstance(scores, list):
        scores_to_compute = [s.lower() for s in scores]
    else:
        raise ValueError("Invalid value for 'scores' argument.")

    # Initialize results dictionary
    results = {}

    # Compute Dreamsim Score

    data, output_filename = load_gen_images(
        model_id=model_id, gen_folder=gen_folder, guidance_scale=guidance_scale,
        baseline=baseline, merge_categories=True, tau=tau, t_start=t_start, t_end=t_end, cads=cads)
    if 'dreamsim' in scores_to_compute:
        dreamsim_score_obj = DreamsimScore(device="cuda" if torch.cuda.is_available() else "cpu")
        img2img_sim = []
        for _, images in tqdm.tqdm(data.items(), desc="Computing Dreamsim Scores"):
            sim = get_im2img_sim(dreamsim_score_obj, images).item()
            img2img_sim.append(sim)
        avg_dreamsim_score = np.mean(img2img_sim)
        results['Dreamsim Score'] = avg_dreamsim_score
        print(f"Dreamsim Score: {avg_dreamsim_score}")
    
    data, _ = load_gen_images(
        model_id=model_id, gen_folder=gen_folder, guidance_scale=guidance_scale,
        baseline=baseline, merge_categories=False, tau=tau, t_start=t_start, t_end=t_end, cads=cads)

    # Compute VQAScore
    if 'vqa' in scores_to_compute:
        avg_vqa_score = compute_vqa_score(data)
        results['VQAScore'] = avg_vqa_score
        print(f"VQAScore: {avg_vqa_score}")

    # Compute Aesthetic Score
    if 'aes' in scores_to_compute:
        avg_aes_score = compute_aes_score(data)
        results['Aesthetic Score'] = avg_aes_score
        print(f"Aesthetic Score: {avg_aes_score}")

    # Compute Inception Score
    if 'inception' in scores_to_compute:
        inc_score, inc_std = compute_inception_score(data)
        results['Inception Score'] = f"{inc_score} ± {inc_std}"
        print(f"Inception Score: {inc_score} ± {inc_std}")

    if 'clip' in scores_to_compute:
        c_score = clip_score(data["images"], data["prompts"])
        results['CLIP Score'] = c_score
        print(f"CLIP Score: {c_score}")

    # Display results
    print(f"\nMetrics for {output_filename}:")
    import json
    with patch_file_open(os.path.join(gen_folder, model_id.replace("/", "_"), output_filename+"_quat_eval.json"), "w") as f:
        json.dump(results, f)

    for metric, value in results.items():
        print(f"{metric}: {value}")


def compute_metrics_copyright_mitigation(gen_folder, model_id="SG161222/RealVisXL_V4.0", guidance_scale=5.0, baseline=False, scores='all'):
    """
    Compute selected metrics for images in the provided data path.

    Args:
        data_path (str): Path to the .pkl data file.
        scores (str or list): Metrics to compute. Options are 'all', or any combination of
                              ['dreamsim', 'vqa', 'aes', 'inception'].
    """
    # Load data
    # data = load_pickle_data(data_path)
    from collections import defaultdict
    # Parse the scores argument
    if scores == 'all':
        scores_to_compute = [              
            'vqa', 
            'aes',
            'inception',
            'clip']
    elif isinstance(scores, str):
        scores_to_compute = [scores.lower()]
    elif isinstance(scores, list):
        scores_to_compute = [s.lower() for s in scores]
    else:
        raise ValueError("Invalid value for 'scores' argument.")

    # Initialize results dictionary
    results = {}

    # Compute Dreamsim Score

    data, output_filename = load_gen_images(
        model_id=model_id, gen_folder=gen_folder, guidance_scale=guidance_scale,
        baseline=baseline, merge_categories=False)


    data_parts = {
        "part1": {"prompts": [], "images": []},
        "part2": {"prompts": [], "images": []},
        "part3": {"prompts": [], "images": []},
        "part4": {"prompts": [], "images": []}
    }
    for idx, (prompt, images) in enumerate(zip(data["prompts"], data["images"])):
        for part_idx in range(1, 5):
            data_parts[f"part{part_idx}"]["prompts"].append(prompt)
        curr_split = defaultdict(list)
        for img_idx, img in enumerate(images):
            curr_split[f"part{img_idx%4+1}"].append(img)
        for part_idx in range(1, 5):
            data_parts[f"part{part_idx}"]["images"].append(curr_split[f"part{part_idx}"])
    print(data_parts["part1"]["prompts"][:5])
    print(len(data_parts["part1"]["images"][0]))

    for part, data_part in data_parts.items():
        results[part] = {}
    # Compute VQAScore
        if 'vqa' in scores_to_compute:
            # split the images into 4 parts
            avg_vqa_score = compute_vqa_score(data_part)
            results[part]['VQAScore'] = avg_vqa_score
            print(f"VQAScore: {avg_vqa_score}")

        # Compute Aesthetic Score
        if 'aes' in scores_to_compute:
            avg_aes_score = compute_aes_score(data_part)
            results[part]['Aesthetic Score'] = avg_aes_score
            print(f"Aesthetic Score: {avg_aes_score}")

        # Compute Inception Score
        if 'inception' in scores_to_compute:
            inc_score, inc_std = compute_inception_score(data_part)
            results[part]['Inception Score'] = f"{inc_score} ± {inc_std}"
            print(f"Inception Score: {inc_score} ± {inc_std}")

        if 'clip' in scores_to_compute:
            c_score = clip_score(data_part["images"], data_part["prompts"])
            results[part]['CLIP Score'] = c_score
            print(f"CLIP Score: {c_score}")

    # Display results
    print(f"\nMetrics for {output_filename}:")
    import json
    with patch_file_open(os.path.join(gen_folder, model_id.replace("/", "_"), output_filename+"_is_quat_eval.json"), "w") as f:
        json.dump(results, f)

    print(results)



def create_and_push_to_hf(baseline, ours, otuput_dataset_name):
    import datasets
    from datasets import Dataset, Features
    data_dict = {'prompt': [], 'category': [], 'ours': [], 'baseline': []}
    for cat in baseline.keys():
        for b_item, o_item in tqdm.tqdm(zip(baseline[cat], ours[cat])):
            b_prompt, b_images = b_item
            o_prompt, o_images = o_item
            assert b_prompt == o_prompt, "Prompt mismatch, {} != {}".format(b_prompt, o_prompt)
            data_dict['prompt'].append(b_prompt)
            data_dict['category'].append(cat)
            data_dict['ours'].append(o_images)
            data_dict['baseline'].append(b_images)


    # Define the dataset features with Image type and string for character names
    features = Features({
        'baseline': [datasets.Image()],
        'ours': [datasets.Image()],
        'prompt': datasets.Value('string'),
        'category': datasets.Value('string'),
    })

    # Convert to Hugging Face dataset
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    with patch_file_open("aux_data/credentials/hf_token.txt") as f:
        hf_token = f.read().strip()
    hf_dataset.push_to_hub(
        f"copycat-project/{otuput_dataset_name}",
        create_pr=False,
        token=hf_token)
    return


def create_hf_dataset_for_qual_eval(gen_folder='copycat/output_diversity', model_id="SG161222/RealVisXL_V4.0"):
    output_dataset_name = "output_diversity"
    if model_id == "SG161222/RealVisXL_V4.0":
        output_dataset_name += '_SDXL'
        guidance_scales = [
            2.0, 3.0, 4.0, 
            5.0, 
            6.0, 7.0, 8.0]
        # 5.0, 
    else:
        output_dataset_name += '_FLUX'
        guidance_scales = [
            1.5, 2.5, 3.5]
    for guidance_scale in guidance_scales:
        output_dataset_name_final = output_dataset_name + f"_guidance{guidance_scale}"
        data, _ = load_gen_images(
            model_id=model_id, gen_folder=gen_folder, guidance_scale=guidance_scale,
            baseline=True, merge_categories=False)

        print(len(data["prompts"]))
        
        categories = ['bird', 'mammal', 'animal', 'boat', 'building', 'bus', 'car', 'airplane', 'fish', 'insect/bug', 'dog', 'cat',  'bridge', 'person', 'woman', 'man', 'child','shirt', 'dress', 'dragon']
        from collections import defaultdict
        merged_data = defaultdict(list)
        for prompt, images in tqdm.tqdm(zip(data["prompts"], data["images"])):
            for cat in categories:
                if cat in prompt:
                    # for img in images:
                    # concatenate every four images
                    for i in range(0, len(images), 4):
                        img = images[i:i+4]
                        merged_data[cat].append((prompt, img))
                    break
        import random
        random.seed(int(guidance_scale))

        target_number = 100
        target_number_per_category = (target_number // len(categories))+1
        final_data = defaultdict(list)
        random_select_index = defaultdict(list)
        for cat, data in merged_data.items():
            random_select_index[cat].extend(random.sample(range(len(data)), target_number_per_category))
        
        for cat, data in merged_data.items():
            for idx in random_select_index[cat]:
                final_data[cat].append(data[idx])
        

        data_ours, _ = load_gen_images(
            model_id=model_id, gen_folder=gen_folder, guidance_scale=guidance_scale,
            baseline=False, merge_categories=False)


        merged_data_ours = defaultdict(list)
        for prompt, images in tqdm.tqdm(zip(data_ours["prompts"], data_ours["images"])):
            for cat in categories:
                if cat in prompt:
                    # for img in images:
                    # concatenate every four images
                    for i in range(0, len(images), 4):
                        img = images[i:i+4]
                        merged_data_ours[cat].append((prompt, img))
                    break
        final_data_ours = defaultdict(list)
        for cat, ours_data in merged_data_ours.items():
            for idx in random_select_index[cat]:
                final_data_ours[cat].append(ours_data[idx])

        create_and_push_to_hf(final_data, final_data_ours, output_dataset_name_final)


def compute_metrics_output_diversity_long_prompts(gen_folder, model_id="models/black-forest-labs/FLUX.1-dev", guidance_scale=3.5, baseline=False, scores='all', group_prompts=1):
    """
    Compute selected metrics for images in the provided data path.

    Args:
        data_path (str): Path to the .pkl data file.
        scores (str or list): Metrics to compute. Options are 'all', or any combination of
                              ['dreamsim', 'vqa', 'aes', 'inception'].
    """
    # Load data
    # data = load_pickle_data(data_path)

    # Parse the scores argument
    if scores == 'all':
        scores_to_compute = [
            'dreamsim',                
            'vqa', 
            'aes',
            'inception']
    elif isinstance(scores, str):
        scores_to_compute = [scores.lower()]
    elif isinstance(scores, list):
        scores_to_compute = [s.lower() for s in scores]
    else:
        raise ValueError("Invalid value for 'scores' argument.")

    # Initialize results dictionary
    results = {}

    # Compute Dreamsim Score

    data, output_filename = load_gen_images(
        model_id=model_id, gen_folder=gen_folder, guidance_scale=guidance_scale,
        baseline=baseline, merge_categories=False)
    if 'dreamsim' in scores_to_compute:
        dreamsim_score_obj = DreamsimScore(device="cuda" if torch.cuda.is_available() else "cpu")
        img2img_sim = []
        # for _, images in tqdm.tqdm(data.items(), desc="Computing Dreamsim Scores"):
        # if needed group prompts together

        for idx in range(0, len(data["prompts"]), group_prompts):
            all_images = []
            for i in range(group_prompts):
                images = data["images"][idx+i]
                # images = images[:int(len(images)//2)] halfen the seeds
                # select the first two images per four images
                # images = images[::2]
                all_images.extend(images)
            sim = get_im2img_sim(dreamsim_score_obj, all_images).item()
            img2img_sim.append(sim)
        avg_dreamsim_score = np.mean(img2img_sim)
        results['Dreamsim Score'] = avg_dreamsim_score
        print(f"Dreamsim Score: {avg_dreamsim_score}")

    # Compute Aesthetic Score
    if 'aes' in scores_to_compute:
        avg_aes_score = compute_aes_score(data)
        results['Aesthetic Score'] = avg_aes_score
        print(f"Aesthetic Score: {avg_aes_score}")

    # Compute Inception Score
    if 'inception' in scores_to_compute:
        inc_score, inc_std = compute_inception_score(data)
        results['Inception Score'] = f"{inc_score} ± {inc_std}"
        print(f"Inception Score: {inc_score} ± {inc_std}")

    # Compute VQAScore
    if 'vqa' in scores_to_compute:
        avg_vqa_score = compute_vqa_score(data)
        results['VQAScore'] = avg_vqa_score
        print(f"VQAScore: {avg_vqa_score}")

    if 'clip' in scores_to_compute:
        c_score = clip_score(data["images"], data["prompts"])
        results['CLIP Score'] = c_score
        print(f"CLIP Score: {c_score}")

    # Display results
    print(f"\nMetrics for {output_filename}:")
    import json
    with patch_file_open(os.path.join(gen_folder, model_id.replace("/", "_"), output_filename+"_quat_eval.json"), "w") as f:
        json.dump(results, f)

    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    fire.Fire()