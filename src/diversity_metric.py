import os
import sys
import torch
from transformers import CLIPProcessor, CLIPModel
src_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(src_folder))
from src.utils import patch_file_open, file_exists
import json


def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def calc_fid(gen_folder='copycat/output_diversity', model_id="SG161222/RealVisXL_V4.0", baseline=False, categories=['bird', 'mammal', 'animal', 'boat', 'building', 'bus', 'car', 'airplane', 'fish', 'insect/bug', 'dog', 'cat',  'dragon', 'bridge', 'person', 'woman', 'man', 'child','shirt', 'dress'], guidance_scale=5.0, cads=False):
    from datasets import load_dataset
    from collections import defaultdict
    all_real_images = defaultdict(list)
    for category in categories:
        real_image_dataset = f"laion2b6plus_{category.replace('/', '_')}"
        # load the real image dataset from hf
        real_image_dataset = load_dataset(os.path.join('copycat-project', real_image_dataset))
        real_images = real_image_dataset['train']['image']
        all_real_images[category] = real_images
    
    print(len(all_real_images))
    from src.utils import get_list_of_files_to_prepare
    import re
    import pickle
    gen_folder = os.path.join(gen_folder, model_id.replace("/", "_"))

    file_list = get_list_of_files_to_prepare(gen_folder)
    print(file_list)
    # search for output files following the pattern using regex
    if baseline:
        # baseline
        pattern = re.compile(fr"guidance{guidance_scale}_seed(\d+)_output.pkl")
    elif cads:
        pattern = re.compile(fr"cads_tau1(\d+).(\d+)_tau2(\d+).(\d+)_noise(\d+).(\d+)_mix(\d+).(\d+)_rescale(\d+).(\d+)_guidance{guidance_scale}_seed(\d+)_output.pkl")
    else:
        # alpha and t can be float
        pattern = re.compile(fr"alpha(\d+).(\d+)_t(\d+).(\d+)_start(\d+)_end(\d+)_guidance{guidance_scale}_seed(\d+)_output.pkl")
    
    gen_images = defaultdict(list)
    all_gen_count = 0
    for file in file_list:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if match and filename.endswith(".pkl"):
            with patch_file_open(file, "rb") as f:
                loaded_data = pickle.load(f)
            print(f"Processing {file}")
            prompts = loaded_data["prompts"]
            images = loaded_data["images"]
            for prompt, image in zip(prompts, images):
                # check if the prompt contains the category
                for category in categories:
                    if category in prompt:
                        gen_images[category].extend(image)
                        break
                all_gen_count += len(image)
    assert all_gen_count > 0, "No generated images found"

    import random
    import numpy as np
    random.seed(0)
    real_images = []
    fake_images = []
    for category in categories:
        gen_image_len = len(gen_images[category])
        print(f"Category: {category}, Gen images: {gen_image_len}")
        # real_images.extend(random.sample(all_real_images[category], gen_image_len))
        real_images.extend(all_real_images[category])
        fake_images.extend(gen_images[category])
        # output_name = "baseline" if baseline else "ours"
        # visualize_to_html(gen_images[category], f"copycat/output_diversity/visualize/{output_name}_{category}_gen.html")

    
    random.shuffle(real_images)
    real_images = real_images[:int(len(fake_images)*2.5)]
    from torchvision.transforms import functional as F

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        # resize the image to shorter side 256
        if image.shape[2] < image.shape[3]:
            image = F.resize(image, (image.shape[2], 256))
        else:
            image = F.resize(image, (256, image.shape[3]))
        return F.center_crop(image, (256, 256))

    # randomly sample real images subset to match the number of generated images

    real_images_fid = torch.cat([preprocess_image(np.array(image.convert("RGB"))) for image in real_images])
    fake_images_fid = torch.cat([preprocess_image(np.array(image)) for image in fake_images])

    # calculate the FID using pytorch-fid
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images_fid, real=True)
    fid.update(fake_images_fid, real=False)

    print(f"FID: {float(fid.compute())}")

    # def preprocess_image_kid(image):
    #     image = torch.tensor(image).unsqueeze(0)
    #     image = image.permute(0, 3, 1, 2)
    #     # resize the image to shorter side 256
    #     if image.shape[2] < image.shape[3]:
    #         image = F.resize(image, (image.shape[2], 256))
    #     else:
    #         image = F.resize(image, (256, image.shape[3]))
    #     return F.center_crop(image, (256, 256))

    # real_images_kid = torch.cat([preprocess_image_kid(np.array(image.convert("RGB"))) for image in real_images])
    # fake_images_kid = torch.cat([preprocess_image_kid(np.array(image)) for image in fake_images])

    # from torchmetrics.image.kid import KernelInceptionDistance
    # kid = KernelInceptionDistance(subset_size=50)
    # kid.update(real_images_kid, real=True)
    # kid.update(fake_images_kid, real=False)
    # print(f"KID: {kid.compute()}")


def calc_fid_coco_val_100(gen_folder='copycat/rebuttal', model_id="SG161222/RealVisXL_V4.0", baseline=False, guidance_scale=5.0, cads=False, real_image_dataset=f"coco_val_100"):
    from datasets import load_dataset
    from collections import defaultdict
    all_real_images = []
    
    # load the real image dataset from hf
    real_image_dataset = load_dataset(os.path.join('copycat-project', real_image_dataset))
    real_images = real_image_dataset['train']['image']
    all_real_images = real_images
    
    print(len(all_real_images))
    from src.utils import get_list_of_files_to_prepare
    import re
    import pickle
    gen_folder = os.path.join(gen_folder, model_id.replace("/", "_"))

    file_list = get_list_of_files_to_prepare(gen_folder)
    print(file_list)
    # search for output files following the pattern using regex
    if baseline:
        # baseline
        pattern = re.compile(fr"guidance{guidance_scale}_seed(\d+)_output.pkl")
    elif cads:
        pattern = re.compile(fr"cads_tau(\d+).(\d+)_tau(\d+).(\d+)_noise(\d+).(\d+)_mix(\d+).(\d+)_rescaleTrue_guidance{guidance_scale}_seed(\d+)_output.pkl")
    else:
        # alpha and t can be float
        pattern = re.compile(fr"alpha(\d+).(\d+)_t(\d+).(\d+)_start(\d+)_end(\d+)_guidance{guidance_scale}_seed(\d+)_output.pkl")
    
    gen_images = list()
    all_gen_count = 0
    for file in file_list:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if match and filename.endswith(".pkl"):
            with patch_file_open(file, "rb") as f:
                loaded_data = pickle.load(f)
            print(f"Processing {file}")
            prompts = loaded_data["prompts"]
            images = loaded_data["images"]
            for prompt, image in zip(prompts, images):
                # check if the prompt contains the category
                gen_images.extend(image)
                all_gen_count += len(image)
    assert all_gen_count > 0, "No generated images found"

    import random
    import numpy as np
    random.seed(0)
    real_images = []
    fake_images = []

    gen_image_len = len(gen_images)
    print(f"Gen images: {gen_image_len}")
    # real_images.extend(random.sample(all_real_images[category], gen_image_len))
    real_images.extend(all_real_images)
    fake_images.extend(gen_images)
        # output_name = "baseline" if baseline else "ours"
        # visualize_to_html(gen_images[category], f"copycat/output_diversity/visualize/{output_name}_{category}_gen.html")

    
    # random.shuffle(real_images)
    # real_images = real_images
    from torchvision.transforms import functional as F

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        # resize the image to shorter side 256
        if image.shape[2] < image.shape[3]:
            image = F.resize(image, (image.shape[2], 256))
        else:
            image = F.resize(image, (256, image.shape[3]))
        return F.center_crop(image, (256, 256))

    # randomly sample real images subset to match the number of generated images

    real_images_fid = torch.cat([preprocess_image(np.array(image.convert("RGB"))) for image in real_images])
    fake_images_fid = torch.cat([preprocess_image(np.array(image)) for image in fake_images])

    # calculate the FID using pytorch-fid
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images_fid, real=True)
    fid.update(fake_images_fid, real=False)

    print(f"FID: {float(fid.compute())}")


def calc_precision_recall(gen_folder='copycat/rebuttal', model_id="SG161222/RealVisXL_V4.0", baseline=False, guidance_scale=5.0, cads=False, real_image_dataset=f"coco_val_100"):
    from datasets import load_dataset
    from collections import defaultdict
    all_real_images = []
    # real_image_dataset = f"coco_val_100"
    # load the real image dataset from hf
    real_image_dataset = load_dataset(os.path.join('copycat-project', real_image_dataset))
    real_images = real_image_dataset['train']['image']
    real_image_ids = real_image_dataset['train']['image_id']
    all_real_images = real_images
    
    print(len(all_real_images))
    from src.utils import get_list_of_files_to_prepare
    import re
    import pickle
    gen_folder = os.path.join(gen_folder, model_id.replace("/", "_"))

    file_list = get_list_of_files_to_prepare(gen_folder)
    print(file_list)
    # search for output files following the pattern using regex
    if baseline:
        # baseline
        pattern = re.compile(fr"guidance{guidance_scale}_seed(\d+)_output.pkl")
    elif cads:
        pattern = re.compile(fr"cads_tau(\d+).(\d+)_tau(\d+).(\d+)_noise(\d+).(\d+)_mix(\d+).(\d+)_rescaleTrue_guidance{guidance_scale}_seed(\d+)_output.pkl")
    else:
        # alpha and t can be float
        pattern = re.compile(fr"alpha(\d+).(\d+)_t(\d+).(\d+)_start(\d+)_end(\d+)_guidance{guidance_scale}_seed(\d+)_output.pkl")
    
    prompts = []
    with patch_file_open("prompts/coco_val_100.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    
    prompt_image_ids = defaultdict(list)
    for i, prompt in enumerate(prompts):
        for j, image_id in enumerate(real_image_ids):
            if prompt in image_id:
                prompt_image_ids[prompt].append(image_id)

    gen_images = defaultdict(list)

    all_gen_count = 0
    for file in file_list:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if match and filename.endswith(".pkl"):
            with patch_file_open(file, "rb") as f:
                loaded_data = pickle.load(f)
            print(f"Processing {file}")
            prompts = loaded_data["prompts"]
            images = loaded_data["images"]
            for prompt, image in zip(prompts, images):
                # check if the prompt contains the category
                img_id = prompt_image_ids[prompt]
                gen_images[img_id].extend(image)
                all_gen_count += len(image)
    assert all_gen_count > 0, "No generated images found"

    import random
    import numpy as np
    random.seed(0)
    real_images = []
    fake_images = []

    gen_image_len = len(gen_images)
    print(f"Gen images: {gen_image_len}")
    # real_images.extend(random.sample(all_real_images[category], gen_image_len))
    real_images.extend(all_real_images)
    fake_images.extend(gen_images)
        # output_name = "baseline" if baseline else "ours"
        # visualize_to_html(gen_images[category], f"copycat/output_diversity/visualize/{output_name}_{category}_gen.html")

    
    # random.shuffle(real_images)
    # real_images = real_images
    from torchvision.transforms import functional as F

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        # resize the image to shorter side 256
        if image.shape[2] < image.shape[3]:
            image = F.resize(image, (image.shape[2], 256))
        else:
            image = F.resize(image, (256, image.shape[3]))
        return F.center_crop(image, (256, 256))

    # randomly sample real images subset to match the number of generated images

    real_images_fid = torch.cat([preprocess_image(np.array(image.convert("RGB"))) for image in real_images])
    fake_images_fid = torch.cat([preprocess_image(np.array(image)) for image in fake_images])

    # calculate the FID using pytorch-fid
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images_fid, real=True)
    fid.update(fake_images_fid, real=False)

    print(f"FID: {float(fid.compute())}")


def von_neumann_entropy(rho):
    """
    Compute the von Neumann entropy of a density matrix rho.
    
    Parameters
    ----------
    rho : np.ndarray
        A density matrix (square, Hermitian, positive semi-definite, trace=1).
    base : float, optional
        The logarithm base. By default, uses the natural base (e).
        Use base=2 for bits, for example.
    
    Returns
    -------
    float
        The von Neumann entropy of rho.
    """
    import numpy as np
    # base=np.e
    print(rho.shape)
    print(np.min(rho), np.max(rho))
    # Ensure rho is a NumPy array
    rho = np.asarray(rho, dtype=complex)
    
    # Diagonalize the density matrix to get eigenvalues
    eigenvalues, _ = np.linalg.eigh(rho)
    
    # Filter out (approximately) zero eigenvalues to avoid log(0)
    # This threshold can be adjusted based on numerical precision needs
    eps = 1e-12
    eigenvalues = eigenvalues[eigenvalues > eps]
    
    # Compute the von Neumann entropy
    # S = - sum(lambda_i log(lambda_i))
    # Use change-of-base formula: log_{base}(x) = log(x) / log(base)
    log_vals = np.log(eigenvalues) # / np.log(base)
    entropy = - np.sum(eigenvalues * log_vals)

    entropy = np.exp(entropy)
    
    return entropy


def calc_cads_metric_coco(gen_folder='copycat/rebuttal', model_id="SG161222/RealVisXL_V4.0", baseline=False, guidance_scale=5.0, cads=False, feature_extractor="dreamsim", real_image_dataset=f"coco_val_100"):
    from datasets import load_dataset
    from collections import defaultdict
    all_real_images = []
    # load the real image dataset from hf
    real_image_dataset = load_dataset(os.path.join('copycat-project', real_image_dataset))
    real_images = real_image_dataset['train']['image']
    real_image_ids = real_image_dataset['train']['image_id']
    all_real_images = real_images
    
    print(len(all_real_images))
    from src.copycat.utils import get_list_of_files_to_prepare
    import re
    import pickle
    gen_folder = os.path.join(gen_folder, model_id.replace("/", "_"))

    file_list = get_list_of_files_to_prepare(gen_folder)
    print(file_list)
    # search for output files following the pattern using regex
    if baseline:
        # baseline
        pattern = re.compile(fr"guidance{guidance_scale}_seed(\d+)_output.pkl")
    elif cads:
        pattern = re.compile(fr"cads_tau(\d+).(\d+)_tau(\d+).(\d+)_noise(\d+).(\d+)_mix(\d+).(\d+)_rescaleTrue_guidance{guidance_scale}_seed(\d+)_output.pkl")
    else:
        # alpha and t can be float
        pattern = re.compile(fr"alpha(\d+).(\d+)_t(\d+).(\d+)_start(\d+)_end(\d+)_guidance{guidance_scale}_seed(\d+)_output.pkl")
    
    prompts = []
    with patch_file_open(f"prompts/{real_image_dataset}.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    
    prompt_image_ids = defaultdict(str)
    for i, prompt in enumerate(prompts):
        prompt_image_ids[prompt] = real_image_ids[i]
    
    print(prompt_image_ids)

    gen_images = defaultdict(list)

    all_gen_count = 0
    for file in file_list:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if match and filename.endswith(".pkl"):
            with patch_file_open(file, "rb") as f:
                loaded_data = pickle.load(f)
            print(f"Processing {file}")
            prompts = loaded_data["prompts"]
            images = loaded_data["images"]
            for img_id, prompt, image in zip(real_image_ids, prompts, images):
                gen_images[img_id].extend(image)
                all_gen_count += len(image)
    assert all_gen_count > 0, "No generated images found"

    # get dreamsim features for real images and generated images
    if feature_extractor == "dreamsim":
        from src.quant_eval import DreamsimScore
        dreamsim = DreamsimScore()
        real_features = dreamsim.get_image_features(all_real_images)
        for img_id, images in gen_images.items():
            gen_images[img_id] = dreamsim.get_image_features(images)
    elif feature_extractor == "SSCD":
        raise NotImplementedError("SSCD feature extractor not implemented yet")
    else:
        raise ValueError("Invalid feature extractor")

    # get SSCD features
    
    
    predicted_image_ids = []
    # build cluster, and predict real feature image id based on the most similar cluster
    for real_img_id, real_feature in zip(real_image_ids, real_features):
        max_score = 0
        max_img_id = None
        for img_id, gen_features in gen_images.items():
            real_feature_to_compare = real_feature.expand_as(gen_features).to(dreamsim.device)
            all_score = (real_feature_to_compare * gen_features.to(dreamsim.device)).sum(axis=-1).cpu()
            score = all_score.mean()
            if score > max_score:
                max_score = score
                max_img_id = img_id
        predicted_image_ids.append(max_img_id)
    print(real_image_ids)
    print(predicted_image_ids)
    # calculate precision and recall
    correct = 0
    for gt_img_id, predicted_img_id in zip(real_image_ids, predicted_image_ids):
        if gt_img_id == predicted_img_id:
            correct += 1
    recall = correct / len(predicted_image_ids)
    print(f"Recall: {recall}")

    predicted_generated_image_ids = []
    gt_gen_image_ids = []
    for img_id, gen_features in gen_images.items():
        for gen_feat in gen_features:
            max_score = 0
            max_img_id = None
            gen_feat_to_compare = gen_feat.expand_as(real_features).to(dreamsim.device)
            all_score = (real_features.to(dreamsim.device) * gen_feat_to_compare).sum(axis=-1).cpu()
            max_img_id = real_image_ids[all_score.argmax()]
            predicted_generated_image_ids.append(max_img_id)
            gt_gen_image_ids.append(img_id)
    correct = 0
    for gt_img_id, predicted_img_id in zip(gt_gen_image_ids, predicted_generated_image_ids):
        if gt_img_id == predicted_img_id:
            correct += 1
    precision = correct / len(predicted_generated_image_ids)
    print(f"Precision: {precision}")

    # calculate vendi score
    '''
    we first compute the pairwise cosine similarity matrix Ky
    among generated images with the same condition, using SSCD (Pizzi et al., 2022) as the pretrained
    feature extractor. The results are then aggregated for different conditions using two methods: the
    Mean Similarity Score (MSS), which is a simple average over the similarity matrix Ky , and the Vendi
    Score (Friedman & Dieng, 2022), which is based on the Von Neumann entropy of Ky .
    '''
    mss = 0
    vendi_score = 0
    for img_id, gen_features in gen_images.items():
        all_scores = gen_features @ gen_features.T
        all_scores = all_scores.cpu().numpy()
        print(all_scores.mean())
        mss += all_scores.mean()
        vendi_score += von_neumann_entropy(all_scores)
        
    mss /= len(gen_images)
    vendi_score /= len(gen_images)
    print(f"MSS: {mss}, Vendi Score: {vendi_score}")


def zeroshot_classifier(model, processor, classnames, templates):
    from tqdm import tqdm

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            # print(texts)
            texts = processor(texts, return_tensors="pt", padding=True, truncation=True).to(device=model.device)
            
            class_embeddings = model.get_text_features(**texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device=model.device)
    return zeroshot_weights


def is_animal(synset):
    from nltk.corpus import wordnet as wn
    animal_synset = wn.synset('animal.n.01')
    # Traverse all hypernyms recursively
    for hypernym in synset.closure(lambda s: s.hypernyms()):
        if hypernym == animal_synset:
            return True
    for category in ['fish', 'bird', 'mammal', 'reptile', 'amphibian']:
        if is_a_class(synset, category):
            return True
    return False


def is_a_class(synset, class_name):
    from nltk.corpus import wordnet as wn
    amphibian_synset = wn.synset(f'{class_name}.n.01')
    # Traverse all hypernyms recursively
    for hypernym in synset.closure(lambda s: s.hypernyms()):
        if hypernym == amphibian_synset:
            return True
    return False


def extract_synsets(file_path: str):
    """
    Extracts synset IDs and their corresponding class names from a file.

    Args:
        file_path (str): The path to the input file.

    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing:
            - Synset ID (str)
            - List of Class Names (List[str])
    """
    import re
    from nltk.corpus import wordnet as wn
    synsets = []
    synset_pattern = re.compile(r"'(n\d+)\s+([^']+)'")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            match = synset_pattern.match(line)
            if match:
                synset_id, class_names_str = match.groups()
                
                # Split class names by comma and strip whitespace
                class_names = [name.strip().lower().replace("''", "'") for name in class_names_str.split(',')]

                pos = synset_id[0]  # 'n' for noun
                offset = int(synset_id[1:])  # The numeric part
                
                synset = wn.synset_from_pos_and_offset(pos, offset)
                
                synsets.append((synset, class_names))
            else:
                print(f"Warning: Line {line_num} format is incorrect: {line}")
    return synsets


def filter_animal_nouns(word_list):
    output = []
    for (synset, word) in word_list:
        if synset and is_animal(synset):
            output.append(word)
    # Filter the list for animal-related nouns
    return output

def filter_by_class(word_list, class_name):
    output = []
    for (synset, word) in word_list:
        if synset and is_a_class(synset, class_name):
            output.append(word)
    # Filter the list for animal-related nouns
    return output


def filter_insect_nouns(word_list):
    output = []
    for (synset, word) in word_list:
        if synset and is_a_class(synset, 'insect'):
            output.append(word)
    return output

imagenet_classes = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]


imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def load_gen_images(gen_folder='copycat/output_diversity', model_id="SG161222/RealVisXL_V4.0", baseline=True, categories=['bird', 'mammal', 'animal', 'boat', 'building', 'bus', 'car', 'airplane', 'fish', 'insect/bug', 'dog', 'cat',  'dragon', 'bridge', 'person', 'woman', 'man', 'child','shirt', 'dress'], guidance_scale=5.0, save_to_img_file=False, load=True):
    from collections import defaultdict
    from src.copycat.utils import get_list_of_files_to_prepare
    import re
    import pickle
    gen_folder = os.path.join(gen_folder, model_id.replace("/", "_"))

    file_list = get_list_of_files_to_prepare(gen_folder)
    # print(file_list)
    # search for output files following the pattern using regex
    # define the regex pattern using guidance scale
    # guidance_scale = f"guidance{guidance_scale}"
    # _seed(\d+)_output.pkl"
    if baseline:
        # baseline
        pattern = re.compile(fr"guidance{guidance_scale}_seed(\d+)_output.pkl")
    else:
        # alpha and t can be float
        pattern = re.compile(fr"alpha(\d+).(\d+)_t(\d+).(\d+)_start(\d+)_end(\d+)_guidance{guidance_scale}_seed(\d+)_output.pkl")
    
    gen_images = defaultdict(list)
    found_match = False
    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = pattern.match(filename)
        if match and filename.endswith(".pkl"):
            found_match = True
            print(f"Processing {file_path}")
            output_filename = filename
            if load:
                with patch_file_open(file_path, "rb") as f:
                    loaded_data = pickle.load(f)
                prompts = loaded_data["prompts"]
                images = loaded_data["images"]
                for prompt, image in zip(prompts, images):
                    # check if the prompt contains the category
                    for category in categories:
                        if category in prompt:
                            gen_images[category].extend(image)
                            break
    assert found_match, f"No files found in {gen_folder} with guidance scale {guidance_scale}"
    if save_to_img_file:
        image_paths = []
        # remove seed from the filename
        # write the image path to a csv file
        output_filename = re.sub(r"_seed\d+", "", output_filename)
        for category, images in gen_images.items():
            output_folder = os.path.join(gen_folder, output_filename)
            for i, image in enumerate(images):
                image_path = os.path.join(output_folder, f"{category}_{i}.jpg")
                if not file_exists(image_path):
                    with patch_file_open(image_path, "wb") as f:
                        image.save(f, format="JPEG")
                image_paths.append(image_path)
        import pandas as pd
        output_csv_path = os.path.join(gen_folder, f"{output_filename}_image_paths.csv")
        with patch_file_open(output_csv_path, "w") as f:
            # save the image paths to a csv file, with header as "img_path"
            df = pd.DataFrame(image_paths, columns=["img_path"])
            df.to_csv(f, index=False)
        return output_csv_path
    return gen_images, output_filename


def load_synsets():
    synsets = extract_synsets("src/copycat/mitigation/imagenet_synet.txt")
    imagenet_classes2synset = []
    found_synsets = set()
    for class_ in imagenet_classes:
        for synset, class_names in synsets:
            if synset in found_synsets:
                continue
            if class_.lower().strip() in " ".join(class_names).lower():
                imagenet_classes2synset.append((synset, class_))
                found_synsets.add(synset)
                break
        # if not found:
        #     import ipdb; ipdb.set_trace()
    
    for synset, class_ in synsets:
        if synset not in found_synsets:
            imagenet_classes2synset.append((synset, " ".join(class_)))
    # add missing classes
    # for class_ in imagenet_classes:
        # print(f"Missing: {class_}")
    print(f"Total number of classes: {len(imagenet_classes2synset)}")
    return imagenet_classes2synset

def augment_with_wordnet_child_nodes(category):
    from nltk.corpus import wordnet as wn
    # Get the synsets for 'shirt'
    synsets = wn.synsets(category, pos=wn.NOUN)

    # Assuming we're interested in the most common sense
    if synsets:
        current_word = synsets[0]
        hyponyms = current_word.hyponyms()
        hyponym_names = [lemma.name().replace('_', ' ') for syn in hyponyms for lemma in syn.lemmas()]
        
        # print(f"Hyponyms of '{current_word}':")
        # for name in hyponym_names:
        #     print(f"- {name}")
        return hyponym_names
    else:
        print(f"No synsets found for '{current_word}'.")
        return []

def calc_entropy_per_imagenet_category(model, processor, imagenet_classes2synset, gen_images, main_category='animal'):
    # print("Total number of synsets: ", len(synsets))
    # print(synsets[:10])
    
    if main_category == 'animal':
        filtered_imagenet_classes = filter_animal_nouns(imagenet_classes2synset)
        print("After filtering, there are {} animal-related classes.".format(len(filtered_imagenet_classes)))
    elif main_category == 'insect/bug':
        filtered_imagenet_classes = filter_insect_nouns(imagenet_classes2synset)
        print("After filtering, there are {} insect-related classes.".format(len(filtered_imagenet_classes)))
    else:
        filtered_imagenet_classes = filter_by_class(imagenet_classes2synset, main_category)
        print("After filtering, there are {} {}-related classes.".format(len(filtered_imagenet_classes), main_category))
    
    if len(filtered_imagenet_classes) < 20:
        filtered_imagenet_classes.extend(augment_with_wordnet_child_nodes(main_category))
        print("After augmenting, there are {} {}-related classes.".format(len(filtered_imagenet_classes), main_category))

    filtered_imagenet_classes = list(set(filtered_imagenet_classes))
    print("After deduplication, there are {} {}-related classes.".format(len(filtered_imagenet_classes), main_category))
    
    fake_images = []
    categories = [main_category]
    for category in categories:
        gen_image_len = len(gen_images[category])
        print(f"Category: {category}, Gen images: {gen_image_len}")
        # real_images.extend(random.sample(all_real_images[category], gen_image_len))
        # real_images.extend(all_real_images[category])
        fake_images.extend(gen_images[category])
        # output_name = "baseline" if baseline else "ours"
        # visualize_to_html(gen_images[category], f"copycat/output_diversity/visualize/{output_name}_{category}_gen.html")
    zeroshot_weights = zeroshot_classifier(model, processor, filtered_imagenet_classes, imagenet_templates)


    count = torch.zeros(len(filtered_imagenet_classes))
    
    for image in fake_images:
        image = processor(images=image, return_tensors="pt").to(device=model.device)
        image_features = model.get_image_features(**image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights
        pred = logits.topk(1, 1, True, True)[1].t().item()
        # print(pred)
        count[pred] += 1
    
    # calculate entropy
    count = count / count.sum()
    # how to ignore 0s?
    count = count[count != 0]
    # print(count)
    entropy = -torch.sum(count * torch.log(count))
    # print(entropy.item())
    return entropy.item()


def calc_entropy_per_person_categories(gen_images, main_category='person'):
    fake_images = []
    categories = [main_category]
    for category in categories:
        gen_image_len = len(gen_images[category])
        print(f"Category: {category}, Gen images: {gen_image_len}")
        # real_images.extend(random.sample(all_real_images[category], gen_image_len))
        # real_images.extend(all_real_images[category])
        fake_images.extend(gen_images[category])
    

def calc_entropy_imagenet_categories(gen_folder, model_id="SG161222/RealVisXL_V4.0", guidance_scale=5.0, baseline=False):
    imagenet_classes2synset = load_synsets()
    from collections import defaultdict
    model, processor = load_clip_model()
    categories = [
        'animal', 'mammal', 'bird', 'fish', 'insect/bug', 'dog', 'cat', 
                  'car', 'bus', 'building', 'bridge', 'airplane', 'shirt', 'dress']
    for guidance_scale in [guidance_scale]:
    # for guidance_scale in range(2, 9):
        # guidance_scale = f"{guidance_scale}.0"
        per_category_entropy_baseline = defaultdict(float)
        per_category_entropy_ours = defaultdict(float)
        for b in [baseline]:
        # for b in [True, False]:
            print(f"Baseline: {b}")
        # for guidance_scale in [5]:
            gen_images, output_filename = load_gen_images(gen_folder, model_id, guidance_scale=guidance_scale, baseline=b, load=True)

            if b:
                output_path = os.path.join(
                    gen_folder,
                    model_id.replace("/", "_"),
                    f"{output_filename}_imagenet_entropy_baseline.json")
                if file_exists(output_path):
                    with patch_file_open(output_path, "r") as f:
                        per_category_entropy_baseline = json.load(f)
                    continue
            else:
                output_path = os.path.join(
                    gen_folder,
                    model_id.replace("/", "_"),
                    f"{output_filename}_imagenet_entropy_ours.json")
                if file_exists(output_path):
                    with patch_file_open(output_path, "r") as f:
                        per_category_entropy_ours = json.load(f)
                    continue
            for category in categories:
                entropy = calc_entropy_per_imagenet_category(model, processor, imagenet_classes2synset, gen_images, category)
                # print(f"\t\t{category}: {entropy}")
                if b:
                    per_category_entropy_baseline[category] = entropy
                else:
                    per_category_entropy_ours[category] = entropy
    
            # for category in categories:
            #     print(f"Category: {category}")
            #     baseline_numbers = [f'{x:.3f}' for x in per_category_entropy_baseline[category]]
            #     print(f"\tBaseline: {baseline_numbers}")
            #     ours_numbers = [f'{x:.3f}' for x in per_category_entropy_ours[category]]
            #     print(f"\tOurs: {ours_numbers}")
            if b:
                print(f"Baseline: {per_category_entropy_baseline}")
                with patch_file_open(output_path, "w") as f:
                    json.dump(per_category_entropy_baseline, f)
            else:
                print(f"Ours: {per_category_entropy_ours}")
                with patch_file_open(output_path, "w") as f:
                    json.dump(per_category_entropy_ours, f)
        baseline_score = 0
        our_score = 0
        for category in categories:
            # print(f"Attribute: {column}")
            # print(f"\tBaseline: {entropy_per_attribute_baseline[key]}")
            # print(f"\tOurs: {entropy_per_attribute_ours[key]}")
            baseline_score += per_category_entropy_baseline[category]
            our_score += per_category_entropy_ours[category]
        baseline_score /= len(categories)
        our_score /= len(categories)
        print(f"\tBaseline: {baseline_score}")
        print(f"\tOurs: {our_score}")
    return


def calc_entropy_human_faces(gen_folder, model_id="SG161222/RealVisXL_V4.0", guidance_scale=5, baseline=False):
    import numpy as np
    from collections import defaultdict
    entropy_per_attribute_baseline = defaultdict(list)
    entropy_per_attribute_ours = defaultdict(list)

    categories = ['person', 'woman', 'man', 'child']

    for guidance_scale in [guidance_scale]:

        entropy_per_attribute_baseline = defaultdict(float)
        entropy_per_attribute_ours = defaultdict(float)
        for baseline in [baseline]:
            print(f"Baseline: {baseline}")
            gen_images, output_filename = load_gen_images(gen_folder, model_id, guidance_scale=guidance_scale, baseline=baseline, categories=categories, load=True)

            if baseline:
                output_path = os.path.join(
                 gen_folder,
                 model_id.replace("/", "_"),
                 f"{output_filename}_person_entropy_baseline.json")
                if file_exists(output_path):
                    with patch_file_open(output_path, "r") as f:
                        entropy_per_attribute_baseline = json.load(f)
                    continue
            else:
                output_path = os.path.join(
                    gen_folder,
                    model_id.replace("/", "_"),
                    f"{output_filename}_person_entropy_ours.json")
                if file_exists(output_path):
                    with patch_file_open(output_path, "r") as f:
                        entropy_per_attribute_ours = json.load(f)
                    continue

            for category in categories:
                print(f"Category: {category}")
                fake_images = gen_images[category]
                
                from src.copycat.mitigation.face_attribute_detect import predidct_age_gender_race_w_pilimgs
                # result = run_face_attribute_detection(output_csv_path)
                result = predidct_age_gender_race_w_pilimgs(fake_images)
                for column in ['race', 'gender', 'age']:
                    # get the count for values in each column
                    counts = result[column].value_counts()
                    total = counts.sum()
                    # calculate entropy
                    entropy = 0
                    for count in counts:
                        entropy += -count/total * np.log(count/total)
                    if baseline:
                        entropy_per_attribute_baseline[category+"_"+column] = entropy
                    else:
                        entropy_per_attribute_ours[category+"_"+column] = entropy
            if baseline:
                print(f"Baseline: {entropy_per_attribute_baseline}")
                with patch_file_open(output_path, "w") as f:
                    json.dump(entropy_per_attribute_baseline, f)
            else:
                print(f"Ours: {entropy_per_attribute_ours}")
                with patch_file_open(output_path, "w") as f:
                    json.dump(entropy_per_attribute_ours, f)
    
    all_baseline = 0
    all_ours = 0
    for column in ['race', 'gender', 'age']:
        baseline_score = 0
        our_score = 0
        for category in categories:
            key = category + "_" + column
            # print(f"Attribute: {column}")
            # print(f"\tBaseline: {entropy_per_attribute_baseline[key]}")
            # print(f"\tOurs: {entropy_per_attribute_ours[key]}")
            baseline_score += entropy_per_attribute_baseline[key]
            our_score += entropy_per_attribute_ours[key]
        baseline_score /= len(categories)
        our_score /= len(categories)
        all_baseline += baseline_score
        all_ours += our_score
        print(f"Attribute: {column}")
        print(f"\tBaseline: {baseline_score}")
        print(f"\tOurs: {our_score}")
    all_baseline /= 3
    all_ours /= 3
    print(f"Overall")
    print(f"\tBaseline: {all_baseline}")
    print(f"\tOurs: {all_ours}")


if __name__ == "__main__":
    from fire import Fire
    Fire()