import torch
import numpy as np
import os

def map2tmp(path):
    # return os.path.join("/tmp/azfuse", path)
    from azfuse import File
    return File.get_cache_file(path)


def list_files_in_folder(folder):
    try:
        from azfuse import File
        return File.list(folder)
    except ImportError:
        return os.listdir(folder)

def get_list_of_files_to_prepare(folder):
    from azfuse import File
    filename = os.path.basename(folder)
    if filename.startswith('.'):
        return []
    to_prepare = []

    list_of_subfolders = [f for f in File.list(folder)]

    if len(list_of_subfolders) == 0:
        return [folder]
    
    for f in list_of_subfolders:
        list_of_files = get_list_of_files_to_prepare(f)
        to_prepare.extend(list_of_files)
    return to_prepare


def seed_everything(seed: int):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def file_exists(file_path):
    try:
        from azfuse import File
        return File.isfile(file_path)
    except ImportError:
        import os
        return os.path.exists(file_path)


def patch_file_open(file_path, mode="r"):
    try:
        from azfuse import File
        f = File.open(file_path, mode)
    except ImportError:
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path)
        f = open(file_path, mode)
    return f


def write_to_file(file_path, content, mode="w"):
    try:
        from azfuse import File
        with File.open(file_path, mode) as f:
            f.write(content)
    except ImportError:
        with open(file_path, mode) as f:
            f.write(content)


def read_from_file(file_path, mode="r"):
    try:
        from azfuse import File
        with File.open(file_path, mode) as f:
            content = f.read()
    except ImportError:
        with open(file_path, mode) as f:
            content = f.read()
    return content


def nparray_from_base64(base64string):
    # load a numpy array from base64 string
    import io
    import numpy as np
    import base64
    bytes = base64.b64decode(base64string)
    nparray = np.load(io.BytesIO(bytes))
    return nparray


def pilimg_from_base64(imagestring):
    from PIL import Image
    import base64
    try:
        import io
        jpgbytestring = base64.b64decode(imagestring)
        image = Image.open(io.BytesIO(jpgbytestring))
        image = image.convert('RGB')
        return image
    except:
        return None


def npmask_from_base64(imagestring):
    from PIL import Image
    import base64
    try:
        import io
        jpgbytestring = base64.b64decode(imagestring)
        mask_pil = Image.open(io.BytesIO(jpgbytestring))
        mask_np = np.asarray(mask_pil).astype(bool).astype(np.uint8)
        return mask_np
    except:
        return None


def to_jpeg_base64(img_or_path):
    import io
    from PIL import Image
    import base64

    if isinstance(img_or_path, str):
        ext = os.path.splitext(img_or_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            # does not need to load into PIL anymore
            with patch_file_open(img_or_path, "rb") as f:
                contents = f.read()
                return base64.b64encode(contents).decode("utf-8")
        with patch_file_open(img_or_path, "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path
    else:
        raise ValueError("img_or_path must be a file path or a Image object")
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        contents = output.getvalue()
        return base64.b64encode(contents).decode("utf-8")
    

def mask_to_jpeg_base64(np_array_or_path):
    from PIL import Image
    if isinstance(np_array_or_path, str):
        with patch_file_open(np_array_or_path, "rb") as f:
            mask = np.load(f)
    elif isinstance(np_array_or_path, np.ndarray):
        mask = np_array_or_path
    else:
        raise ValueError("np_array_or_path must be a file path or a numpy array")
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask_jpeg_base64 = to_jpeg_base64(mask)
    return mask_jpeg_base64


def build_grounding_dino():
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

    # environment settings
    # use bfloat16
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build grounding dino from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return grounding_model, processor


def call_grounding_dino_to_crop(model, processor, img_or_path, text="main character."):
    from PIL import Image
    # # setup the input image and text prompt for SAM 2 and Grounding DINO
    # # VERY important: text queries need to be lowercased + end with a dot
    if isinstance(img_or_path, str):
        with patch_file_open(img_or_path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
    elif isinstance(img_or_path, Image.Image):
        image = img_or_path
    else:
        raise ValueError("img_or_path must be a file path or a PIL image")

    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()

    output = {"image": image}

    if len(input_boxes):
        output["bboxes"] = input_boxes
        # crop the image based on the the box coordinates
        cropped_image = image.crop(input_boxes[0])
        output["cropped_image"] = cropped_image
    else:
        output["bboxes"] = None
        output["cropped_image"] = None
    return output


def build_sam2_img():
    import os
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError("Please install sam2 first, from https://github.com/facebookresearch/segment-anything-2")

    # build SAM 2 model

    sam2_checkpoint = "models/sam2/sam2_hiera_large.pt"
    if not os.path.exists(sam2_checkpoint):
        try:
            from azfuse import File
            File.prepare([sam2_checkpoint])
            sam2_checkpoint = map2tmp(sam2_checkpoint)
        except ImportError:
            raise FileNotFoundError(f"Checkpoint {sam2_checkpoint} does not exist, follow  https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/checkpoints/download_ckpts.sh to download the checkpoint.")
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def build_hqsam():
    from segment_anything_hq import sam_model_registry, SamPredictor# from segment_anything import SamPredictor, sam_model_registry
    model_type = "vit_h" #"vit_l/vit_b/vit_h/vit_tiny"
    hqsam_checkpoint = "models/hqsam/sam_hq_vit_h.pth"
    if not os.path.exists(hqsam_checkpoint):
        try:
            from azfuse import File
            File.prepare([hqsam_checkpoint])
            hqsam_checkpoint = map2tmp(hqsam_checkpoint)
        except ImportError:
            raise FileNotFoundError(f"Checkpoint {hqsam_checkpoint} does not exist, follow  https://github.com/SysCV/sam-hq?tab=readme-ov-file to download the checkpoint.")
    sam = sam_model_registry[model_type](checkpoint=hqsam_checkpoint)
    sam = sam.cuda()
    predictor = SamPredictor(sam)
    return predictor


def get_only_mask_img_with_bb_prompt(model, image, input_boxes):
    if len(input_boxes) == 0:
        raise ValueError("No bounding box found in the image.")
    
    output = {}
    image = np.array(image.convert("RGB"))
    model.set_image(image)
    try:
        masks, _, _ = model.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    except Exception as e:
        print(f"Error in predicting the mask: {e}")
        return output
        

    if masks.ndim == 4:
        masks = masks.squeeze(1)


    binary_mask = masks[0].astype(bool)
    output_image = np.full_like(image, 255)

    # Apply the mask to the image
    output_image[binary_mask] = image[binary_mask]

    # make output_image PIL
    from PIL import Image
    output_image = Image.fromarray(output_image)
    cropped_image = output_image.crop(input_boxes[0])
    # also crop the mask
    cropped_binary_mask = Image.fromarray(binary_mask).crop(input_boxes[0])
    
    # print("cropped_image", cropped_image.size)
    # print("binary_mask", binary_mask.size)
    # convert to binary mask
    cropped_binary_mask = np.array(cropped_binary_mask)
    output["cropped_masked_image"] = cropped_image
    output["cropped_binary_mask"] = cropped_binary_mask
    output["binary_mask"] = binary_mask
    output["masked_image"] = output_image
    return output


def grounding_and_then_mask(image_or_path, model_kwargs={}, to_save=False, overwrite=False):
    dino_model = model_kwargs.get("dino_model", None)
    dino_processor = model_kwargs.get("dino_processor", None)
    sam2_model = model_kwargs.get("sam2_model", None)
    hqsam_model = model_kwargs.get("hqsam_model", None)
    from PIL import Image
    import io
    if dino_model is None:
        return
    from src.copycat.utils import file_exists
    save_path = None
    if isinstance(image_or_path, str):
        if not file_exists(image_or_path):
            raise FileNotFoundError(f"image {image_or_path} does not exist")
        if to_save:
            save_path = image_or_path
    elif isinstance(image_or_path, Image.Image):
        pass
    else:
        raise ValueError("image_or_path must be a file path or a PIL image")
    # # VERY important: text queries need to be lowercased + end with a dot

    # this does not work well
    # dino_prompt = prompt.lower()
    # if not dino_prompt.endswith("."):
    #     dino_prompt += "."
    dino_prompt = "main character."

    from src.copycat.utils import call_grounding_dino_to_crop, get_only_mask_img_with_bb_prompt
    with torch.autocast("cuda", enabled=True):
        output = {}
        try:
            grounding_output = call_grounding_dino_to_crop(dino_model, dino_processor, image_or_path, text=dino_prompt)
            boxes = grounding_output.get("bboxes", None)
            image = grounding_output.get("image", None)
            cropped_image = grounding_output.get("cropped_image", None)
            if save_path is not None:
                crop_output_path = os.path.splitext(save_path)[0] + "_cropped.jpg"
                if not file_exists(crop_output_path) or overwrite:
                    with patch_file_open(crop_output_path, "wb") as f:
                        img_byte_arr = io.BytesIO()
                        cropped_image.save(img_byte_arr, format='jpeg')
                        # write the bytes to file
                        f.write(img_byte_arr.getvalue())
            output.update(grounding_output)
        except Exception as e:
            print(f"Error in grounding and cropping image: {e}")
            
        if boxes is None or len(boxes) == 0:
            print(f"No boxes detected")
            return output
        if sam2_model is not None:
            if save_path is not None:
                crop_output_path = os.path.splitext(save_path)[0] + "_cropped_mask.jpg"
                output_crop_mask_file = os.path.splitext(save_path)[0] + "_cropped_mask_binary.npy"
                output_mask_file = os.path.splitext(save_path)[0] + "_mask_binary.npy"
                mask_output_path = os.path.splitext(save_path)[0] + "_masked.jpg"
                if not file_exists(mask_output_path) or overwrite:

                    try:
                        mask_output = get_only_mask_img_with_bb_prompt(sam2_model, image, boxes)
                        cropped_masked_image = mask_output["cropped_masked_image"]
                        cropped_binary_mask = mask_output["cropped_binary_mask"]
                        binary_mask = mask_output["binary_mask"]
                        masked_image = mask_output["masked_image"]
                        output["sam2_output"] = mask_output
                    except Exception as e:
                        print(f"Error in SAM2: {e}")
                        return
                
                    with patch_file_open(crop_output_path, "wb") as f:
                        img_byte_arr = io.BytesIO()
                        cropped_masked_image.save(img_byte_arr, format='jpeg')
                        # write the bytes to file
                        f.write(img_byte_arr.getvalue())
                    with patch_file_open(output_mask_file, "wb") as f:
                        np.save(f, binary_mask)
                    with patch_file_open(output_crop_mask_file, "wb") as f:
                        np.save(f, cropped_binary_mask)
                    with patch_file_open(mask_output_path, "wb") as f:
                        img_byte_arr = io.BytesIO()
                        masked_image.save(img_byte_arr, format='jpeg')
                        # write the bytes to file
                        f.write(img_byte_arr.getvalue())
            else:
                mask_output = get_only_mask_img_with_bb_prompt(sam2_model, image, boxes)
                output["sam2_output"] = mask_output

        if hqsam_model is not None:
            if save_path is not None:
                crop_output_path = os.path.splitext(save_path)[0] + "_cropped_hqsam_mask.jpg"
                output_crop_mask_file = os.path.splitext(save_path)[0] + "_cropped_hqsam_mask_binary.npy"
                output_mask_file = os.path.splitext(save_path)[0] + "_hqsam_mask_binary.npy"
                mask_output_path = os.path.splitext(save_path)[0] + "_hqsam_masked.jpg"
                if not file_exists(mask_output_path) or overwrite:
                    try:
                        mask_output = get_only_mask_img_with_bb_prompt(hqsam_model, image, boxes)
                        cropped_masked_image = mask_output["cropped_masked_image"]
                        cropped_binary_mask = mask_output["cropped_binary_mask"]
                        binary_mask = mask_output["binary_mask"]
                        masked_image = mask_output["masked_image"]
                        output["hqsam_output"] = mask_output
                    except Exception as e:
                        print(f"Error in HQSAM: {e}")
                        return
                    with patch_file_open(crop_output_path, "wb") as f:
                        img_byte_arr = io.BytesIO()
                        cropped_masked_image.save(img_byte_arr, format='jpeg')
                        # write the bytes to file
                        f.write(img_byte_arr.getvalue())
                    with patch_file_open(output_mask_file, "wb") as f:
                        np.save(f, binary_mask)
                    with patch_file_open(output_crop_mask_file, "wb") as f:
                        np.save(f, cropped_binary_mask)
                    with patch_file_open(mask_output_path, "wb") as f:
                        img_byte_arr = io.BytesIO()
                        masked_image.save(img_byte_arr, format='jpeg')
                        # write the bytes to file
                        f.write(img_byte_arr.getvalue())
            else:
                mask_output = get_only_mask_img_with_bb_prompt(hqsam_model, image, boxes)
                output["hqsam_output"] = mask_output
        return output


def crop_and_mask_image(img_or_path, crop_mask_model_kwargs, mode=None):
    from PIL import Image
    if isinstance(img_or_path, str):
        with patch_file_open(img_or_path, "rb") as f:
            pilimg = Image.open(f)
            pilimg = pilimg.convert("RGB")
            to_save = True
    elif isinstance(img_or_path, Image.Image):
        pilimg = img_or_path
        to_save = False
    else:
        raise ValueError("img_or_path must be a file path or a PIL image")
    if mode is None:
        return pilimg
    elif mode == "crop":
        grounding_output = grounding_and_then_mask(pilimg, crop_mask_model_kwargs, to_save=to_save)
        if grounding_output.get("cropped_image", None) is None:
            return pilimg
        return grounding_output["cropped_image"]
    elif mode == "mask":
        mask_output = grounding_and_then_mask(pilimg, crop_mask_model_kwargs, to_save=to_save)
        if "sam2_output" not in mask_output or mask_output["sam2_output"].get("masked_image", None) is None:
            return pilimg
        return mask_output["sam2_output"]["masked_image"]
    elif mode == "crop_mask":
        mask_output = grounding_and_then_mask(pilimg, crop_mask_model_kwargs, to_save=to_save)
        if "sam2_output" not in mask_output or mask_output["sam2_output"].get("cropped_masked_image", None) is None:
            return pilimg
        return mask_output["sam2_output"]["cropped_masked_image"]
    else:
        raise ValueError("mode must be one of [None, 'crop', 'mask', 'crop_mask']")
