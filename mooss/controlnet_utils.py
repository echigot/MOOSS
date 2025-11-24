import cv2
import numpy as np
import torch
from PIL import Image
from .gta2ade import city_to_ade20k
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import logging

CROP_SIZE = (1024, 512)

def convert_label_to_ade(label_image):
    label_array = np.array(label_image)
    if label_array.ndim == 3:
        label_array = label_array[:, :, 0]
    ade_label = city_to_ade20k(label_array)
    return ade_label.resize(CROP_SIZE, Image.NEAREST)


def generate_canny_image(input_image):
    input_array = np.array(input_image)
    edges = cv2.Canny(input_array, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges] * 3, axis=2)
    canny_image = Image.fromarray(edges).resize(CROP_SIZE, Image.NEAREST)
    return canny_image


    
def load_controlnet_pipeline(checkpoint_path="runwayml/stable-diffusion-v1-5"):
    
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    
    controlnets = [
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True
        ),
    ]
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        checkpoint_path,
        controlnet=controlnets,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)

    return pipe
    
def run_inference(pipe, style_image, ade_label, canny_image, style_name=""):
    prompt = "photography of urban scene, driving, dashcam, road"
    prompt = [t + prompt for t in [f"sks {style_name}"]]
    generator = [torch.Generator(device="cpu").manual_seed(1) for _ in prompt]
    
    result = pipe(
        prompt=prompt,
        image=[style_image],
        control_image=[ade_label, canny_image],
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
        generator=generator,
        num_inference_steps=20,
    )
    return result.images


def normalize_with_reference(input, reference):
    # Convert to numpy arrays
    arr1 = np.array(input).astype(np.float32)
    arr2 = np.array(reference).astype(np.float32)
    
    # Compute mean and std of reference (img2)
    mean_ref = arr2.mean()
    std_ref = arr2.std()
    
    # Compute mean and std of img1
    mean1 = arr1.mean()
    std1 = arr1.std()
    
    # Normalize arr1 to have the same mean & std as arr2
    normalized = (arr1 - mean1) / (std1 + 1e-8)  # standardize
    normalized = normalized * std_ref + mean_ref  # scale to reference stats
    
    # Clip to valid range [0, 255] and convert back to uint8
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    # Convert back to PIL image
    return Image.fromarray(normalized)

