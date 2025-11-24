"""Utilities for CMMD computation copied from cmmd-pytorch package."""

import glob
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import tqdm


# Configuration constants
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
SIGMA = 10
SCALE = 1000


class CMMDDataset(Dataset):
    """Dataset for loading images for CMMD computation."""
    
    def __init__(self, path, reshape_to, max_count=-1):
        self.path = path
        self.reshape_to = reshape_to
        self.max_count = max_count
        img_path_list = self._get_image_list()
        if max_count > 0:
            img_path_list = img_path_list[:max_count]
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["png", "jpg", "jpeg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        image_list.sort()
        return image_list

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        return im.resize((size, size), resample=Image.BICUBIC)

    def _read_image(self, path, size):
        im = Image.open(path)
        if size > 0:
            im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        x = self._read_image(img_path, self.reshape_to)
        if x.ndim == 3:
            return x
        elif x.ndim == 2:
            return np.tile(x[..., np.newaxis], (1, 1, 3))


class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
        self._model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME).eval()
        
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        
        self.input_image_size = self.image_processor.crop_size["height"]

    @staticmethod
    def _resize_bicubic(images, size):
        """Resize images using bicubic interpolation."""
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
        images = images.permute(0, 2, 3, 1).numpy()
        return images

    @torch.no_grad()
    def embed(self, images):
        """Compute CLIP embeddings for the given images.
        
        Args:
            images: An image array of shape (batch_size, height, width, 3). 
                   Values are in range [0, 1].
        
        Returns:
            Embedding array of shape (batch_size, embedding_width).
        """
        images = self._resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs.numpy()


def compute_embeddings_for_dir(img_dir, embedding_model, batch_size, max_count=-1):
    """Compute embeddings for all images in a directory.
    
    Args:
        img_dir: Directory containing .jpg or .png image files.
        embedding_model: The embedding model to use.
        batch_size: Batch size for the embedding model inference.
        max_count: Max number of images in the directory to use.
    
    Returns:
        Computed embeddings of shape (num_images, embedding_dim).
    """
    dataset = CMMDDataset(
        str(img_dir), 
        reshape_to=embedding_model.input_image_size, 
        max_count=max_count
    )
    count = len(dataset)
    print(f"    Calculating embeddings for {count} images from {Path(img_dir).name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_embs = []
    for batch in tqdm.tqdm(dataloader, total=count // batch_size, leave=False):
        image_batch = batch.numpy() / 255.0
        
        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError(
                f"Image values are expected to be in [0, 1]. "
                f"Found: [{np.min(image_batch)}, {np.max(image_batch)}]."
            )
        
        embs = embedding_model.embed(image_batch)
        all_embs.append(embs)
    
    return np.concatenate(all_embs, axis=0).astype('float32')


def mmd(x, y):
    """Compute Maximum Mean Discrepancy between two sets of embeddings.
    
    This implements the minimum-variance/biased version of the estimator.
    
    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
    
    Returns:
        The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))
    
    gamma = 1 / (2 * SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + 
                           torch.unsqueeze(x_sqnorms, 1) + 
                           torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + 
                           torch.unsqueeze(x_sqnorms, 1) + 
                           torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + 
                           torch.unsqueeze(y_sqnorms, 1) + 
                           torch.unsqueeze(y_sqnorms, 0)))
    )
    
    return SCALE * (k_xx + k_yy - 2 * k_xy)