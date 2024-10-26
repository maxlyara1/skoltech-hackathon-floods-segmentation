import rasterio
import os
from rasterio.windows import Window
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SplittingImage:
    """
    Class that allows user to preprocess files of size different from 640x640
    """

    def _get_tiles_with_overlap_(
        self, image_width: int, image_height: int, tile_size: int, overlap: int
    ) -> List[Window]:
        """
        Calculate the windows for tiles with specified overlap across the image.

        Parameters:
            image_width (int): The width of the input image in pixels.
            image_height (int): The height of the input image in pixels.
            tile_size (int): The size of each tile (assumes square tiles).
            overlap (int): The number of overlapping pixels between adjacent tiles.

        Returns:
            List[Window]: A list of rasterio Window objects representing each tile.
        """
        step_size = tile_size - overlap
        tiles = []
        for y in range(0, image_height, step_size):
            for x in range(0, image_width, step_size):
                window = Window(x, y, tile_size, tile_size)
                # Adjust window if it exceeds the image bounds
                window = window.intersection(Window(0, 0, image_width, image_height))
                tiles.append(window)
        return tiles

    def _save_tile_(
        self,
        src_dataset: rasterio.io.DatasetReader,
        window: Window,
        output_folder: str,
        tile_index: int,
        image_id: int,
    ) -> None:
        """
        Extract and save a single tile from the source dataset.

        Parameters:
            src_dataset (rasterio.io.DatasetReader): The opened rasterio dataset (the input image).
            window (Window): The window (rasterio Window object) defining the tile.
            output_folder (str): The folder where the tiles will be saved.
            tile_index (int): Index of the tile to be used for naming the file.
            image_id (int): Image id to be used for naming the file.

        Returns:
            None
        """
        transform = src_dataset.window_transform(window)
        tile_data = src_dataset.read(window=window)

        profile = src_dataset.profile
        profile.update(
            {
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "transform": transform,
            }
        )

        output_filename = os.path.join(
            output_folder, f"tile_{image_id}_{tile_index}.tif"
        )
        with rasterio.open(output_filename, "w", **profile) as dst:
            dst.write(tile_data)

    def image_split(
        self,
        image_path: str,
        output_folder: str,
        mask_path: Optional[str] = None,
        tile_size: int = 640,
        overlap: int = 0,
        image_id: int = 0,
    ) -> None:
        """
        Split a large GeoTIFF image and its corresponding mask (if provided) into tiles with overlap
        and save them.

        Parameters:
            image_path (str): The file path of the input TIFF image.
            mask_path (Optional[str]): The file path of the corresponding mask TIFF image. If None, only image is processed.
            output_folder (str): The folder where the tiles will be saved.
            tile_size (int, optional): The size of the tiles. Default is 512x512.
            overlap (int, optional): The number of pixels to overlap between tiles. Default is 128 pixels.
            image_id (int, optional): ID of the input image to be used for naming the file.
                Defaults to 0.

        Returns:
            None
        """
        with rasterio.open(image_path) as src_image:
            image_width = src_image.width
            image_height = src_image.height
            print(f"width: {image_width}")
            print(f"height: {image_height}")

            # Create output directories for images and masks (if available)
            images_folder = os.path.join(output_folder, "images")
            os.makedirs(images_folder, exist_ok=True)

            if mask_path:
                masks_folder = os.path.join(output_folder, "masks")
                os.makedirs(masks_folder, exist_ok=True)

            # Get list of tiles with overlap
            tiles = self._get_tiles_with_overlap_(
                image_width, image_height, tile_size, overlap
            )

            # Save image tiles (and mask tiles if provided)
            if mask_path:
                with rasterio.open(mask_path) as src_mask:
                    for idx, window in tqdm(enumerate(tiles)):
                        self._save_tile_(
                            src_image, window, images_folder, idx, image_id
                        )
                        self._save_tile_(src_mask, window, masks_folder, idx, image_id)
            else:
                for idx, window in tqdm(enumerate(tiles)):
                    self._save_tile_(src_image, window, images_folder, idx, image_id)


def image_padding(image, target_size=640):
    """
    Pad an image to a target size using reflection padding.
    """
    height, width = image.shape[1:3]
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    padded_image = np.pad(
        image, ((0, 0), (0, pad_height), (0, pad_width)), mode="reflect"
    )
    height, width = padded_image.shape[1:3]
    return padded_image


def mask_padding(mask, target_size=640):
    """
    Pad a mask to a target size using reflection padding.
    """
    height, width = mask.shape
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    padded_mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="reflect")
    return padded_mask


def get_data_list(img_path: str) -> np.array:
    """Retrieves a list of file names from the given directory.

    Args:
        img_path (str): Folder path

    Returns:
        np.array: file names from folder
    """
    name = []
    for _, _, filenames in os.walk(
        img_path
    ):  # given a directory iterates over the files
        for filename in filenames:
            f = filename.split(".")[0]
            name.append(f)

    df = (
        pd.DataFrame({"id": name}, index=np.arange(0, len(name)))
        .sort_values("id")
        .reset_index(drop=True)
    )
    df = df["id"].values

    return np.delete(df, 0)


class WaterDataset(Dataset):
    """
    A custom dataset class for loading and preprocessing paired image and mask files for water segmentation.

    Attributes:
    ----------
    img_path : str
        Directory path to the input image files.
    mask_path : str
        Directory path to the mask files corresponding to each image.
    file_names : list of str
        List of file names (without extensions) for the image and mask files to be loaded.

    Methods:
    -------
    __len__()
        Returns the total number of image-mask pairs in the dataset.

    __getitem__(idx)
        Retrieves the image and mask at the specified index `idx`, applies padding, and returns them.

    Examples:
    --------
    >>> ds = WaterDataset(img_path='data/images/', mask_path='data/masks/', file_names=data_list)
    >>> dl = DataLoader(ds)

    The DataLoader `dl` can then be used to retrieve batches of paired images and masks.

    """

    def __init__(self, img_path, mask_path, file_names):
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with rasterio.open(self.img_path + self.file_names[idx] + ".tif") as fin:
            image = fin.read()
        image = image_padding(image).astype(np.float32)

        with rasterio.open(self.mask_path + self.file_names[idx] + ".tif") as fin:
            mask = fin.read(1)
        mask = mask_padding(mask)

        return image, mask


def get_data_loader(img_path: str, mask_path: str) -> DataLoader:
    """Function for DataLoader preparation

    Args:
        img_path (str): Folder with fixed size images
        mask_path (str): Folder with masks for fixed size images

    Returns:
        DataLoader: DataLoader with paired image and mask files
    """
    data_list = get_data_list(img_path)
    ds = WaterDataset(img_path=img_path, mask_path=mask_path, file_names=data_list)
    dl = DataLoader(ds)
    return dl
