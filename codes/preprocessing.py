import os
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from tqdm import tqdm


class SplittingImage:
    """
    A class for preprocessing large GeoTIFF images by splitting them into smaller tiles
    with specified overlap.

    Methods:
    -------
    _get_tiles_with_overlap_(image_width, image_height, tile_size, overlap):
        Calculates windows for overlapping tiles across the image.
    _save_tile_(src_dataset, window, output_folder, tile_index, image_id):
        Saves a single tile from the source image or mask dataset.
    image_split(image_path, output_folder, mask_path=None, tile_size=640, overlap=0, image_id=0):
        Splits a large image and its mask (if provided) into tiles and saves them.
    """

    def _get_tiles_with_overlap_(
        self, image_width: int, image_height: int, tile_size: int, overlap: int
    ) -> List[Window]:
        """
        Calculates a list of windows (tiles) for an image based on the specified tile size and overlap.

        Args:
            image_width (int): Width of the original image in pixels.
            image_height (int): Height of the original image in pixels.
            tile_size (int): Dimension of each square tile.
            overlap (int): Overlap in pixels between adjacent tiles.

        Returns:
            List[Window]: List of rasterio Window objects representing the tiles.
        """
        step_size = tile_size - overlap
        tiles = []
        for y in range(0, image_height, step_size):
            for x in range(0, image_width, step_size):
                window = Window(x, y, tile_size, tile_size)
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
        Extracts and saves a single tile from a dataset within a specified window.

        Args:
            src_dataset (rasterio.io.DatasetReader): The input image or mask dataset.
            window (Window): Window representing the tile boundaries.
            output_folder (str): Directory to save the tile.
            tile_index (int): Index used in naming the output file.
            image_id (int): ID used in naming the output file.

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
        Splits an image (and optional mask) into tiles and saves them to the output directory.

        Args:
            image_path (str): Path to the input image file.
            mask_path (Optional[str]): Path to the mask file, if available.
            output_folder (str): Directory to save the tiles.
            tile_size (int, optional): Size of each tile, default is 640x640 pixels.
            overlap (int, optional): Overlap between tiles, default is 0.
            image_id (int, optional): ID used in naming output files.

        Returns:
            None
        """
        with rasterio.open(image_path) as src_image:
            image_width = src_image.width
            image_height = src_image.height

            images_folder = os.path.join(output_folder, "images")
            os.makedirs(images_folder, exist_ok=True)

            if mask_path:
                masks_folder = os.path.join(output_folder, "masks")
                os.makedirs(masks_folder, exist_ok=True)

            tiles = self._get_tiles_with_overlap_(
                image_width, image_height, tile_size, overlap
            )

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


def _image_padding_(image, target_size=640):
    """
    Pads an image to the target size using reflection padding.

    Args:
        image (np.ndarray): The image array to be padded.
        target_size (int): The target size for padding.

    Returns:
        np.ndarray: The padded image array.
    """
    height, width = image.shape[1:3]
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    return np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode="reflect")


def _mask_padding_(mask, target_size=640):
    """
    Pads a mask to the target size using reflection padding.

    Args:
        mask (np.ndarray): The mask array to be padded.
        target_size (int): The target size for padding.

    Returns:
        np.ndarray: The padded mask array.
    """
    height, width = mask.shape
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    return np.pad(mask, ((0, pad_height), (0, pad_width)), mode="reflect")


def get_sorted_data_list(img_path: str) -> np.array:
    """
    Retrieves a sorted list of filenames from the specified directory.

    Args:
        img_path (str): Path to the directory containing image files.

    Returns:
        np.array: Array of sorted file names (without extensions).
    """
    names = [
        filename.split(".")[0]
        for _, _, filenames in os.walk(img_path)
        for filename in filenames
    ]
    return np.array(sorted(names))


class WaterDataset(Dataset):
    """
    A dataset class for loading paired image and mask files for segmentation.

    Attributes:
        img_path (str): Directory path to input images.
        mask_path (str): Directory path to masks.
        file_names (List[str]): List of file names (without extensions).

    Methods:
        __len__(): Returns the number of samples.
        __getitem__(idx): Returns a padded image and mask pair.
    """

    def __init__(self, img_path, mask_path, file_names):
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with rasterio.open(f"{self.img_path}{self.file_names[idx]}.tif") as fin:
            image = fin.read()
        image = _image_padding_(image).astype(np.float32)

        with rasterio.open(f"{self.mask_path}{self.file_names[idx]}.tif") as fin:
            mask = fin.read(1)
        mask = _mask_padding_(mask)

        return image, mask


def get_data_loader(img_path: str, mask_path: str) -> DataLoader:
    """
    Initializes a DataLoader for paired image and mask files.

    Args:
        img_path (str): Directory with processed images.
        mask_path (str): Directory with corresponding masks.

    Returns:
        DataLoader: DataLoader for retrieving batches of paired image and mask files.
    """
    data_list = get_sorted_data_list(img_path)
    dataset = WaterDataset(img_path=img_path, mask_path=mask_path, file_names=data_list)
    return DataLoader(dataset)


def preprocess_data(
    big_images_path: str,
    big_masks_path: Optional[str] = None,
    train: bool = True,
    image_for_model_size: int = 640,
) -> DataLoader:
    paths_big_images = get_sorted_data_list(big_images_path)
    paths_masks_big_images = (
        get_sorted_data_list(big_masks_path) if train and big_masks_path else None
    )

    overlap = 32 if train else 0
    class_to_split = SplittingImage()

    for path_to_file in os.listdir(big_images_path):
        if not path_to_file.lower().endswith(".tif"):
            continue  # Skip files that are not .tif
        full_path = os.path.join(big_images_path, path_to_file)
        if os.path.isfile(full_path):
            class_to_split.image_split(
                image_path=full_path,
                output_folder="data",
                mask_path=(
                    os.path.join(big_masks_path, path_to_file)
                    if big_masks_path
                    else None
                ),
                tile_size=image_for_model_size,
                overlap=overlap,
                image_id=path_to_file.split(".")[0],
            )

    data_loader = get_data_loader(
        img_path="data/images/",
        mask_path="data/masks/" if train else None,
    )
    return data_loader


def main():
    preprocess_data(
        big_images_path="train/images",
        big_masks_path="train/masks",
        train=True,
        image_for_model_size=640,
    )


main()
