import rasterio
import os
from rasterio.windows import Window
from typing import List, Optional
from tqdm import tqdm


def get_images_from_folder(a):
    pass


class SplittingImage:
    """
    Class that allows user to im
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
        tile_size: int = 512,
        overlap: int = 128,
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
