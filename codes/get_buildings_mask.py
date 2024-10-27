import os
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
from pyproj import Transformer


def apply_building_mask(tif_path: str, geojson_data: dict) -> str:
    """
    Apply a building mask to a .tif image based on GeoJSON data.

    Parameters:
    tif_path (str): The file path to the .tif image.
    geojson_data (dict): The GeoJSON data containing building geometries.

    Returns:
    str: The file path to the created building mask.
    """
    # List of necessary tags for filtering objects
    building_tags = {"house", "garages", "detached"}
    mask_folder = "train/building_masks"  # Folder for building masks

    with rasterio.open(tif_path) as src:
        # Get the bounds of the image and its coordinate reference system (CRS)
        image_bounds = box(*src.bounds)
        image_crs = src.crs
        transformer = Transformer.from_crs("EPSG:4326", image_crs, always_xy=True)

        # Filter buildings by the necessary tags that intersect with the image bounds
        filtered_buildings = []
        for feature in geojson_data["features"]:
            tag = feature["properties"].get("tags")
            if (
                tag in building_tags
            ):  # Check if the tag is one of house, garages, or detached
                geom = shape(feature["geometry"])

                # Transform the building geometry to the image's CRS
                transformed_geom = transform(transformer.transform, geom)

                # Check if the transformed geometry intersects with the image bounds
                if transformed_geom.intersects(image_bounds):
                    filtered_buildings.append(mapping(transformed_geom))

        # Check if there are any buildings to process
        if not filtered_buildings:
            print(f"No available GeoJSON objects for this folder: {tif_path}")
            return
        else:
            print(f"Found available GeoJSON for {tif_path}")

        # Extract the segmented image based on the filtered buildings
        out_image, out_transform = mask(src, filtered_buildings, crop=True)

        # Create a mask where buildings are marked as 1 and all other pixels are 0
        mask_array = np.zeros(
            out_image.shape[1:], dtype=np.uint8
        )  # Create a mask with zeros
        mask_array[out_image[0] > 0] = (
            1  # Set 1 for all pixels corresponding to buildings
        )

        # Parameters for saving the mask
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mask_array.shape[0],
                "width": mask_array.shape[1],
                "count": 1,  # Single channel
                "dtype": "uint8",
                "transform": out_transform,
            }
        )

        # Create the mask folder if it does not exist
        os.makedirs(mask_folder, exist_ok=True)

        # Save the mask in the specified folder
        output_mask_path = os.path.join(
            mask_folder, f"mask_{os.path.basename(tif_path)}"
        )
        with rasterio.open(output_mask_path, "w", **out_meta) as dest:
            dest.write(mask_array, 1)  # Write the mask to the first channel

        return output_mask_path
