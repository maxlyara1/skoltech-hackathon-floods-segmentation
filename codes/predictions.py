import rasterio
from preprocessing import preprocess_data
from formatting import preprocess_NDWI_format, preprocess_NDBI_format
from models.Models import segment_water
from models.Models import segment_buildings


def get_predictions(a):

    test_ndwi_dataloader, test_ndbi_dataloader = preprocess_data(
        big_images_path="train/images",
        big_masks_path=None,
        train=False,
        image_for_model_size=640,
    )
    pass
