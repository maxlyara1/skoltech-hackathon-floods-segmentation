import rasterio
from preprocessing import preprocess_data
from formatting import preprocess_NDWI_format, preprocess_NDBI_format
from models import segment_water
from models import segment_buildings


def get_predictions(a):
    # train_dataloader = preprocess_data(
    #     big_images_path="train/images",
    #     big_masks_path="train/masks",
    #     train=True,
    #     image_for_model_size=640,
    # )

    test_dataloader = preprocess_data(
        big_images_path="train/images",
        big_masks_path=None,
        train=False,
        image_for_model_size=640,
    )
