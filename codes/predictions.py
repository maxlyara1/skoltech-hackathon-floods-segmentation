import rasterio
from preprocessing import preprocess_data
from preprocessing import get_sorted_data_list
from formatting import preprocess_NDWI_format, preprocess_NDBI_format
from models import segment_water
from models import segment_buildings


def get_predictions(a):
    big_images_files = get_sorted_data_list()
