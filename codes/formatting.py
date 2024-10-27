def calculate_ndwi(green_band, nir_band):
    """
    Calculate the Normalized Difference Water Index (NDWI).

    NDWI is used to monitor changes related to water content in water bodies.

    Parameters:
    green_band (array-like): The green band of the image.
    nir_band (array-like): The near-infrared (NIR) band of the image.

    Returns:
    array-like: The NDWI values calculated for each pixel.
    """
    return (green_band - nir_band) / (green_band + nir_band)


def calculate_ndbi(nir_band, swir_band):
    """
    Calculate the Normalized Difference Built-up Index (NDBI).

    NDBI is used to map built-up areas.

    Parameters:
    nir_band (array-like): The near-infrared (NIR) band of the image.
    swir_band (array-like): The shortwave infrared (SWIR) band of the image.

    Returns:
    array-like: The NDBI values calculated for each pixel.
    """
    return (swir_band - nir_band) / (swir_band + nir_band)


def preprocess_NDWI_format(image):
    """
    Preprocess the image to calculate NDWI.

    This function assumes that the image has channels in a specific order:
    - Green band is at index 2
    - NIR band is at index 8

    Parameters:
    image (array-like): The input image with multiple spectral bands.

    Returns:
    array-like: The NDWI values calculated for each pixel in the image.
    """
    # Extract the green and NIR bands from the image
    green_band = image[2]
    nir_band = image[8]

    # Calculate NDWI using the extracted bands
    ndwi = calculate_ndwi(green_band, nir_band)

    return ndwi


def preprocess_NDBI_format(image):
    """
    Preprocess the image to calculate NDBI and identify built-up areas.

    This function assumes that the image has channels in a specific order:
    - NIR band is at index 7
    - SWIR band is at index 9

    Parameters:
    image (array-like): The input image with multiple spectral bands.

    Returns:
    array-like: A binary array where True indicates built-up areas.
    """
    # Extract the NIR and SWIR bands from the image
    nir_band = image[7]
    swir_band = image[9]

    # Calculate NDBI using the extracted bands
    ndbi = calculate_ndbi(nir_band, swir_band)

    # Binarize NDBI to identify built-up areas
    threshold = 0.1
    built_up_areas = ndbi > threshold

    return built_up_areas
