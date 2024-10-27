from preprocessing import preprocess_data


class Models:
    def __init__(self):
        self.train_ndwi_dataloader, self.train_ndbi_dataloader = preprocess_data(
            big_images_path="train/images",
            big_masks_path="train/masks",
            train=True,
            image_for_model_size=640,
        )

    def segment_water(self, a):
        pass

    def segment_buildings(self, a):
        pass
