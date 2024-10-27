from preprocessing import preprocess_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np


class CustomDataset(Dataset):
    """
    Custom Dataset class to handle the data properly
    """

    def __init__(self, dataloader):
        """
        Args:
            dataloader: Original dataloader containing (image, mask, meta)
        """
        self.data = []
        # Convert iterator to list first
        data_list = list(dataloader)

        for item in data_list:
            if len(item) == 3:  # Ensure we have image, mask, and meta
                image, mask, meta = item
                # Make contiguous copies of arrays
                if isinstance(image, np.ndarray):
                    image = np.ascontiguousarray(image.copy())
                if isinstance(mask, np.ndarray):
                    mask = np.ascontiguousarray(mask.copy())
                self.data.append((image, mask, meta))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask, meta = self.data[idx]

        try:
            # Convert to torch tensors with explicit copying
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(np.ascontiguousarray(image)).float()
            elif isinstance(image, torch.Tensor):
                image = image.clone().float()

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
            elif isinstance(mask, torch.Tensor):
                mask = mask.clone().float()

            # Ensure correct dimensions
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            return image, mask, meta

        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            print(
                f"Image shape: {image.shape if hasattr(image, 'shape') else 'No shape'}"
            )
            print(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
            raise e


class Models:
    def __init__(self):
        """
        Initialize the Models class with data loaders and model parameters
        """
        # Initialize original dataloaders
        original_train_ndwi, original_train_ndbi = preprocess_data(
            big_images_path="train/images",
            big_masks_path="train/masks",
            train=True,
            image_for_model_size=640,
        )

        # Create custom dataset
        custom_dataset = CustomDataset(original_train_ndwi)

        # Create new dataloader with custom dataset
        self.train_ndwi_dataloader = DataLoader(
            custom_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True
        )

    def _initialize_water_model(self):
        """
        Initialize U-Net model for water segmentation
        """
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        return model.to(self.device)

    def segment_water(self, training_epochs=50):
        """
        Train model for water segmentation
        """
        self.water_model.train()
        optimizer = optim.Adam(self.water_model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(training_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                self.train_ndwi_dataloader, desc=f"Epoch {epoch+1}/{training_epochs}"
            )

            for batch_idx, (images, masks, _) in enumerate(progress_bar):
                # Ensure correct shape and type
                images = images.float().to(self.device)
                masks = masks.float().to(self.device)

                optimizer.zero_grad()
                outputs = self.water_model(images)

                # Ensure matching dimensions
                if outputs.shape != masks.shape:
                    masks = masks.unsqueeze(1)  # Add channel dimension if needed

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (batch_idx + 1)})

            print(
                f"Epoch {epoch+1}, Loss: {epoch_loss/len(self.train_ndwi_dataloader):.4f}"
            )

        return self.water_model

    def predict_water(self, image):
        """
        Predict water segmentation for a single image

        Args:
            image (torch.Tensor): Input image tensor

        Returns:
            numpy.ndarray: Binary mask of water segmentation
        """
        self.water_model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.water_model(image)
            pred = torch.sigmoid(output) > 0.5
            return pred.cpu().numpy().squeeze()


# Example usage script
def main():
    # 1. Initialize the Models class
    model_handler = Models()

    # 2. Train the water segmentation model
    trained_model = model_handler.segment_water(training_epochs=50)

    # 3. Save the trained model (optional but recommended)
    torch.save(trained_model.state_dict(), "water_segmentation_model.pth")

    # 4. Example of making predictions (if you have a test image)
    # Assuming you have a test image prepared as a tensor
    test_image = next(iter(model_handler.train_ndwi_dataloader))[0][
        0
    ]  # Get first image from dataloader
    prediction = model_handler.predict_water(test_image)

    # 5. Visualize results (optional)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.permute(1, 2, 0))  # Convert from CHW to HWC for display
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap="gray")
    plt.title("Water Segmentation")
    plt.show()


if __name__ == "__main__":
    main()
