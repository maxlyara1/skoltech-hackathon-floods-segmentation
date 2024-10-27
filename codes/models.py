import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from preprocessing import preprocess_data


class Models:
    def __init__(self):
        # Инициализация загрузчика данных NDWI для тренировки
        self.train_ndwi_dataloader, _ = preprocess_data(
            big_images_path="train/images",
            big_masks_path="train/masks",
            train=True,
            image_for_model_size=640,
        )
        # Сохранение данных из загрузчика на диск
        self.save_dataloader_to_disk()

    def save_dataloader_to_disk(self):
        """Сохранение изображений и масок из dataloader на диск."""
        os.makedirs("train/saved_images", exist_ok=True)
        os.makedirs("train/saved_masks", exist_ok=True)

        for i, (images, masks) in enumerate(self.train_ndwi_dataloader):
            try:
                # Копирование массива для предотвращения ошибки с отрицательными strides
                image_np = images[0].numpy().copy().astype("uint8")
                mask_np = masks[0].numpy().copy().astype("uint8")

                # Сохранение изображений
                image_path = f"train/saved_images/image_{i}.tif"
                cv2.imwrite(image_path, image_np)

                # Сохранение масок
                mask_path = f"train/saved_masks/mask_{i}.tif"
                cv2.imwrite(mask_path, mask_np)
            except Exception as e:
                print(f"Ошибка при сохранении данных из dataloader: {e}")

    def check_images(self, path):
        """Проверка наличия и целостности изображений в заданной папке."""
        valid_images = []
        for filename in os.listdir(path):
            if filename.endswith(".tif"):
                img_path = os.path.join(path, filename)
                if cv2.imread(img_path) is not None:
                    valid_images.append(img_path)
                else:
                    print(f"WARNING: Ignoring corrupt image file: {img_path}")
        return valid_images

    def segment_water(self, epochs=3, device="cpu"):
        # Конвертация масок в формат YOLO Seg
        convert_segment_masks_to_yolo_seg(
            masks_dir="train/saved_masks",  # Путь к папке с масками
            output_dir="train/yolo_data",  # Папка для сохранения YOLO-формата
            classes=2,  # Количество классов
        )

        # Проверка наличия валидных изображений
        valid_images = self.check_images("train/saved_images")
        if not valid_images or not os.listdir("train/saved_masks"):
            print(
                "Папка с изображениями или масками пуста или содержит поврежденные файлы!"
            )
            return

        # Создание конфигурации yaml для использования в модели YOLO
        yaml_content = f"""
        train: {os.path.abspath('train/saved_images')}  # путь к обучающим изображениям
        val: {os.path.abspath('train/saved_images')}  # путь к проверочным изображениям
        nc: 2  # число классов
        names: ['Flood', 'Nothing']  # названия классов
        """

        # Запись в файл
        os.makedirs("train/yolo_data", exist_ok=True)
        with open("train/yolo_data/annotations.yaml", "w") as f:
            f.write(yaml_content)

        # Загрузка и тренировка модели YOLO
        model = YOLO("yolo11n-seg.pt")
        model.train(
            data="train/yolo_data/annotations.yaml",
            epochs=epochs,
            device=device,
        )

        # Получение прогнозов на изображениях
        self.predict_flood(model)

    def predict_flood(self, model):
        # Пример прогноза на изображении из тестового набора
        predictions = model.predict(
            "data/images/tile_2_0.tif", save=True
        )  # Укажите путь к тестовому изображению
        for pred in predictions:
            print(pred)  # Отображение каждого прогноза


if __name__ == "__main__":
    models = Models()
    models.segment_water(
        epochs=3, device="cuda" if torch.cuda.is_available() else "cpu"
    )
