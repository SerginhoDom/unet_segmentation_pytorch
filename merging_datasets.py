import os
import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils

# Параметры
ANNOTATION_FILE = '../nail_segmentation/dataset/images_test/_annotations.coco.json'  # путь к вашему COCO JSON
OUTPUT_MASK_DIR = '../nail_segmentation/dataset/labels_test'  # путь к папке для сохранения масок
IMAGE_SIZE = (640, 640)  # ширина и высота изображений

# Создаем папку для масок, если она не существует
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def create_masks_from_coco(annotation_file, output_dir, image_size):
    # Загружаем COCO JSON
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Размеры изображений
    width, height = image_size
    
    # Словарь с id изображений и их названиями
    images_info = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Словарь для группировки аннотаций по id изображений
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # Перебираем все изображения
    for image_id, annotations in annotations_by_image.items():
        # Создаем пустую чёрно-белую маску для изображения
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Перебираем аннотации для данного изображения
        for annotation in annotations:
            segmentation = annotation['segmentation']
            
            if isinstance(segmentation, list):  # Многоугольники
                for poly in segmentation:
                    points = np.array(poly).reshape(-1, 2)
                    points = np.round(points * [width / 640, height / 640]).astype(np.int32)
                    
                    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
                        print(f"Пропуск многоугольника с некорректными координатами в изображении {image_id}")
                        continue
                    
                    cv2.fillPoly(mask, [points], 255)
            
            elif isinstance(segmentation, dict):  # RLE
                rle = maskUtils.frPyObjects(segmentation, height, width)
                rle_mask = maskUtils.decode(rle) * 255
                mask = np.maximum(mask, rle_mask)
        
        if np.sum(mask) == 0:
            print(f"Маска пустая для изображения {image_id}. Пропускаем.")
            continue
        
        # Используем имя исходного изображения
        original_filename = images_info[image_id]
        output_mask_file = os.path.join(output_dir, original_filename)
        
        # Сохраняем маску в формате JPEG
        cv2.imwrite(output_mask_file, mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f'Маска сохранена: {output_mask_file}')

# Пример использования
create_masks_from_coco(ANNOTATION_FILE, OUTPUT_MASK_DIR, IMAGE_SIZE)