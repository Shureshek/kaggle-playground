import pickle
import torch
import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms
from torchvision.transforms import v2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset

# разные режимы датасета
DATA_MODES = ['train', 'val', 'test']

# работаем на видеокарте
DEVICE = torch.device("cuda")

#определим директории с тренировочными и тестовыми файлами
TRAIN_DIR = Path('content/train') #Path('./data/train/')
TEST_DIR = Path('content/testset') #Path('./data/testset')

# параметры нормировки изображений по трем каналам перед подачей в модель
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = [224, 224]

class SimpsonsDataset(Dataset):
    def __init__(self, files, label_encoder, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode
        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.label_encoder = label_encoder
        self.len_ = len(self.files)

    def __len__(self):
        return self.len_ # сейчас self.__len__() возвращает количество картинок, подаваемых на вход.
        # Если вы решите перевзвесить размеры категорий внутри класса -
        # не забудьте изменить вывод self.__len__()

    def __getitem__(self, index):
        x = self.load_image(self.files[index])
        x = self.transform_images_to_tensors(x)

        if self.mode == 'test':
            return x
        else:
            path = self.files[index]
            y = self.label_encoder.transform([path.parent.name,]).item()
            return x, y

    # принимает путь к файлу изображения и возвращает само изображение
    def load_image(self, file):
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # преобразует изображение в тензор
    def transform_images_to_tensors(self, image):
        if self.mode == 'train':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1, 
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7
                ),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=0.1,
                    rotate=(-10, 10),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.08,
                    p=0.7
                ),
                A.Resize(RESCALE_SIZE[0], RESCALE_SIZE[1]),
                A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                    A.Resize(RESCALE_SIZE[0], RESCALE_SIZE[1]),
                    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ToTensorV2(),
                ])
        transformed  = transform(image=image)
        return transformed ['image']


