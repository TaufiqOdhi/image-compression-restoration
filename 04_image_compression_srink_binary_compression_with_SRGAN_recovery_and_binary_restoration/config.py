import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

#First Train using all loss func until Epoch 176

TRAIN_PATH = "../../datasets/DIV2K_train_HR/"
# TRAIN_PATH = "../../datasets/dicom_images_kaggle/"
# TRAIN_PATH = "results/dicom_to_jpg/"
# TRAIN_PATH = "../../datasets/Mri_Brain_Contrast-663058577/3D_Brain_FLAIR_201"

VAL_PATH = "../../datasets/DIV2K_valid_LR_bicubic/X4/"
# VAL_PATH = "../../datasets/dicom_images_kaggle_valid/"
# VAL_PATH = "results/dicom_to_jpg/"
# VAL_PATH = "../../datasets/Mri_Brain_Contrast-663058577/3D_Brain_T2_301"

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
START_EPOCHS = 1
NUM_EPOCHS = 176
BATCH_SIZE = 2
NUM_WORKERS = 4
HIGH_RES = 224
RATIO = 4
LOW_RES = HIGH_RES // RATIO
IMG_CHANNELS = 3

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        # A.Normalize(mean=[0], std=[1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        # A.Normalize(mean=[0], std=[1]),
        ToTensorV2(),
    ]
)
