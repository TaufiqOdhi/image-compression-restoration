from utils import bit8to4, bit4to8
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import make_grid
import pydicom
import cv2
from PIL import Image
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from compress import compress_binary
import os


from model.prune_random_unstructured_global import GeneratorPruned
CKPT_PATH = "/mnt/Windows/Users/taufi/MyFile/Projects/image-compression-restoration/04_1_binaryCompression_srganRecovery_layerWisePruning/checkpoints/randomUnstructuredGlobal/176-epoch-pruned-model/30/gen.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_ORIGINAL_LR_PATH = "/mnt/Windows/Users/taufi/MyFile/Projects/datasets/DIV2K_valid_LR_bicubic/X4/0805x4.png"
# IMAGE_ORIGINAL_LR_PATH = "../../datasets/dicom_images_kaggle/ID_0a0adf93f.dcm"
IMAGE_ORIGINAL_HR_PATH = "/mnt/Windows/Users/taufi/MyFile/Projects/datasets/DIV2K_valid_HR/0805.png"
RESULT_DIR_PATH = "./hasil/"

gen_global = GeneratorPruned() # if use pruned model
gen_global.load_state_dict(torch.load(CKPT_PATH)["state_dict"])
gen_global.eval().to(DEVICE)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

def recovery_srgan(img, gen=gen_global):
    with torch.no_grad():
        upscaled_img = gen(
            test_transform(image=np.asarray(img))["image"]
            .unsqueeze(0)
            .to(DEVICE)
        )
        upscaled_img = upscaled_img * 0.5 + 0.5
        upscaled_img = make_grid(upscaled_img)
        upscaled_img = upscaled_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        return upscaled_img


def split(img):
    bottom_img = bit8to4(img)
    top_img = img - bottom_img*16
    return np.vstack((top_img, bottom_img))


def recovery_binary(img):
    split_img = split(img)
    img_8_bit = bit4to8(split_img)

    return img_8_bit


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def to_hr():
    image_original_lr = np.asarray(Image.open(IMAGE_ORIGINAL_LR_PATH))
    image_original_hr = np.asarray(Image.open(IMAGE_ORIGINAL_HR_PATH))
    image_recovery_srgan = recovery_srgan(image_original_lr)
    print(f"PSNR HR Image: {psnr(image_original_hr, image_recovery_srgan)}")
    Image.fromarray(image_recovery_srgan).save(f"{RESULT_DIR_PATH}hr_image(from original image).png")


if __name__ == '__main__':
    image_original_lr = np.asarray(Image.open(IMAGE_ORIGINAL_LR_PATH))
    # image_original_lr = cv2.cvtColor(pydicom.dcmread(IMAGE_ORIGINAL_LR_PATH).pixel_array, cv2.COLOR_GRAY2RGB) # untuk load data dicom
    image_original_hr = np.asarray(Image.open(IMAGE_ORIGINAL_HR_PATH))
    

    # untuk load data dicom
    # ds=pydicom.dcmread(IMAGE_ORIGINAL_LR_PATH)
    # img=ds.pixel_array
    # image = np.asarray(img)
    image = np.asarray(Image.open(IMAGE_ORIGINAL_LR_PATH)) # untuk load gambar rgb biasa
    
    image = compress_binary(image_original_lr) 
    image_recovery_binary = recovery_binary(image)
    image_recovery_srgan = recovery_srgan(image_original_lr)

    print(f"PSNR LR Image: {psnr(image_original_lr, image_recovery_binary)}")
    print(f"PSNR HR Image: {psnr(image_original_hr, image_recovery_srgan)}")
    print(f"SSIM HR Image: {ssim(image_original_hr, image_recovery_srgan, channel_axis=2)}")

    Image.fromarray(image_original_lr).save(f"{RESULT_DIR_PATH}original_image.png")
    Image.fromarray(image).save(f"{RESULT_DIR_PATH}compressed_image.png")
    Image.fromarray(image_recovery_binary).save(f"{RESULT_DIR_PATH}decompressed_image.png")
    Image.fromarray(image_recovery_srgan).save(f"{RESULT_DIR_PATH}hr_image.png")
    os.system('nvidia-smi >> vram_consumption.txt')
