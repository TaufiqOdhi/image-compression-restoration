from utils import *
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Generator
from torchvision.utils import make_grid
import pydicom
import cv2
from math import log10, sqrt

# CKPT_PATH = "gen.pth.tar"
CKPT_PATH = "checkpoints/170-an-epoch/gen.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# gen = Generator(in_channels=1)
gen = Generator()
gen.load_state_dict(torch.load(CKPT_PATH)["state_dict"])
gen.eval().to(DEVICE)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        # A.Normalize(mean=[0], std=[1]),
        ToTensorV2(),
    ]
)

def recovery_srgan(img):
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


def combinebit(img_4_bit):
    h,w, _ = img_4_bit.shape
    # h,w = img_4_bit.shape

    if h % 2 == 0:
        top_img = img_4_bit[:h//2]
        bottom_img = img_4_bit[h//2:]

    else:
        raise ValueError("height shape should be even, please resize your height of image into become even number")

    return top_img+(bottom_img*16)


def compress(img):
    img_4_bit = bit8to4(img)
    img_combined = combinebit(img_4_bit)

    return img_combined


if __name__ == '__main__':
    from PIL import Image

    IMAGE_ORIGINAL_LR_PATH = "../../datasets/DIV2K_valid_LR_bicubic/X4/0805x4.png"
    # IMAGE_ORIGINAL_LR_PATH = "../../datasets/dicom_images_kaggle/ID_0a0adf93f.dcm"
    IMAGE_ORIGINAL_HR_PATH = "../../datasets/DIV2K_valid_HR/0805.png"
    # IMAGE_PATH = "../03_image_compression_srink_binary_compression_with_SRGAN_recovery/asset/Compressed/result.png"
    # IMAGE_PATH = "../02_binary_compression/sample_image/compressed_image.png"
    RESULT_PATH = "results/result6_2.png"

    image_original_lr = cv2.imread(IMAGE_ORIGINAL_LR_PATH)
    # image_original_lr = cv2.cvtColor(pydicom.dcmread(IMAGE_ORIGINAL_LR_PATH).pixel_array, cv2.COLOR_GRAY2RGB)
    image_original_hr = cv2.imread(IMAGE_ORIGINAL_HR_PATH)

    # ds=pydicom.dcmread(IMAGE_PATH)
    # img=ds.pixel_array
    # image = np.asarray(img)
    # image = np.asarray(Image.open(IMAGE_PATH))
    image = compress(image_original_lr)

    image_recovery_binary = recovery_binary(image)
    # image_recovery_srgan = recovery_srgan(image_recovery_binary)
    image_recovery_srgan = recovery_srgan(image_original_lr)

    print(f"PSNR: {psnr(image_original_lr, image_recovery_binary)}")
    print(f"PSNR: {psnr(image_original_hr, image_recovery_srgan)}")

    # image = Image.fromarray(image)
    # image.save(RESULT_PATH)
    # Image.fromarray(image_original_lr).save("results/original_image4.png")
    # Image.fromarray(image).save("results/compressed_image4.png")
    # Image.fromarray(image_recovery_binary).save("results/decompressed_image4.png")
    Image.fromarray(image_recovery_srgan).save(RESULT_PATH)

    # cv2.imshow("result", image)
    # cv2.waitKey(0)