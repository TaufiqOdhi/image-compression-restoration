from math import log10, sqrt
import cv2
import numpy as np
import pydicom


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr



def bit8to4(img):
    return (img//16).astype(np.uint8)


def bit4to8(img):
    return img*16


def combinebit(img_4_bit):
    h,w, _ = img_4_bit.shape
    # h,w = img_4_bit.shape

    print(img_4_bit.shape)

    if h % 2 == 0:
        top_img = img_4_bit[:h//2]
        bottom_img = img_4_bit[h//2:]

    else:
        raise ValueError("height shape should be even, please resize your height of image into become even number")

    return top_img+(bottom_img*16)


def split(img):
    bottom_img = bit8to4(img)
    top_img = img - bottom_img*16
    return np.vstack((top_img, bottom_img))


def compress(img):
    img_4_bit = bit8to4(img)
    img_combined = combinebit(img_4_bit)

    return img_combined


def decompress(img):
    split_img = split(img)
    img_8_bit = bit4to8(split_img)

    return img_8_bit

if __name__ == '__main__':
    # image_sample="sample_image/Grayscale_Cat.jpg"
    # image_sample="../../datasets/DIV2K_valid_LR_bicubic/X4/0805x4.png"
    # image_sample="sample_image/ID_080056891.dcm"
    # image_sample="sample_image/ID_0a0adf93f.jpeg"
    image_sample="../../datasets/dicom_images_kaggle/ID_0a0adf93f.dcm"
    
    # img = cv2.imread(image_sample)
    ds=pydicom.dcmread(image_sample)
    img=ds.pixel_array
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    img_compressed = compress(img)
    img_decompressed = decompress(img_compressed)

    print(f"PSNR: {psnr(img, img_decompressed)}")

    cv2.imwrite("sample_image/original_image4.png", img)
    cv2.imwrite("sample_image/compressed_image4.png", img_compressed)
    cv2.imwrite("sample_image/img_decompressed4.png", img_decompressed)

    # pydicom.dcmwrite("sample_image/original_image.dcm", img)
    # pydicom.dcmwrite("sample_image/compressed_image.dcm", img_compressed)
    # pydicom.dcmwrite("sample_image/img_deompressed.dcm", img_decompressed)


    # cv2.imshow("original_image", img)
    # cv2.imshow("compressed_image", img_compressed)
    # cv2.imshow("img_decompressed", img_decompressed)

    # cv2.waitKey(0)