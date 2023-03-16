from config import DEVICE
from PIL import Image
from compress import compress_binary
from recovery import recovery_binary, recovery_srgan
import numpy as np
import torch
import datetime


base_dir = '04_1_binaryCompression_srganRecovery_layerWisePruning/test_recovery/input_image'
image_1 = f'{base_dir}/1.png'
image_2 = f'{base_dir}/2.png'
image_3 = f'{base_dir}/3.png'
image_4 = f'{base_dir}/4.png'
image_5 = f'{base_dir}/5.png'
image_6 = f'{base_dir}/6.png'


def srgan(gen, image, CKPT_PTH, filename, srgan_type):
    file_path = f'04_1_binaryCompression_srganRecovery_layerWisePruning/hasil/{filename}_{srgan_type}_{datetime.datetime.now()}.png'
    gen.load_state_dict(torch.load(CKPT_PTH)["state_dict"])
    gen.eval().to(DEVICE)

    input_image = np.asarray(Image.open(image))
    input_image = compress_binary(input_image)
    input_image = recovery_binary(input_image)
    input_image = recovery_srgan(img=input_image, gen=gen)
    Image.fromarray(input_image).save(file_path)
