from test_recovery import srgan, image_1, image_2, image_3, image_4, image_5, image_6
from model import Generator
import os
import datetime


base_dir_checkpoint = '/mnt/Windows/Users/taufi/MyFile/Projects/image-compression-restoration/04_1_binaryCompression_srganRecovery_layerWisePruning/checkpoints'


def test_no_prune_1():
    filename = '1'
    CKPT_PTH = f'{base_dir_checkpoint}/no_prune/gen.pth.tar'
    gen = Generator()
    srgan(
        gen=gen,
        image=image_1,
        CKPT_PTH=CKPT_PTH,
        filename=filename,
        srgan_type='noPrune'
    )
    os.system(f'nvidia-smi > "04_1_binaryCompression_srganRecovery_layerWisePruning/vram_logs/no_prune/vram_consumption_{filename}_noPrune_{datetime.datetime.now()}.txt"')
