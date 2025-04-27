from os.path import join

import matplotlib.pyplot as plt

from util.img_util import readImageFile, saveImageFile, ImageDataLoader
from util.inpaint_util import removeHair

directory = './data/skin_images/test'
data = ImageDataLoader(directory, shuffle=False, transform=None)

save_dir = "./result/hair_removed"

for file_name in data.file_list: 
    full_path = join(directory, file_name)
    img_rgb, img_gray = readImageFile(full_path)

    blackhat, thresh, img_out = removeHair(img_rgb, img_gray, kernel_size=12, threshold=10)

    output_filename = f"output_{file_name}.jpg"
    save_file_path = join(save_dir, output_filename)
    
    saveImageFile(img_out, save_file_path)
    
    print(f"Saved: {save_file_path}")