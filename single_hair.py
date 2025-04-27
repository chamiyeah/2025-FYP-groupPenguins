from os.path import join
import cv2
from util.img_util import readImageFile, saveImageFile
from util.inpaint_util import removeHair

file_path = './data/skin_images/single/example.jpg'
save_dir = './result/hair'

# read the image file
img_rgb, img_gray = readImageFile(file_path)

# apply hair removal
blackhat, thresh, img_out = removeHair(img_rgb, img_gray, kernel_size=15, threshold=10)

# save the output image
save_file_path = join(save_dir, 'output.jpg')
saveImageFile(img_out, save_file_path)