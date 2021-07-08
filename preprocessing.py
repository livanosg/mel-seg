import cv2
import numpy as np
from hair_removal import remove_and_inpaint

image = cv2.imread("/home/livanosg/projects/mel-seg/data/ph2/data/IMD014/IMD014_Dermoscopic_Image/IMD014.bmp")
label = cv2.imread("/home/livanosg/projects/mel-seg/data/ph2/data/IMD014/IMD014_lesion/IMD014_lesion.bmp")


def mask_resize_pad_image(image, mask, scale_factor):
    mask = mask / np.max(mask)  # Range 0-1
    indices = np.nonzero(mask)  # Get non zero pixel positions mask limits
    top, bot, left, right = np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1])
    height, width = bot - top, right - left  # Define dimensions of square mask
    scaled_height, scaled_width = height * scale_factor, width * scale_factor  # Scale rectangular mask
    hd_height, hd_width = int((scaled_height - height) // 2), int((scaled_width - width) // 2)  # compute half scaling
    mask[top - hd_height:bot + hd_height, left - hd_width:right + hd_width, :] = 1.  # Create rectangular scaled mask
    cropped_image = image[top - hd_height:bot + hd_height, left - hd_width:right + hd_width]  # Crop image to mask size
    crop_height, crop_width = cropped_image.shape[0], cropped_image.shape[1]
    if crop_height > crop_width:
        scale_ratio = 224. / crop_height
    else:
        scale_ratio = 224. / crop_width
    resized_image = cv2.resize(src=cropped_image, dsize=(int(crop_width * scale_ratio), int(crop_height * scale_ratio)))

    if resized_image.shape[0] > resized_image.shape[1]:
        padding = (resized_image.shape[0] - resized_image.shape[1]) // 2
        resized_image = cv2.copyMakeBorder(src=resized_image, top=0, bottom=0, left=padding, right=padding,
                                           borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        padding = (resized_image.shape[1] - resized_image.shape[0]) // 2
        resized_image = cv2.copyMakeBorder(src=resized_image, top=padding, bottom=padding, left=0, right=0,
                                           borderType=cv2.BORDER_CONSTANT, value=0)

    return resized_image


res_img = mask_resize_pad_image(image=image, mask=label, scale_factor=1.1)
hairless_image, steps = remove_and_inpaint(image=res_img)

cv2.imshow("before", image)
cv2.imshow("cropped", res_img)
cv2.imshow("hairless", hairless_image)
cv2.waitKey()
exit()
