import cv2


# def removeHair(img_org, img_gray, kernel_size=25, threshold=10, radius=3):
#     # kernel for the morphological filtering
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

#     # perform the blackHat filtering on the grayscale image to find the hair countours
#     blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

#     # intensify the hair countours in preparation for the inpainting algorithm
#     _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

#     # inpaint the original image depending on the mask
#     img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

#     return blackhat, thresh, img_out

# combined tophat and blackhat - fine results, blurry
def removeHair(img_org, img_gray, kernel_size=15, threshold=10, radius=3):
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    #blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    #repeat on white hair
    img_gray_2 = cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY)
    tophat = cv2.morphologyEx(img_gray_2, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    img_out_2 = cv2.inpaint(img_out, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out_2