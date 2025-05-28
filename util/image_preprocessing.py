import cv2

def enhance_color_hsv_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_enhanced = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    return enhanced_image