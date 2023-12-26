import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

selected_filter = '0' 
brightness_factor = 1.0 
enhanced_img = None  

def enhance_and_adjust_brightness(image_path):
    global enhanced_img
    original_img = cv2.imread(image_path)
    enhanced_img = improve_quality(original_img)

    cv2.imshow('Original vs. Adjusted', np.hstack((original_img, enhanced_img)))

    def on_brightness_change(val):
        global brightness_factor  
        brightness_factor = float(val) / 50.0  
        adjusted_img = apply_filter(enhanced_img, selected_filter)
        adjusted_img = increase_brightness_internal(adjusted_img, brightness_factor)

        comparison_img = np.hstack((original_img, adjusted_img))
        cv2.imshow('Original vs. Adjusted', comparison_img)

        percentage_change = ((adjusted_img - original_img) / original_img) * 100
        print(f"Percentage change: {np.mean(percentage_change):.2f}%")

    cv2.namedWindow('Original vs. Adjusted')
    cv2.createTrackbar('Brightness', 'Original vs. Adjusted', 50, 200, on_brightness_change)

    on_brightness_change(50)

    save_path = None  

    filters = {
        '0': 'Enhanced Image Without Filters',
        '1': 'Increased Sharpness',
        '2': 'High Pass Filter',
        '3': 'Sepia Tone',
        '4': 'Vintage',
        '5': 'Intense Blurring Filter',
    }

    print("Select an enhancing filter:")
    for key, value in filters.items():
        print(f"{key}: {value}")

    keys_manual = "Keys Manual:\n\n"
    keys_manual += "Enter Key: Save Adjusted Image\n"
    keys_manual += "Escape Key: Exit\n"
    for key, action in filters.items():
        keys_manual += f"{key}: Select {action}\n"

    print(keys_manual)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  
            save_path = filedialog.asksaveasfilename(title="Save Adjusted Image As", defaultextension=".png",
                                                      filetypes=[("PNG files", "*.png")])
            break
        elif key == 27:  
            break
        elif chr(key) in filters:
            global selected_filter
            selected_filter = chr(key)
            print(f"Selected filter: {filters[selected_filter]}")
            on_brightness_change(cv2.getTrackbarPos('Brightness', 'Original vs. Adjusted'))

    cv2.destroyAllWindows()

    if save_path:
        adjusted_img = apply_filter(enhanced_img, selected_filter)
        adjusted_img = increase_brightness_internal(adjusted_img, brightness_factor)
        cv2.imwrite(save_path, adjusted_img)
        print(f"Adjusted image saved at: {save_path}")

def improve_quality(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    sharpened_img = cv2.filter2D(denoised_img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    return sharpened_img

def increase_brightness_internal(img, factor):
    b_channel, g_channel, r_channel = cv2.split(img)

    b_channel = np.clip(factor * b_channel, 0, 255).astype(np.uint8)
    g_channel = np.clip(factor * g_channel, 0, 255).astype(np.uint8)
    r_channel = np.clip(factor * r_channel, 0, 255).astype(np.uint8)
    adjusted_img = cv2.merge([b_channel, g_channel, r_channel])

    return adjusted_img

def apply_filter(img, filter_key):
    if filter_key == '0':
        return enhanced_img
    elif filter_key == '1':
        return apply_increased_sharpness(img)
    elif filter_key == '2':
        return apply_high_pass_filter(img)
    elif filter_key == '3':
        return apply_sepia_tone(img)
    elif filter_key == '4':
        return apply_vintage_filter(img)
    elif filter_key == '5':
        return apply_intense_blurring_filter(img)

def apply_increased_sharpness(img):
    return cv2.filter2D(img, -1, np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]))

def apply_high_pass_filter(img):
    blur_img = cv2.GaussianBlur(img, (25, 25), 0)
    high_pass_img = cv2.subtract(img, blur_img)
    return high_pass_img

def apply_sepia_tone(img):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img, sepia_matrix)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img

def apply_vintage_filter(img):
    vintage_matrix = np.array([[0.8, 0.4, 0.1],
                               [0.4, 0.8, 0.1],
                               [0.2, 0.5, 0.8]])
    vintage_img = cv2.transform(img, vintage_matrix)
    vintage_img = np.clip(vintage_img, 0, 255).astype(np.uint8)
    return vintage_img

def apply_intense_blurring_filter(img):
    blur_kernel = np.array([[0.0625, 0.125, 0.0625],
                            [0.125, 0.25, 0.125],
                            [0.0625, 0.125, 0.0625]])
    return cv2.filter2D(img, -1, blur_kernel)

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("All files", "*.*")])

    if not file_path:
        print("No image selected. Exiting.")
        return

    enhance_and_adjust_brightness(file_path)

if __name__ == "__main__":
    main()
