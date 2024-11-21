import cv2
import numpy as np
from PIL import Image
from docuwarp.unwarp import Unwarp
import os

# Define folder paths
input_folder = r"docuwarp\img\Original"
output_folder = os.path.join(input_folder, "adaptive_enhanced_unwarped")
os.makedirs(output_folder, exist_ok=True)

# Initialize the unwarp model
unwarper = Unwarp()

# Function to estimate noise level
def estimate_noise(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.var(image_gray)  # ใช้ค่า variance เพื่อวัดระดับ noise

# Loop through and adaptively enhance each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        
        # Load and unwarp the image
        image = Image.open(image_path)
        try:
            unwarped_image = unwarper.inference(image)
            
            # Convert PIL image to OpenCV format
            unwarped_cv = np.array(unwarped_image)
            unwarped_cv = cv2.cvtColor(unwarped_cv, cv2.COLOR_RGB2BGR)
            
            # ประเมินระดับ noise
            noise_level = estimate_noise(unwarped_cv)
            
            # Adaptive noise reduction
            if noise_level > 500:  # Noise สูง
                denoised_image = cv2.bilateralFilter(unwarped_cv, d=9, sigmaColor=100, sigmaSpace=100)
            elif noise_level > 200:  # Noise ปานกลาง
                denoised_image = cv2.bilateralFilter(unwarped_cv, d=9, sigmaColor=75, sigmaSpace=75)
            else:  # Noise ต่ำ
                denoised_image = cv2.bilateralFilter(unwarped_cv, d=9, sigmaColor=50, sigmaSpace=50)
            
            # Adaptive sharpening based on detail level (sharpen less if high detail)
            if noise_level > 500:
                sharpening_kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])  # เพิ่ม sharpening
            else:
                sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # ปกติ

            sharpened_image = cv2.filter2D(denoised_image, -1, sharpening_kernel)
            
            # Convert back to PIL format and save
            final_image = Image.fromarray(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
            output_path = os.path.join(output_folder, f"adaptive_enhanced_{filename}")
            final_image.save(output_path)
            print(f"Adaptive enhanced and saved: {output_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
