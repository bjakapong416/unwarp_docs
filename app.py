import os
from PIL import Image
from docuwarp.unwarp import Unwarp

# Define the folder containing the images
folder_path = r"C:\unwarp\docuwarp\img\Original"
output_folder = os.path.join(folder_path, "unwarped")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize the unwarp model
unwarper = Unwarp()

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Process only files with image extensions
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)
        
        # Load the image
        image = Image.open(image_path)
        
        # Process the document with the unwarp model
        try:
            unwarped_image = unwarper.inference(image)
            
            # Save the unwarped document in the output folder
            output_path = os.path.join(output_folder, f"unwarped_{filename}")
            unwarped_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
