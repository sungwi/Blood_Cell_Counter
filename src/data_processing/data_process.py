import os
import cv2


base_input_folder = 'datasets/dataset_1/raw/'  # Base folder containing subfolders with images
base_output_folder = 'datasets/dataset_1/processed/'  # Base folder to store processed images

# Define target size for resizing
target_size = (256, 256)

# Iterate through subfolders in the base input folder
for folder_name in os.listdir(base_input_folder):
    folder_path = os.path.join(base_input_folder, folder_name)

    # Check if it's a directory and not a file
    if os.path.isdir(folder_path):
        input_folder = folder_path
        output_folder = os.path.join(base_output_folder, f"{folder_name}_processed")

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Processing images in: {input_folder}")

        # Process each image in the current subfolder
        for filename in os.listdir(input_folder):
            # Process only image files with the following extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                try:
                    image = cv2.imread(image_path)

                    # Check if the image was loaded successfully
                    if image is None:
                        print(f"Error: Could not read image at {image_path}")
                        continue  # Skip to the next image

                    # Resize the image
                    resized_image = cv2.resize(image, target_size)

                    # Convert to grayscale
                    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                    # Apply Gaussian blur for noise reduction
                    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

                    # Save the processed image in the output folder
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, denoised_image)

                    print(f"Processed and saved: {output_path}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
