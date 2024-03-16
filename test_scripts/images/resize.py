from PIL import Image

# Path to the image
image_path = '/example_path/'

# Desired dimensions
desired_width = 576
desired_height = 1024

# Open the image
with Image.open(image_path) as img:
    # Get current dimensions
    current_width, current_height = img.size

    # Check if the image dimensions are already the desired ones
    if (current_width, current_height) != (desired_width, desired_height):
        # Resize the image
        resized_img = img.resize((desired_width, desired_height))
        
        # Save the resized image, overwriting the original
        # You can also save to a different path if you don't want to overwrite
        resized_img.save(image_path)
        print(f"Image resized to {desired_width}x{desired_height} and saved.")
    else:
        print("Image is already at the desired dimensions.")