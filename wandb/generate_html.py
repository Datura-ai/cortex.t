import os

def generate_html(image_folder, output_file='gallery.html'):
    with open(output_file, 'w') as file:
        file.write('<html>\n<head>\n<title>Image Gallery</title>\n</head>\n<body>\n')
        file.write('<h1>Image Gallery</h1>\n')

        for subdir, _, files in os.walk(image_folder):
            for image in files:
                if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, image)
                    # Replace backslashes with forward slashes for HTML compatibility
                    img_path = img_path.replace("\\", "/")
                    file.write(f'<img src="{img_path}" alt="{image}" style="width:300px; height:auto; margin:10px;">\n')

        file.write('</body>\n</html>')

generate_html('wandb_images')
