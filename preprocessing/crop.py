import os
from PIL import Image

def create_patches(input_dir, output_dir, patch_size=(192, 192)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    patch_counter = 0  # start counting from 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        patch_w, patch_h = patch_size
        skip_w = int(patch_w*2)
        skip_h = int(patch_h*2)

        for i in range(0, height - patch_h + 1, skip_h):
            for j in range(0, width - patch_w + 1, skip_w):
                patch = image.crop((j, i, j + patch_w, i + patch_h))
                patch_filename = f"{patch_counter}.png"
                patch.save(os.path.join(output_dir, patch_filename))
                patch_counter += 1

        print(f"Processed {filename}")

if __name__ == "__main__":
    input_folder = "dataset/GoPro/train/LQ"
    output_folder = input_folder + '_crops_64x64_skip_128'
    create_patches(input_folder, output_folder, patch_size=(64, 64))
