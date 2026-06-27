from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

from pathlib import Path

import random

import numpy as np

def add_noise(image, amount):
    arr = np.array(image).astype(np.int16)
    noise = np.random.normal(0, amount, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_effect(image, effect):
    angle = None
    if effect == 'grayscale':
        image = ImageOps.grayscale(image).convert('RGB')
    elif effect == 'rotate':
        angle = random.uniform(-25, 25)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    elif effect == 'brighten':
        image = ImageEnhance.Brightness(image).enhance(random.uniform(1.2, 1.8))
    elif effect == 'dim':
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.4, 0.8))
    elif effect == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(1, 3)))
    elif effect == 'noise':
        image = add_noise(image, random.uniform(10, 30))

    return image, angle

def process_batch(source_dir, output_dir, effect):
    """Processes all images in a source folder using a selected effect."""
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    for file_path in src_path.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            try:
                # Open and convert to RGB
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    
                    # Apply effect
                    modified_img, angle = apply_effect(img, effect)
                    
                    # Save with modified suffix
                    new_filename = f"{file_path.stem}_{effect}{file_path.suffix}"
                    modified_img.save(out_path / new_filename)
                    print(f"Processed: {file_path.name}")
                    
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")

# Run the batch process
if __name__ == "__main__":
    process_batch(
        source_dir="input_images", 
        output_dir="output_images", 
        effect="blur"
    )
    process_batch(
        source_dir="input_images", 
        output_dir="output_images", 
        effect="grayscale"
    )
    process_batch(
        source_dir="input_images", 
        output_dir="output_images", 
        effect="rotate"
    )
    process_batch(
        source_dir="input_images", 
        output_dir="output_images", 
        effect="brighten"
    )
    process_batch(
        source_dir="input_images", 
        output_dir="output_images", 
        effect="dim"
    )
    process_batch(
        source_dir="input_images", 
        output_dir="output_images", 
        effect="noise"
    )

