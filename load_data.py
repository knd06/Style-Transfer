from PIL import Image
import os

def load_images(content_idx, style_idx):
    img_dir = os.path.join(os.getcwd(), 'Images')
    img_names = [
        f'content{content_idx}.png',
        f'style{style_idx}.png'
    ]
    imgs = [Image.open(os.path.join(img_dir, name)) for name in img_names]
    return imgs