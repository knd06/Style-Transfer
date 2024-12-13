import torchvision.transforms as transforms
import torch

img_size = 512

prep = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.tensor([2, 1, 0], dtype=torch.long)]),  # rgb to bgr
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
    transforms.Lambda(lambda x: x.mul_(255)),  # scale to [0, 255]
])

recover = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1. / 255)),  # scale back to [0, 1]
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
    transforms.Lambda(lambda x: x[torch.tensor([2, 1, 0], dtype=torch.long)]),  # bgr back to rgb
])

toPIL = transforms.ToPILImage()
def post(tensor):
    """
    Postprocesses a tensor to convert it back to a valid image format.
    """
    tensor = recover(tensor)
    tensor.clamp_(0, 1)
    return toPIL(tensor)