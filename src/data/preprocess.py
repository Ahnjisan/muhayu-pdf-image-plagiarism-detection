from PIL import Image
from torchvision import transforms

common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

a_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

b_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_for_A(img_path: str):
    img = Image.open(img_path)
    return a_transform(img)


def preprocess_for_B(img_path: str):
    img = Image.open(img_path).convert('RGB')
    return b_transform(img)


def preprocess_for_cosine(img_path: str):
    img = Image.open(img_path).convert('RGB')
    return common_transform(img)
