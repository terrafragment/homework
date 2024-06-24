import torch
from model import ViolenceClassifier
from torchvision import transforms

class ViolenceClass:
    def __init__(self, checkpoint_path, device=None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViolenceClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
        self.model.to(self.device)
        self.model.eval()

    def misc(self):

        pass

    def classify(self, img: torch.Tensor) -> list:
        # 图像分类
        img = img.to(self.device)
        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
        return predicted.cpu().tolist()


if __name__ == "__main__":
    import sys
    from PIL import Image

    checkpoint_path = sys.argv[1]
    image_paths = sys.argv[2:]

    classifier = ViolenceClass(checkpoint_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)


    batch = torch.stack(images)

    predictions = classifier.classify(batch)
    print(predictions)