import torch
from model import ViolenceClassifier
from torchvision import transforms
from PIL import Image
import sys

def classify_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    image_paths = sys.argv[2:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model from checkpoint, specifying num_classes
    model = ViolenceClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
    model.to(device)

    predictions = []
    for image_path in image_paths:
        prediction = classify_image(model, image_path, device)
        predictions.append(prediction)

    print(predictions)
