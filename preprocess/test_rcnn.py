import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Convert the image to a PyTorch tensor
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    return image_tensor

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

features = []
def save_features(mod, inp, outp):
    features.append(outp)

# you can also hook layers inside the roi_heads
layer_to_hook = 'roi_heads.box_head.fc6'
for name, layer in model.named_modules():
    print(name)
    if layer_to_hook == name:
        layer.register_forward_hook(save_features)
        print(f'**Hooked to {name}')

# train model and after training, you can see the output activations in the "features" list
image_tensor = preprocess_image('/mnt/seagate12t/VQA/scienceqa/images/train/1/image.png')

# duplicate the image tensor to simulate a batch
image_tensor = image_tensor.repeat(2, 1, 1, 1)
with torch.no_grad():
    model(image_tensor)

# Extract region features from the last hook
region_features = features[-1]
pass