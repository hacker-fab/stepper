import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, transform, num_copies=5):
        """
        Custom dataset to replicate each sample and apply random transformations.
        :param base_dataset: Original dataset (e.g., datasets.ImageFolder)
        :param transform: Transformation to apply
        :param num_copies: Number of augmented copies per sample
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.num_copies = num_copies

    def __len__(self):
        return len(self.base_dataset) * self.num_copies

    def __getitem__(self, idx):
        # Determine the index of the original dataset
        original_idx = idx // self.num_copies
        img, label = self.base_dataset[original_idx]
        # Apply random transformations
        if self.transform:
            img = self.transform(img)
        return img, label

def visualize_transformation(image_path, transform):
    """
    Visualizes the effect of a torchvision transformation on an image.
    
    :param image_path: Path to the image to transform
    :param transform: torchvision.transforms.Compose object
    """
    # Open the image using PIL
    image = Image.open(image_path).convert("RGB")
    
    # Apply the transformation
    transformed_image = transform(image)
    
    # Convert the transformed tensor back to a format suitable for visualization
    transformed_image = TF.to_pil_image(transformed_image)
    
    # Plot the original and transformed images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(transformed_image)
    axes[1].set_title("Transformed Image")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
    plt.waitforbuttonpress()
