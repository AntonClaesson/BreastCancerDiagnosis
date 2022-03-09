import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageOps

import glob
from pathlib import Path

# NOTE changing size requires changing fully connected layer for classification in
# ClassificationHead in model.py
image_width = 224
image_height = 224

norm_params = {'mean': [0.327812], 'std': [0.201863]}

def get_dataloaders(path, batch_size, test_size=0.2, include_test_loader=False):
    full_dataset = BreastCancerDataset(path=path)

    # Make the test split
    train_size = int((1 - test_size) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

    # Make train/val split
    val_size = test_size
    train_size = train_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(1))

    # Make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=1) if include_test_loader else None

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


class BreastCancerDataset(Dataset):

    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_height), interpolation=InterpolationMode.NEAREST),
    ])

    normalization = transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])

    def __init__(self, path):
        path = Path(path)
        if not (path.exists() and path.is_dir()):
            raise ValueError(f"Data path '{path}' is invalid")
        self._samples = self._collect_samples(path)

    def __getitem__(self, index):
        # Access the stored paths and label for the given index
        img_path, mask_path, label = self._samples[index]

        # Load PIL images into memory
        img = Image.open(img_path)
        mask_img = Image.open(mask_path)

        # Make sure images are grayscale
        img = ImageOps.grayscale(img)
        mask_img = ImageOps.grayscale(mask_img)

        # To tensor and resize image
        img = BreastCancerDataset.preprocessing(img)
        mask_img = BreastCancerDataset.preprocessing(mask_img)

        # Normalize input image
        img = BreastCancerDataset.normalization(img)

        # Other fixes
        mask_img = mask_img.long()

        # Note image paths to loaded data
        loaded_paths = {"input_image": img_path, "segmentation_image": mask_path}

        return img, mask_img, label, loaded_paths

    def __len__(self):
        return len(self._samples)

    @staticmethod
    def _collect_samples(path):
        benign_image_list = sorted(glob.glob(f'{path}/benign/images/*'))
        benign_masks_list = sorted(glob.glob(f'{path}/benign/masks/*'))

        malignant_image_list = sorted(glob.glob(f'{path}/malignant/images/*'))
        malignant_masks_list = sorted(glob.glob(f'{path}/malignant/masks/*'))

        normal_image_list = sorted(glob.glob(f'{path}/normal/images/*'))
        normal_masks_list = sorted(glob.glob(f'{path}/normal/masks/*'))

        # Place benign and normal image samples first followed by the malignant examples
        image_list = benign_image_list + normal_image_list + malignant_image_list
        mask_list = benign_masks_list + normal_masks_list + malignant_masks_list

        # Note down the label of each sample. Mark benign and normal images as "0" and malignant as "1"
        label = [0 for _ in benign_image_list + normal_image_list]
        label.extend([1 for _ in malignant_image_list])
        label = torch.tensor(label, dtype=torch.long)

        # Index 0 example:
        # 0: (image_path, mask_path, target)
        return list(zip(image_list, mask_list, label))


def compute_mean_std():
    mean = 0
    std = 0
    nb_samples = 0
    loaders = get_dataloaders('./data/', batch_size=8, include_test_loader=True)

    for img, _, _, _ in loaders['train']:
        batch_samples = img.size(0)
        img = img.view(batch_samples, img.size(1), -1)
        mean += img.mean(2).sum(0).item()
        std += img.std(2).sum(0).item()
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(f"Mean: {mean}, Std: {std}")


if __name__ == "__main__":
    compute_mean_std()
