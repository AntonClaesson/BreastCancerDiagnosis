import pickle
import matplotlib.pyplot as plt
from model import DiagnosisModel
import torch
from dataset import get_dataloaders, BreastCancerDataset
import numpy as np
from torchvision import transforms, utils
from PIL import Image, ImageOps
from sklearn.metrics import roc_curve

def plot_train_history(path):
    with open(path, 'rb') as f:
        train_history = pickle.load(f)
    # {'total_loss':[], 'semantic_loss': [], 'cls_loss': []}
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    n_epochs = len(train_history['train_loss']['cls_loss'])
    print(n_epochs)
    x_ticks = list(range(0, n_epochs + 2, 5))
    plt.setp(ax, xticks=x_ticks, xticklabels=x_ticks)

    # Classification losses
    train_loss_cls = train_history['train_loss']['cls_loss']
    val_loss_cls = train_history['val_loss']['cls_loss']
    ax[0].plot(train_loss_cls, label='Training Loss [CLS]')
    ax[0].plot(val_loss_cls, label='Validation Loss [CLS]')
    ax[0].set_title('Classification loss')
    ax[0].legend()
    # Segmentation losses
    train_loss_semantic = train_history['train_loss']['semantic_loss']
    val_loss_semantic = train_history['val_loss']['semantic_loss']
    ax[1].plot(train_loss_semantic, label='Training Loss [SEG]')
    ax[1].plot(val_loss_semantic, label='Validation Loss [SEG]')
    ax[1].set_title('Segmentation loss')
    ax[1].legend()
    # Total losses
    total_train_loss = train_history['train_loss']['total_loss']
    total_val_loss = train_history['val_loss']['total_loss']
    ax[2].plot(total_train_loss, label='Training Loss')
    ax[2].plot(total_val_loss, label='Validation Loss')
    ax[2].set_title('Total loss')
    ax[2].legend()
    plt.show()


def plot_sample_segmentations_for_checkpoint(checkpoint, n_samples):
    dataloaders = get_dataloaders(path="./data/", batch_size=1, include_test_loader=True)
    loader = dataloaders['test']

    model = DiagnosisModel()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    for _ in range(n_samples):
        # Load a sample
        index = np.random.randint(0, len(loader))
        (input_image, semantic_target, label, paths) = loader.dataset.__getitem__(index)

        # Make forward pass
        input_image = input_image.unsqueeze(0)
        semantic_target = semantic_target.unsqueeze(0)
        with torch.no_grad():
            # Make forward pass to calculate probabilities
            result = model(input_image)
            cls_logits, semantic_logits = result['cls_logits'], result['semantic_logits']

            # Calculate hard predictions
            semantic_prediction = semantic_logits.argmax(dim=1, keepdim=False)
            cls_prediction = cls_logits.argmax(dim=1, keepdim=False)

        # Plotting
        # input_image = input_image.squeeze(0)
        # semantic_target = semantic_target.squeeze(0)
        # grid = utils.make_grid([input_image, semantic_target, semantic_prediction])
        # plt.imshow(grid.permute(1, 2, 0))
        # plt.show()

        # Load un-normalized input image
        input_image = BreastCancerDataset.preprocessing(ImageOps.grayscale(Image.open(paths['input_image'])))
        input_image_rgb = (input_image.expand(3, -1, -1)*255).to(torch.uint8)

        # Pre-plotting fixes
        semantic_target = semantic_target.squeeze()
        semantic_prediction = semantic_prediction.squeeze()


        # Plot as overlay:
        img_seg_true=utils.draw_segmentation_masks(image=input_image_rgb, masks=(semantic_target == 1), alpha=0.4, colors=['red'])
        img_seg_pred=utils.draw_segmentation_masks(image=input_image_rgb, masks=(semantic_prediction == 1), alpha=0.4, colors=['red'])

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        true_diagnosis = 'malignant' if label == 1 else 'not malignant'
        predicted_diagnosis = 'malignant' if cls_prediction == 1 else 'not malignant'
        title = f'True diagnosis: {true_diagnosis}\nPredicted diagnosis: {predicted_diagnosis}'
        fig.suptitle(title, fontsize=18)
        ax[0].imshow(img_seg_true.permute(1,2,0).numpy())
        ax[1].imshow(img_seg_pred.permute(1,2,0).numpy())
        ax[0].set_title('Ground truth')
        ax[1].set_title('Predicted')
        ax[0].set_adjustable('box')
        ax[1].set_adjustable('box')
        plt.setp(ax, xticks=[], yticks=[])
        plt.show()


def plot_roc_curves(test_y_target, test_y_probs_cls1, val_y_target,val_y_probs_cls1):
    #y_probs_cls1 is the probability of class 1 with shape (batch, 1)

    test_fpr, test_tpr, _ = roc_curve(test_y_target, test_y_probs_cls1)
    val_fpr, val_tpr, _ = roc_curve(val_y_target, val_y_probs_cls1)
    # Plot ROC curves
    plt.figure(figsize=(8, 8))
    plt.title('ROC')
    plt.plot(test_fpr, test_tpr, marker='.', label='UNet Classifier [Test]')
    plt.plot(val_fpr, val_tpr, marker='*', label='UNet Classifier [Val]')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_train_history('./checkpoints/train_history.pickle')
    plot_sample_segmentations_for_checkpoint('./checkpoints/checkpoint_best.pth', n_samples=50)
