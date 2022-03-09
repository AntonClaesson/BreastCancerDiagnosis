import torch
import pickle
from dataset import get_dataloaders
from sklearn.metrics import confusion_matrix


def get_class_weights(loader):
    """
    "Weight of class c is the size of largest class divided by the size of class c."
    https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
    """
    # Compute class counts for both tasks
    class_counts = {0: 0, 1: 0}
    pixel_counts = {0: 0, 1: 0}
    for (_, semantic_target, label, _) in loader:
        ones = label.sum()
        class_counts[1] += ones
        class_counts[0] += label.numel() - ones
        pixel_ones = semantic_target.sum()
        pixel_counts[1] += pixel_ones
        pixel_counts[0] += semantic_target.numel() - pixel_ones

    # Compute weights
    lpc = max(pixel_counts[0], pixel_counts[1])
    pixel_class_weights = torch.FloatTensor([lpc / pixel_counts[0], lpc / pixel_counts[1]])
    lcc = max(class_counts[0], class_counts[1])
    cls_class_weights = torch.FloatTensor([lcc / class_counts[0], lcc / class_counts[1]])

    return cls_class_weights, pixel_class_weights, {'classification': class_counts, 'segmentation': pixel_counts}


def calculate_mpa(semantic_target, prediction):
    # Calculates mean pixel accuracy
    n_pixels = semantic_target.numel()
    n_correct = (prediction == semantic_target).sum()
    pixel_acc = n_correct / n_pixels
    return pixel_acc.item()


def calculate_accuracy(cls_target, prediction):
    # Calculate classification accuracy
    n_samples = cls_target.shape[0]
    n_correct_cls = (cls_target == prediction).sum()
    cls_acc = n_correct_cls / n_samples
    return cls_acc.item()


def compute_acc_sens_spec(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]

    # Accuracy
    acc = (TP+TN)/(TP+FP+FN+TN)
    # Sensitivity/recall/true positive rate
    sens = TP/(TP+FN)
    # Specificity/true negative rate
    spec = TN/(TN+FP)

    return (acc, sens, spec)



def save_train_history(train_history, save_path='./checkpoints/train_history.pickle'):
    with open(save_path, 'wb') as f:
        pickle.dump(train_history, f)


def calculate_dummy_classifier_scores():
    (_, _, counts) = get_class_weights(get_dataloaders('./data/', batch_size=8, include_test_loader=True)['test'])
    # assumes that the 0-label is the majority class, check that this assumption is correct
    assert counts['classification'][0] > counts['classification'][1]
    assert counts['segmentation'][0] > counts['segmentation'][1]

    # If we always predicted the 0 class (not malignant)
    dummy_classifier_acc = (
                counts['classification'][0] / (counts['classification'][0] + counts['classification'][1])).item()

    # If we always predicted the 0 class (no tumor pixel)
    dummy_segmentation_acc = (
                counts['segmentation'][0] / (counts['segmentation'][0] + counts['segmentation'][1])).item()
    print(f"Dummy classifier accuracy: {dummy_classifier_acc:.4f}\n"
          f"Dummy segmentation model MPA: {dummy_segmentation_acc:.4f}")
    print(f"Pixel counts: {counts['segmentation']}")
    print(f"There are {counts['segmentation'][0]/counts['segmentation'][1]:.2f} times more zero pixels.")

if __name__ == '__main__':
    calculate_dummy_classifier_scores()
