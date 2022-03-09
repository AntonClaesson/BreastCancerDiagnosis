import numpy as np
from utils import *
from dataset import get_dataloaders
from model import DiagnosisModel
from torch.nn.functional import softmax
from plotting import plot_roc_curves

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test(model, loader, print_freq=5):
    model.to(device)
    model.eval()

    running_pixel_acc = []
    cls_predictions = []
    cls_class1_probs = []
    cls_targets = []

    for batch_idx, (input_image, semantic_target, label, _) in enumerate(loader):
        input_image = input_image.to(device)  # (batch, channels, width, height)
        semantic_target = semantic_target.squeeze().to(device)  # (batch, n_classes, width, height)
        label = label.to(device)

        # Calculate forward + loss + predictions
        with torch.set_grad_enabled(False):

            # Fix final batch having only 1 sample
            if len(semantic_target.shape) != 3:
                semantic_target = semantic_target.unsqueeze(0)

            # Make forward pass to calculate probabilities
            result = model(input_image)
            cls_logits, semantic_logits = result['cls_logits'], result['semantic_logits']

            # Get class probabilites
            cls_probs = softmax(cls_logits, dim=1)

            # Calculate hard predictions
            semantic_prediction = semantic_logits.argmax(dim=1, keepdim=False)
            cls_prediction = cls_logits.argmax(dim=1, keepdim=False)

            # Calculate metrics
            running_pixel_acc.append(calculate_mpa(semantic_target=semantic_target, prediction=semantic_prediction))

            # Save prediction and target for metric in the end
            cls_targets.extend(list(label.cpu().numpy()))
            cls_predictions.extend(list(cls_prediction.cpu().numpy()))
            cls_class1_probs.extend(list(cls_probs[:, 1].cpu().numpy()))

            if batch_idx % print_freq == 0:
                print(f'[{batch_idx + 1}/{len(loader)}]')

    test_pixel_acc = np.mean(running_pixel_acc)

    return test_pixel_acc, cls_targets, cls_predictions, cls_class1_probs


def run_test(checkpoint):
    print("Initializing...")
    dataloaders = get_dataloaders(path="./data/", batch_size=8, include_test_loader=True)

    # Load model
    print("Loading model...")
    model = DiagnosisModel()
    print(model)
    model.load_state_dict(torch.load(checkpoint))

    # Start test
    test_cls_targets = None
    test_y_probs_cls1 = None
    val_cls_targets = None
    val_y_probs_cls1 = None
    for phase in ['val', 'test']:
        print("Testing...")
        test_pixel_acc, cls_targets, cls_predictions, cls_class1_probs = test(model, loader=dataloaders[phase])

        if phase == 'val':
            val_y_probs_cls1 = cls_class1_probs
            val_cls_targets = cls_targets
        else:
            test_y_probs_cls1 = cls_class1_probs
            test_cls_targets = cls_targets

        (acc, sens, spec) = compute_acc_sens_spec(y_true=cls_targets, y_pred=cls_predictions)
        to_print = f"\n[{phase}] result:\n"
        to_print += f'[Classification] Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}]'
        to_print += f'\n[Segmentation] MPA: {test_pixel_acc:.4f}]'
        print(to_print)
        print("")

    plot_roc_curves(test_y_target=test_cls_targets,
                    test_y_probs_cls1=test_y_probs_cls1,
                    val_y_target=val_cls_targets,
                    val_y_probs_cls1=val_y_probs_cls1)


if __name__ == "__main__":
    run_test(checkpoint='checkpoints/checkpoint_best.pth')
