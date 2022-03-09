import torch
from torch.nn import CrossEntropyLoss
import time
import os
import numpy as np

from model import DiagnosisModel
from dataset import get_dataloaders
from utils import get_class_weights, calculate_mpa, calculate_accuracy, save_train_history

checkpoint_path = "checkpoints/checkpoint_best.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(model, optimizer, semantic_loss_fn, cls_loss_fn, dataloaders, epochs, print_freq=1):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = {'total_loss': [], 'semantic_loss': [], 'cls_loss': []}
    train_loss_history = {'total_loss': [], 'semantic_loss': [], 'cls_loss': []}

    best_loss = float('inf')
    best_mpa = None
    best_cls_acc = None

    model.to(device)
    semantic_loss_fn.to(device)
    cls_loss_fn.to(device)

    for epoch in range(1, epochs + 1):
        print('-' * 15 + f' Epoch [{epoch}/{epochs}] ' + '-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = {'total': 0.0, 'semantic': 0.0, 'cls': 0.0}
            running_pixel_acc = []
            running_cls_acc = []

            for batch_idx, (input_image, semantic_target, label, _) in enumerate(dataloaders[phase]):

                input_image = input_image.to(device)  # (batch, channels, width, height)
                semantic_target = semantic_target.squeeze().to(device)  # (batch, n_classes, width, height)
                label = label.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # Calculate forward + loss + predictions
                # Track history only if in training phase
                with torch.set_grad_enabled(phase == 'train'):

                    # Fix final batch having only 1 sample
                    if len(semantic_target.shape) != 3:
                        semantic_target = semantic_target.unsqueeze(0)

                    # Make forward pass to calculate probabilities
                    result = model(input_image)
                    cls_logits, semantic_logits = result['cls_logits'], result['semantic_logits']

                    # Calculate losses
                    cls_loss = cls_loss_fn(cls_logits, label)
                    semantic_loss = semantic_loss_fn(semantic_logits, semantic_target)
                    loss = semantic_loss + cls_loss

                    # Calculate hard predictions
                    semantic_prediction = semantic_logits.argmax(dim=1, keepdim=False)
                    cls_prediction = cls_logits.argmax(dim=1, keepdim=False)

                    if phase == 'train':
                        # Calculate gradients and take optimization step
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss['total'] += loss.item()
                running_loss['semantic'] += semantic_loss.item()
                running_loss['cls'] += cls_loss.item()

                if phase == 'val':
                    # Calculate metrics
                    running_pixel_acc.append(
                        calculate_mpa(semantic_target=semantic_target, prediction=semantic_prediction))
                    running_cls_acc.append(calculate_accuracy(cls_target=label, prediction=cls_prediction))

                if batch_idx % print_freq == 0:
                    print(f'[{phase}][{batch_idx + 1}/{len(dataloaders[phase])}]',
                          f'Minibatch loss: {loss.item():.4f} \t',
                          f'(Classification: {cls_loss.item():.4f} | Semantic: {semantic_loss.item():.4f})')

            # Compute and print metrics
            epoch_loss_total = running_loss['total'] / len(dataloaders[phase].dataset)
            epoch_loss_semantic = running_loss['semantic'] / len(dataloaders[phase].dataset)
            epoch_loss_cls = running_loss['cls'] / len(dataloaders[phase].dataset)

            epoch_pixel_acc = None
            epoch_cls_acc = None
            if phase == 'val':
                val_loss_history['total_loss'].append(epoch_loss_total)
                val_loss_history['semantic_loss'].append(epoch_loss_semantic)
                val_loss_history['cls_loss'].append(epoch_loss_cls)
                epoch_pixel_acc = np.mean(running_pixel_acc)
                epoch_cls_acc = np.mean(running_cls_acc)
            if phase == 'train':
                train_loss_history['total_loss'].append(epoch_loss_total)
                train_loss_history['semantic_loss'].append(epoch_loss_semantic)
                train_loss_history['cls_loss'].append(epoch_loss_cls)

            to_print = f'[{phase}] Epoch [{epoch}/{epochs}] [Loss: {epoch_loss_total:.4f}' + \
                       f'(Classification: {cls_loss.item():.4f} | Semantic: {semantic_loss.item():.4f})]'
            to_print += f'[CLS Acc: {epoch_cls_acc:.4f}]' if epoch_cls_acc else ''
            to_print += f'[MPA: {epoch_pixel_acc:.4f}]' if epoch_pixel_acc else ''
            print(to_print)

            # Save model upon improvement
            if phase == 'val' and epoch_loss_total <= best_loss:
                best_loss = epoch_loss_total
                best_mpa = epoch_pixel_acc
                best_cls_acc = epoch_cls_acc
                best_model_path = os.path.join("./checkpoints/", f"checkpoint_best.pth")
                print(
                    f"Saving new best model [CLS Acc: {epoch_cls_acc:.4f} , MPA={epoch_pixel_acc:.4f}] at path: {best_model_path}")
                torch.save(model.state_dict(), best_model_path)

        save_train_history({'train_acc': train_acc_history,
                            'val_acc': val_acc_history,
                            'train_loss': train_loss_history,
                            'val_loss': val_loss_history})
        print(" ")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best validation results:[CLS Acc: {best_cls_acc:.4f} , MPA={best_mpa:.4f}]')

    train_history = {'train_loss': train_loss_history, 'val_loss': val_loss_history}
    return train_history


def run_training(checkpoint=None):
    print("Initializing...")
    dataloaders = get_dataloaders(path="./data/", batch_size=8)

    # Load model
    model = DiagnosisModel()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    # Load optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

    # Define loss functions
    print("Computing class weights...")
    w_cls, w_sem, _ = get_class_weights(dataloaders['train'])
    print(f"CLS: {w_cls}, SEG: {w_sem}")
    cls_loss_fn = CrossEntropyLoss(weight=w_cls)
    semantic_loss_fn = CrossEntropyLoss(weight=w_sem)

    # Start training
    epochs = 35
    train_history = train(model, optimizer, semantic_loss_fn, cls_loss_fn, dataloaders, epochs=epochs, print_freq=20)

    # Save results
    save_train_history(train_history)


if __name__ == "__main__":
    run_training()
