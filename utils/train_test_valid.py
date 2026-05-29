import os
import torch
import numpy as np
from tqdm import tqdm
from utils.utils import grouper, sliding_window, count_sliding_window

def contrastive_loss(feature1, feature2, margin=1.0):
    distances = torch.sqrt(torch.sum((feature1 - feature2) ** 2, dim=1))
    loss = torch.mean(torch.clamp(margin - distances, min=0) ** 2)
    return loss

def train(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):
    best_acc = -0.1
    losses = []

    for e in tqdm(range(1, epoch + 1), desc=""):
        network.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            image_t1, image_t2 = images
            image_t1, image_t2, targets = image_t1.to(device), image_t2.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = network(image_t1, image_t2)  # Pass both inputs to the network
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())

        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            train_info = "train at epoch {}/{}, loss={:.6f}"
            train_info = train_info.format(e, epoch, mean_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []

        val_acc = validation(network, val_loader, device)

        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)


def validation(network, val_loader, device):
    num_correct = 0.
    total_num = 0.
    network.eval()
    for batch_idx, (images, targets) in enumerate(val_loader):
        image_t1, image_t2 = images
        image_t1, image_t2, targets = image_t1.to(device), image_t2.to(device), targets.to(device)

        # Pass both inputs to the network
        outputs = network(image_t1, image_t2)
        _, outputs = torch.max(outputs, dim=1)

        # Compute accuracy
        for output, target in zip(outputs, targets):
            num_correct = num_correct + (output.item() == target.item())
            total_num = total_num + 1

    overall_acc = num_correct / total_num
    return overall_acc


def test(network, model_dir, images, patch_size, n_classes, device):
    image_t1, image_t2 = images

    network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image_t1.shape[:2]  # Assume both images have the same size
    pad_size = patch_size // 2

    # pad both images
    image_t1 = np.pad(image_t1, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    image_t2 = np.pad(image_t2, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image_t1.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(image_t1, window_size=window_size) // batch_size
    for batch_t1, batch_t2 in tqdm(
            zip(grouper(batch_size, sliding_window(image_t1, window_size=window_size)),
                grouper(batch_size, sliding_window(image_t2, window_size=window_size))),
            total=iterations,
            desc="inference on the HSI"
    ):
        with torch.no_grad():
            data_t1 = []
            data_t2 = []

            for b1, b2 in zip(batch_t1, batch_t2):
                data_t1.append(b1[0])
                data_t2.append(b2[0])

            data_t1 = np.array(data_t1).transpose((0, 3, 1, 2))
            data_t2 = np.array(data_t2).transpose((0, 3, 1, 2))

            data_t1 = torch.from_numpy(data_t1).to(device)
            data_t2 = torch.from_numpy(data_t2).to(device)

            indices = [b[1:] for b in batch_t1]

            output = network(data_t1, data_t2)  # Pass both inputs to the network
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out

    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:  # save the ckpt for each 10 epoch
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))


