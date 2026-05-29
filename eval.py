import os
import torch
import imageio
import argparse
import numpy as np
import seaborn as sns
from utils.dataset import load_mat_hsi
from models.get_model import get_model
from utils.train_test_valid import test
from utils.utils import metrics, show_results

def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI classification evaluation")
    parser.add_argument("--model", type=str, default='behalfnet')
    parser.add_argument("--dataset_name", type=str, default="Bi-Temporal-split")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/Anji")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--weights", type=str, default="./checkpoints/behalfnet/Anji/Bi-Temporal-split/0")
    parser.add_argument("--outputs", type=str, default="./results")

    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    print("dataset: {}".format(opts.dataset_name))
    print("patch size: {}".format(opts.patch_size))
    print("model: {}".format(opts.model))

    image_t1, image_t2, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands_t1 = image_t1.shape[-1]
    num_bands_t2 = image_t2.shape[-1]

    # Anji and Viareggio use the same color
    palette = {
        0: (0, 0, 0),  # class 0 (Undefined)
        1: (220, 20, 60),  # class 1 (Corn)
        2: (30, 144, 255),  # class 2 (Tea)
        3: (50, 205, 50),  # class 3 (Potato)
        4: (255, 215, 0),  # class 4 (Wheat)
        5: (75, 0, 130),  # class 5 (Yam)
        6: (0, 255, 127),  # class 6 (Grassland)
        7: (255, 105, 180),  # class 7 (Broad-leaved forest)
        8: (255, 165, 0),  # class 8 (Needle-leaved forest)
        9: (0, 238, 238),  # class 9 (Mixed forests)
        10: (138, 43, 226),  # class 10 (Building)
        11: (160, 82, 45),  # class 11 (Road)
        12: (245, 245, 245),  # class 12 (Water)
    }

    # load model and weights
    model = get_model(opts.model, opts.dataset_name, opts.patch_size)
    print('loading weights from %s' % opts.weights + '/model_best.pth')
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(opts.weights, 'model_best.pth')))
    model.eval()

    # testing model: metric for the whole HSI, including train, val, and test
    probabilities = test(model, opts.weights, [image_t1, image_t2], opts.patch_size, num_classes, device=device)
    prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(prediction, gt, n_classes=num_classes)

    prediction[gt < 0] = -1

    # color results
    colored_gt = color_results(gt + 1, palette)
    colored_pred = color_results(prediction + 1, palette)

    outfile = os.path.join(opts.outputs, opts.dataset_name, opts.model)
    os.makedirs(outfile, exist_ok=True)

    imageio.imsave(os.path.join(outfile, opts.dataset_name + '_gt.png'), colored_gt)  # eps or png
    imageio.imsave(os.path.join(outfile, opts.dataset_name + '_' + opts.model + '_out.png'), colored_pred)  # or png

    result = show_results(run_results, label_values=labels)
    with open(os.path.join(outfile, 'result.txt'), "w") as file:
        file.write(result)

    del model
