from PIL import Image
import torchvision.transforms as T
import torchvision.utils as tutils
import torch
from collections import OrderedDict
import json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Resize(object):
    def __init__(self, size, interp=Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def save_images(images, path):
    grid = tutils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    imgs = Image.fromarray(ndarr)
    imgs.save(path)

def save_graphs(graphs, path):
    json_graphs = json.dumps(graphs, indent=2)
    with open(path, "w") as outfile:
        outfile.write(json_graphs)


def load_transfer_weights_biases(transfer_model_path, device):
    checkpoint = torch.load(transfer_model_path, device)
    original_od = checkpoint['model_state']
    transfer_weights_biases = OrderedDict()

    # object embedding
    transfer_weights_biases['obj_embedding.weight'] = original_od['obj_embeddings.weight']

    # pred embedding
    transfer_weights_biases['pred_embedding.weight'] = original_od['pred_embeddings.weight']

    # Layer_0
    transfer_weights_biases['gcn.0.net1.0.weight'] = original_od['gconv.net1.0.weight']
    transfer_weights_biases['gcn.0.net1.0.bias'] = original_od['gconv.net1.0.bias']
    transfer_weights_biases['gcn.0.net1.2.weight'] = original_od['gconv.net1.2.weight']
    transfer_weights_biases['gcn.0.net1.2.bias'] = original_od['gconv.net1.2.bias']
    transfer_weights_biases['gcn.0.net2.0.weight'] = original_od['gconv.net2.0.weight']
    transfer_weights_biases['gcn.0.net2.0.bias'] = original_od['gconv.net2.0.bias']
    transfer_weights_biases['gcn.0.net2.2.weight'] = original_od['gconv.net2.2.weight']
    transfer_weights_biases['gcn.0.net2.2.bias'] = original_od['gconv.net2.2.bias']

    # Layer_1 - Layer_4
    for i in range(1, 5):
        transfer_weights_biases[f'gcn.{i}.net1.0.weight'] = original_od[f'gconv_net.gconvs.{i - 1}.net1.0.weight']
        transfer_weights_biases[f'gcn.{i}.net1.0.bias'] = original_od[f'gconv_net.gconvs.{i - 1}.net1.0.bias']
        transfer_weights_biases[f'gcn.{i}.net1.2.weight'] = original_od[f'gconv_net.gconvs.{i - 1}.net1.2.weight']
        transfer_weights_biases[f'gcn.{i}.net1.2.bias'] = original_od[f'gconv_net.gconvs.{i - 1}.net1.2.bias']
        transfer_weights_biases[f'gcn.{i}.net2.0.weight'] = original_od[f'gconv_net.gconvs.{i - 1}.net2.0.weight']
        transfer_weights_biases[f'gcn.{i}.net2.0.bias'] = original_od[f'gconv_net.gconvs.{i - 1}.net2.0.bias']
        transfer_weights_biases[f'gcn.{i}.net2.2.weight'] = original_od[f'gconv_net.gconvs.{i - 1}.net2.2.weight']
        transfer_weights_biases[f'gcn.{i}.net2.2.bias'] = original_od[f'gconv_net.gconvs.{i - 1}.net2.2.bias']

    return transfer_weights_biases
