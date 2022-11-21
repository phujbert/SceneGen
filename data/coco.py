import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
import os, random, math
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from data.utils import Resize, imagenet_preprocess, load_transfer_weights_biases
from PIL import Image
from skimage.transform import resize as imresize
from gcn import GraphConvNet


class CocoStuffDataset(Dataset):

    def __init__(self, image_dir, instances_json, stuff_json, device, image_size=(64, 64),
                 mask_size=16, min_object_size=0.02, min_objects_per_image=2, max_objects_per_image=4):
        super(Dataset, self).__init__()
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.min_object_size = min_object_size
        self.device = device

        transform = [Resize(image_size), T.ToTensor(), imagenet_preprocess()]
        self.transform = T.Compose(transform)
        self.image_size = image_size

        self.coco = COCO(instances_json)
        self.coco_stuff = COCO(stuff_json)

        embedding_dim = 128
        transfer_module_path = "transfer_models/coco64.pt"

        self.gcn = GraphConvNet(input_dim=embedding_dim, hidden_dim=512, output_dim=embedding_dim)
        transfer_weights_biases = load_transfer_weights_biases(transfer_module_path, device)
        self.gcn.load_state_dict(transfer_weights_biases)

        # get image ids and dictionary of all objects with the specified constraints:
        # - minimum number of objects per image
        # - maximum number of objects per image
        # - minimum object size
        image_ids = []
        self.all_objs = set()
        for image_id in self.coco.getImgIds():
            object_datas = self.get_object_datas(image_id, areaRng=[self.min_object_size, float('inf')])
            if min_objects_per_image <= len(object_datas) <= max_objects_per_image:
                image_ids.append(image_id)
                for object_data in object_datas:
                    self.all_objs.add(object_data['category_id'])
        self.image_ids = image_ids

        self.edges = {
            "__in_image__": 0,
            "left of": 1,
            "right of": 2,
            "above": 3,
            "below": 4,
            "inside": 5,
            "surrounding": 6
        }
        self.in_image = 0

        self.cat_ids = self.coco.getCatIds() + self.coco_stuff.getCatIds()

    def __len__(self):
        return len(self.image_ids)

    def get_image_triples(self, image_id, original_width, original_height):
        objs, boxes, masks = [], [], []
        object_datas = self.get_object_datas(image_id)
        for object_data in object_datas:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            x0 = x / original_width
            y0 = y / original_height
            x1 = (x + w) / original_width
            y1 = (y + h) / original_height
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

            mask = seg_to_mask(object_data['segmentation'], original_width, original_height)

            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            mask = mask[my0:my1, mx0:mx1]
            mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                            mode='constant')
            mask = torch.from_numpy((mask > 128).astype(np.int64))
            masks.append(mask)

        # Adding image object to the objects to later specify edges between the real objects on the image with the image
        objs.append(self.in_image)
        boxes.append(torch.FloatTensor([0, 0, 1, 1]))
        masks.append(torch.ones(self.mask_size, self.mask_size).long())

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        obj_centers = []
        _, MH, MW = masks.size()
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mask = (masks[i] == 1)
            xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)

        triples = []
        triple_objs = objs.nonzero().squeeze(1)
        for cur in triple_objs:
            choices = [obj for obj in triple_objs if obj != cur]
            if len(choices) == 0:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur

            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])
            if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                p = 'surrounding'
            elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'
            p = self.edges[p]
            triples.append([s, p, o])

        O = objs.size(0)
        in_image = self.edges["__in_image__"]
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return objs, triples

    def get_graph_emb(self, objs, triples):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        preds = p
        edges = torch.stack([s, o], dim=1)
        with torch.no_grad():
            obj_vecs = self.gcn(objs, preds, edges)
        graph_emb = torch.sum(obj_vecs, dim=0, keepdim=True)
        return graph_emb

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.coco.loadImgs(image_id)
        if len(img) > 1: print('More image with this id')
        filename = img[0]['file_name']
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                original_width, original_height = image.size
                image = self.transform(image.convert('RGB'))

        objs, triples = self.get_image_triples(image_id, original_width, original_height)

        graph_emb = self.get_graph_emb(objs, triples)

        return image, graph_emb

    def get_object_datas(self, image_id, areaRng=[]):
        instances_ann_ids = self.coco.getAnnIds(imgIds=image_id, areaRng=areaRng)
        stuff_ann_ids = self.coco_stuff.getAnnIds(imgIds=image_id, areaRng=areaRng)
        object_datas = self.coco.loadAnns(instances_ann_ids) + self.coco_stuff.loadAnns(stuff_ann_ids)
        return object_datas

    def sample_graphs(self, n_graphs):
        graphs = []
        graph_embs = []
        edge_dict = dict([(value, key) for key, value in self.edges.items()])
        for _ in range(n_graphs):
            image_id = random.choice(self.image_ids)
            img = self.coco.loadImgs(image_id)
            original_width, original_height = img[0]["width"], img[0]["height"]
            objs, triples = self.get_image_triples(image_id, original_width, original_height)
            graph_emb = self.get_graph_emb(objs, triples)
            objs = objs.numpy()
            triples = triples.numpy()
            graph = []
            for triple in triples:
                obj_id = int(objs[triple[0]])
                subj_id = int(objs[triple[2]])
                if obj_id == 0:
                    object = 'image'
                elif obj_id < 92:
                    object = self.coco.loadCats(ids=[obj_id])[0]['name']
                else:
                    object = self.coco_stuff.loadCats(ids=[obj_id])[0]['name']

                if subj_id == 0:
                    subject = 'image'
                elif subj_id < 92:
                    subject = self.coco.loadCats(ids=[subj_id])[0]['name']
                else:
                    subject = self.coco_stuff.loadCats(ids=[subj_id])[0]['name']

                temp = [object, edge_dict[triple[1]], subject]
                graph.append(temp)
            graph_data = {
                "img_id": image_id,
                "graph": graph
            }
            graphs.append(graph_data)
            graph_embs.append(graph_emb)
        graph_embs = torch.cat(graph_embs)
        graph_embs = graph_embs.to(self.device)
        return graphs, graph_embs



def seg_to_mask(seg, width=1.0, height=1.0):
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)


