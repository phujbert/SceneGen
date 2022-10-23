import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
import os, random, math
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from data.utils import Resize, imagenet_preprocess
from PIL import Image
from skimage.transform import resize as imresize


class CocoStuffDataset(Dataset):

    def __init__(self, image_dir, instances_json, stuff_json, image_size=(64, 64),
                 mask_size=16, min_object_size=0.02, min_objects_per_image=3, max_objects_per_image=8):
        super(Dataset, self).__init__()
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.min_object_size = min_object_size

        transform = [Resize(image_size), T.ToTensor(), imagenet_preprocess()]
        self.transform = T.Compose(transform)
        self.image_size = image_size

        self.coco = COCO(instances_json)
        self.coco_stuff = COCO(stuff_json)

        image_ids = []
        for image_id in self.coco.getImgIds():
            object_datas = self.get_object_datas(image_id)
            if min_objects_per_image <= len(object_datas) <= max_objects_per_image:
                image_ids.append(image_id)
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

    def __len__(self):
        return len(self.image_ids)

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

        objs, boxes, masks = [], [], []
        object_datas = self.get_object_datas(image_id, areaRng=[self.min_object_size, float('inf')])
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
        objs = objs.nonzero().squeeze(1)
        for cur in objs:
            choices = [obj for obj in objs if obj != cur]
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
        return image, objs, boxes, masks, triples

    def get_object_datas(self, image_id, areaRng=[]):
        instances_ann_ids = self.coco.getAnnIds(imgIds=image_id, areaRng=areaRng)
        stuff_ann_ids = self.coco_stuff.getAnnIds(imgIds=image_id, areaRng=areaRng)
        object_datas = self.coco.loadAnns(instances_ann_ids) + self.coco_stuff.loadAnns(stuff_ann_ids)
        return object_datas


def seg_to_mask(seg, width=1.0, height=1.0):
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)
