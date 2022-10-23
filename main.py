from data.coco import CocoStuffDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_data(data_type, batch_size, num_workers, shuffle):
    image_dir = f'coco_dataset/images/{data_type}2017'
    instances_json = f'coco_dataset/annotations/instances_{data_type}2017.json'
    stuff_json = f'coco_dataset/annotations/stuff_{data_type}2017.json'

    dataset = CocoStuffDataset(image_dir, instances_json, stuff_json)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


if __name__ == '__main__':
    train, val = "train", "val"
    batch_size = 1
    num_workers = 4

    val_dataloader = load_data(val, batch_size, num_workers, shuffle=False)
    train_dataloader = load_data(train, batch_size, num_workers, shuffle=False)
    data_iter = iter(val_dataloader)
    data = data_iter.next()
    all_imgs, all_objs, all_boxes, all_masks, all_triples = data
    print(all_triples)
