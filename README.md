# Image Generation from Scene Graphs with Diffusion - Képgenerálás jelenetgráfokból diffúzióval

### Hujbert Patrik - D83AE5

## About the project

The goal of this project is to implement such a neural network model, that can generate new images based on scene graphs. A scene graph is graph based on a visual scene, where the nodes represent the the object in the scene, and the edges represent relationships between the objects
As a base, the project is built on the research paper [Image Generation from Scene Graphs](https://arxiv.org/pdf/1804.01622.pdf), in which the researchers used a GAN model to generate realistic images that respects the scene graphs layout.
For my project I want to implement a diffusion model, which is used in state-of-the-art image generation tasks.

## Dataset

To train the model two datasets is used in the paper, I'd like to use only one of them, the [COCO-Stuff Dataset](https://github.com/nightrome/cocostuff).

The dataset is mainly used for segmentation problems, it contains 118K train images and 5K val images, with annotations (boxes, masks) about the objects on the images. An object can be either a thing or a stuff, which makes this dataset perfect to generate scene graphs. Detailed information about the dataset is in the paper [COCO-Stuff: Thing and Stuff Classes in Context](https://arxiv.org/pdf/1612.03716.pdf)

The code for the dataset class is in the data folder, coco.py contains the pytorch dataset class for the dataset.
### Download the dataset:
```
COCO_DIR=coco_dataset
mkdir -p $COCO_DIR

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $COCO_DIR/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip -O $COCO_DIR/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip -O $COCO_DIR/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $COCO_DIR/val2017.zip

unzip $COCO_DIR/annotations_trainval2017.zip -d $COCO_DIR/annotations
unzip $COCO_DIR/stuff_annotations_trainval2017.zip -d $COCO_DIR/annotations
unzip $COCO_DIR/train2017.zip -d $COCO_DIR/images
unzip $COCO_DIR/val2017.zip -d $COCO_DIR/images
```

### Run the code to load the dataset:
The code is developed with Python 3.9 and the packages needed can be found in requirements.txt
To load the data an API is used as well which can be installed from the [cocostuffapi](https://github.com/nightrome/cocostuffapi) github repo.
To run the code run the main.py script.

### Visualize the data
Visualization of the dataset can be found in the visualize_dataset.ipynb

## Train

Training the model requires additional training model, whick can be installed the following way:

```
wget https://storage.googleapis.com/sg2im-data/small/coco64.pt -O transfer_models/coco64.pt
```

To train the model run the script **main.py**

## Sample

After training the saved models can be found in the folder called trained_models

To test them and sample images run the script **sample.py**