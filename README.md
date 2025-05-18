# Animal Image Classifier using PyTorch

This is a deep learning project using PyTorch and Transfer Learning to classify animal images. The model is based on ResNet18 pretrained on ImageNet.

## Features
- Transfer Learning with ResNet18
- Image classification (e.g., Dog vs Cat)
- Custom dataset support
- Easily extendable to more classes

## Project Structure
```
animal-image-classifier/
├── animal_classifier.py       # Main training script
├── requirements.txt           # List of required libraries
├── dataset/                   # Add your training and testing data here
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│       ├── cat/
│       └── dog/
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run Training
```bash
python animal_classifier.py
```

## Dataset
Place your images into the `dataset/train` and `dataset/test` folders with subfolders for each class.

## Author
baxtiyorov3407
