# 🐶 STL-10 Image Classification

> Image classification on **STL-10 dataset** using **EfficientNet-B0** with PyTorch.  
> A compact and efficient CNN achieving ~80% accuracy on 10 object classes.

---

## 🧩 Dataset Overview

**STL-10 Dataset**

> 10 categories: ✈️ airplane, 🐦 bird, 🚗 car, 🐱 cat, 🦌 deer, 🐕 dog, 🐎 horse, 🐒 monkey, 🚢 ship, 🚚 truck.

| Split | #Images | Per Class | Image Size |
|:------|:--------:|:----------:|:------------:|
| Train | 5,000 | 500 | 96×96×3 |
| Test  | 8,000 | 800 | 96×96×3 |

---

## 🧠 Model Architecture

**EfficientNet-B0**
- Pre-trained on **ImageNet**
- ~5.3M parameters
- Replaced final fully-connected (FC) layer → 10 output classes

---

## ⚙️ Installation
# Model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)

# Loss & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

# Hyperparameters
batch_size = 64
epochs = 20
# Train model
python train.py

# Evaluate performance
python evaluate.py --model_path checkpoints/best_model.pth

# Predict a single image
python predict.py --image_path image.jpg --model_path checkpoints/best_model.pth
STL_10/
├── data/               # Auto-downloaded dataset
├── checkpoints/        # Saved model weights
├── assets/             # Images, charts, visuals
│   ├── model_architecture.png
│   └── training_curve.png
├── train.py           
├── evaluate.py        
├── predict.py         
└── README.md


```bash
pip install torch torchvision efficientnet-pytorch matplotlib numpy
