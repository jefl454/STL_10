# STL-10 Image Classification

Image classification on STL-10 dataset using EfficientNet-B0 with PyTorch.

## Dataset

**STL-10**: 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- Training: 5,000 images (500/class)
- Test: 8,000 images (800/class)  
- Image size: 96x96x3

## Model Architecture

**EfficientNet-B0**
- Pre-trained on ImageNet
- ~5.3M parameters
- Modified final FC layer for 10 classes

## Installation

```bash
pip install torch torchvision efficientnet-pytorch matplotlib numpy
```

## Training Configuration

```python
# Model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)

# Loss & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

# Hyperparameters
batch_size = 64
epochs = 50
```

**Key choices**:
- `lr=0.0003`: Lower LR for stable fine-tuning (reduced from 0.001)
- `weight_decay=1e-4`: L2 regularization to prevent overfitting on small dataset

## Usage

```bash
# Train
python train.py

# Evaluate  
python evaluate.py --model_path checkpoints/best_model.pth

# Predict
python predict.py --image_path image.jpg --model_path checkpoints/best_model.pth
```

## Expected Results

- **Validation accuracy**: ~75-80%
- **Training time**: ~30-45 min on GPU (50 epochs)

## Project Structure

```
STL_10/
├── data/               # Auto-downloaded dataset
├── checkpoints/        # Saved models
├── train.py           
├── evaluate.py        
├── predict.py         
└── README.md
```

## Improvement Tips

- Add learning rate scheduler (ReduceLROnPlateau, CosineAnnealing)
- Advanced augmentation (AutoAugment, RandAugment)
- Use unlabeled data (100k images) for semi-supervised learning
- Try larger models (EfficientNet-B1/B2)

## References

- [STL-10 Dataset](https://cs.stanford.edu/~acoates/stl10/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## License

MIT License
