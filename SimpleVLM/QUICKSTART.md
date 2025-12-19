# SimpleVLM Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```python
import torch
import timm
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"TIMM: {timm.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

## Usage

### 1. Basic Model Initialization

```python
from SimpleVLM import SimpleVLMSegmenter

# Initialize model with default configuration
model = SimpleVLMSegmenter(
    vision_encoder='resnet50',
    text_encoder='distilbert-base-uncased',
    fusion_dim=512,
    fusion_method='film',  # or 'cross_attention', 'multi_scale'
    decoder_channels=[256, 128, 64, 32],
    num_classes=5
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Display model info
info = model.get_model_info()
print(f"Total Parameters: {info['total_params']:,}")
print(f"Vision Encoder: {info['vision_encoder_params']:,}")
print(f"Text Encoder: {info['text_encoder_params']:,}")
```

### 2. Training

Use the provided notebook:
```bash
jupyter notebook train_simple_vlm.ipynb
```

Or train programmatically:
```python
# See train_simple_vlm.ipynb for complete training code
# Key components:
# 1. Dataset loading
# 2. Data augmentation
# 3. Loss function (Combined BCE + Dice)
# 4. Training loop with per-class metrics
# 5. Model checkpointing
```

### 3. Inference

```python
import torch
from PIL import Image
import numpy as np

# Load trained model
checkpoint = torch.load('best_simple_vlm_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
image = Image.open('path/to/image.png').convert('RGB')
image = np.array(image)
# Apply preprocessing (resize, normalize)
# ... (see train_simple_vlm.ipynb for preprocessing)

# Prepare text instruction
instruction = "Segment Neoplastic and Inflammatory tissues"

# Run inference
with torch.no_grad():
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    output = model(image_tensor, [instruction])
    masks = torch.sigmoid(output)
    masks = (masks > 0.5).float()

# masks shape: [1, 5, H, W]
# Channel 0: Neoplastic
# Channel 1: Inflammatory
# Channel 2: Connective_Soft_tissue
# Channel 3: Epithelial
# Channel 4: Dead
```

## Configuration Options

### Vision Encoders
Supported via `timm` library:
```python
# ResNet family
vision_encoder='resnet50'  # Default, good balance
vision_encoder='resnet101'  # Larger, more capacity

# EfficientNet family
vision_encoder='efficientnet_b0'  # Lightweight
vision_encoder='efficientnet_b4'  # Heavier, more accurate

# Vision Transformer family
vision_encoder='vit_small_patch16_224'  # ViT-S
vision_encoder='vit_base_patch16_224'   # ViT-B
```

### Text Encoders
Supported via `transformers` library:
```python
# BERT family
text_encoder='distilbert-base-uncased'  # Default, fast
text_encoder='bert-base-uncased'         # Larger
text_encoder='roberta-base'              # Alternative

# Clinical/Biomedical models
text_encoder='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
text_encoder='emilyalsentzer/Bio_ClinicalBERT'
```

### Fusion Methods

**1. FiLM (Feature-wise Linear Modulation)** - Default, fastest
```python
fusion_method='film'
```
- Simplest fusion: learns affine transformation (gamma, beta)
- Formula: `output = gamma * vision + beta`
- Fast and effective

**2. Cross Attention** - More expressive
```python
fusion_method='cross_attention'
```
- Multi-head cross-attention between vision and text
- More parameters, captures complex relationships
- Slower but potentially more accurate

**3. Multi-Scale Fusion** - Most comprehensive
```python
fusion_method='multi_scale'
```
- Applies fusion at each encoder scale
- Best for capturing details at multiple resolutions
- Highest computational cost

### Decoder Configurations

```python
# Standard UNet with skip connections (default)
decoder_channels=[256, 128, 64, 32]  # 4 upsampling blocks

# Deeper decoder
decoder_channels=[512, 256, 128, 64, 32]  # 5 upsampling blocks

# Lighter decoder
decoder_channels=[128, 64, 32]  # 3 upsampling blocks
```

## Advanced Usage

### Custom Dataset

```python
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, idx):
        # Load image and masks
        image = ...  # [H, W, 3]
        masks = ...  # [H, W, num_classes]
        instruction = ...  # text string
        
        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']
        
        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'masks': torch.from_numpy(masks).permute(2, 0, 1).float(),
            'instruction': instruction
        }
```

### Transfer Learning

```python
# Load pretrained model
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze vision encoder for fine-tuning
for param in model.vision_encoder.parameters():
    param.requires_grad = False

# Train only decoder and fusion layers
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images, instructions)
        loss = criterion(outputs, masks)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Troubleshooting

### Out of Memory
1. Reduce batch size
2. Use gradient accumulation
3. Use mixed precision training
4. Choose lighter encoder (e.g., ResNet50 instead of ViT)

### Slow Training
1. Use FiLM fusion instead of CrossAttention
2. Use smaller encoder (ResNet50, EfficientNet-B0)
3. Reduce decoder channels
4. Enable mixed precision training

### Poor Performance
1. Try different fusion methods (CrossAttention, MultiScale)
2. Use larger encoder (ResNet101, ViT-B)
3. Increase decoder channels
4. Train for more epochs
5. Try domain-specific text encoder (Bio_ClinicalBERT)

## Performance Tips

1. **Start simple**: ResNet50 + DistilBERT + FiLM
2. **Monitor metrics**: Track per-class Dice/IoU scores
3. **Experiment systematically**: Change one component at a time
4. **Use pretrained models**: All encoders are pretrained by default
5. **Data augmentation**: Critical for good generalization

## Citation

If you use SimpleVLM in your research, please cite:
```
@software{simplevlm2024,
  title={SimpleVLM: A Simple Vision-Language Model for Referring Segmentation},
  author={Your Name},
  year={2024}
}
```
