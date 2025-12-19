# CIPS-Net: Compositional Instruction-conditioned Pathology Segmentation Network

A novel Vision-Language Model for instruction-guided histopathology image segmentation using compositional graph reasoning.

## 🔬 Overview

CIPS-Net is a state-of-the-art model for multi-class histopathology segmentation that:
- Takes natural language instructions as input
- Uses compositional graph reasoning to understand class relationships
- Generates precise segmentation masks for multiple pathology classes
- Handles complex multi-organ histopathology data

## 🏗️ Architecture

### 1. **Image Encoder** 
- Vision Transformer (ViT-B/ViT-L)
- Pretrained on histopathology images (or ImageNet + self-supervised)
- Extracts dense patch embeddings

### 2. **Text Encoder**
- Clinical-BERT / PubMedBERT
- Specialized for medical text understanding
- Outputs token and sentence embeddings

### 3. **Instruction Grounding Module** (KEY NOVELTY)
- **Compositional Graph Reasoning**
  - Nodes: Pathology classes (Neoplastic, Inflammatory, Connective_Soft_tissue, Epithelial, Dead)
  - Edges: Co-occurrence and semantic relationships
  - Instruction activates relevant subgraph
  - Multi-layer graph attention for reasoning

### 4. **Segmentation Decoder**
- UNet-style decoder with transformer components
- Class-specific segmentation heads
- Conditioned on grounded visual-text features

## 📊 Dataset

Multi-class histopathology segmentation dataset:
- **2,720** unique images
- **15** different organs (Breast, Colon, Liver, etc.)
- **5** pathology classes per image
- **24,326** instruction-image pairs (with permutations)
- **3-fold** cross-validation ready

### Classes:
1. **Neoplastic** - Cancer cells
2. **Inflammatory** - Inflammation
3. **Connective_Soft_tissue** - Connective tissue
4. **Epithelial** - Epithelial cells
5. **Dead** - Necrotic tissue

## 🚀 Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Project Structure

```
CIPS-Net/
├── models/
│   ├── __init__.py
│   ├── image_encoder.py          # ViT encoder
│   ├── text_encoder.py            # Clinical-BERT encoder
│   ├── instruction_grounding.py   # Graph reasoning (KEY NOVELTY)
│   ├── decoder.py                 # Segmentation decoder
│   └── cips_net.py                # Main model
├── utils/
│   ├── dataset.py                 # Dataset loader
│   ├── metrics.py                 # Evaluation metrics
│   └── visualization.py           # Visualization tools
├── configs/
│   └── config.py                  # Configuration
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── inference.py                   # Inference script
└── requirements.txt
```

## 💻 Usage

### Quick Test
```python
from models.cips_net import CIPSNet

# Initialize model
model = CIPSNet(
    img_encoder_name='vit_b_16',
    text_encoder_name="emilyalsentzer/Bio_ClinicalBERT",
    embed_dim=768,
    num_classes=5,
    pretrained=True
)

# Prepare inputs
images = torch.randn(2, 3, 224, 224)  # [B, 3, H, W]
instructions = [
    "Segment Neoplastic and Inflammatory regions in this Breast tissue.",
    "Identify Connective_Soft_tissue in the sample."
]

# Forward pass
outputs = model(images, instructions)
masks = outputs['masks']  # [B, 6, 224, 224] (5 classes + background)
```

### Training
```python
from utils.dataset import get_dataloaders
from configs.config import get_default_config

# Load configuration
config = get_default_config()

# Create dataloaders
train_loader, val_loader = get_dataloaders(
    data_root=config.data.data_root,
    batch_size=config.training.batch_size,
    train_folds=[1, 2],
    val_folds=[3]
)

# Train model (see train.py for full implementation)
# python train.py --config configs/default.yaml
```

### Inference
```python
model.eval()
predictions = model.predict(
    images=test_images,
    instructions=test_instructions,
    threshold=0.5
)

binary_masks = predictions['binary_masks']
mask_probs = predictions['mask_probs']
attention_scores = predictions['attention_scores']
```

## 🎯 Key Features

### 1. **Compositional Graph Reasoning**
- Explicitly models relationships between pathology classes
- Learns co-occurrence patterns from data
- Enables compositional understanding of complex instructions

### 2. **Instruction-Conditioned Segmentation**
- Flexible: handles various natural language descriptions
- Focused: only segments mentioned classes
- Interpretable: provides attention scores

### 3. **Multi-Class Support**
- Handles 1-5 simultaneous classes per image
- Class-specific guidance features
- Balanced loss for class imbalance

### 4. **Medical Domain Expertise**
- Pretrained on medical text (Clinical-BERT)
- Histopathology-specific image features
- Domain-informed graph structure

## 📈 Model Performance

Expected metrics (after training):
- **Dice Score**: >0.80 per class
- **IoU**: >0.70 per class
- **Pixel Accuracy**: >0.90

## 🔧 Configuration

Key hyperparameters in `configs/config.py`:

```python
# Model
embed_dim = 768
num_graph_layers = 3

# Training
learning_rate = 1e-4
batch_size = 8
num_epochs = 100

# Loss weights
seg_loss_weight = 1.0
presence_loss_weight = 0.1
attention_reg_weight = 0.01
```

## 📝 Citation

```bibtex
@article{cipsnet2024,
  title={CIPS-Net: Compositional Instruction-conditioned Pathology Segmentation Network},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Clinical-BERT: [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323)
- Dataset: Multi-organ Histopathology Segmentation

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com]

---

**Status**: 🚧 Under Development | **Version**: 1.0.0 | **Last Updated**: December 2024
