# SimpleVLM Training Notebook

This notebook trains the SimpleVLM model (Vision Encoder + Text Encoder + UNet Decoder) for multi-class histopathology segmentation.

## Architecture
- **Vision Encoder**: ResNet50 (pretrained on ImageNet)
- **Text Encoder**: DistilBERT (pretrained)
- **Fusion**: FiLM (Feature-wise Linear Modulation)
- **Decoder**: UNet with skip connections
- **Output**: 5-channel masks (one per tissue class)

## Advantages over CIPS-Net
1. **Simpler**: No complex graph reasoning module
2. **Faster**: Lighter architecture, faster training
3. **Proven**: Based on well-established UNet + vision-language fusion
4. **Flexible**: Easy to swap encoders (ResNet, ViT, EfficientNet, etc.)
