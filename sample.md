## Problem

### Definition
In Multi-organ Histopathology Images, we are working on Conditional Segmentation under Compositional Textual Queries
where one images contains multi-classes
Queries can request single class, multiple class or joint pathological regions.
Masks are non-mutually exclusive and spatially overlapping.
So most systems are not working upon these type of things, so we can call it as Language conditioned Compositional Pathology Segmentation.

### Existing Methods
- CNNs and Transformers: Hard-coded label space, no conditioning on text, no reasoning, acts as a simple baseline.
- VL foundational Model (CLIP, BLIP): Trained for image-text alignment, and not for Dense Predictions, weak spatial grounding, no pathology-aware semantics. Thus, insufficient without architectural modification.
- Promptable Segmentation (SAM-Med): The prompts are Points or boxes, but not semantic pathological instructions and no pathology understanding. Can't distinguish overlapping histopathological entities and it fails clinically too.
- Referring Expression Segmentation (RefCOCO, LAVT): Meant for Natural images, single referred objects, no Multi-label logic and no overlapping regions supervision. It is not directly and applicable.

## Methodology (proposed)
### Datasets and Information
We have a dataset of 2200 Histopathological Images from H2TData and we have a csv file for each of the image and its respective multi-labels. By using this information, we've created a referring segmentation text for each of the images uniquely, the text containing all the classes in the sentence, later we used permutation and combinations, and made compositional combinations where one image will have 2^n instances where n is the number of classes the image belongs to. Now we can use the set valued Supervision i.e., one image with multiple text and mask pairs for its 2^n instances and by this we can do the conditional masking so for the same image we can get different outputs based on the instruction, and also multi-organ generalization where pathology semantics can be separated from organ context. This allows instruction following segmentation which is rare in Histopathology. 

## Model Proposal
CIPS-Net: Compositional Instruction - conditioned Pathology Segmentation Network

This architecture will have an Image Encoder which is defaultly pre-trained on Image Net or if possible on Histopathology where we can get the dense patch embeddings as output and a textual encoder where we get the token embeddings and sentence embeddings as output, now for the fusion part, we propose instruction grounding module which is based on compositional graph reasoning. Where the Nodes are pathology classes, Edges are co-occurrence and interaction, an instruction activates a subgraph and the Decoder segments based on activated subgraph. This instruction grounding module remains the key novality in our work. The decoder is the same UNet style conditiond on grounded visual text features and the output is the binary mask for the querried instructions



segmenta: 1,2,3,4,5 - multiclasses
cips-net: 1,2,3,4,5 - one mask (all segmentaion mask, dataset lo ala ravali)
cipsnet: (present all, but asked only 1,4) - binary mask with only 1 and 4 entities segmented ravali
