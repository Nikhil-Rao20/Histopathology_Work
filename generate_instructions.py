"""
Generate Text Instructions for PanNuke Dataset

Creates 300+ diverse instruction templates for referring segmentation
and populates the annotations.csv with generated instructions.
"""

import pandas as pd
import random
from pathlib import Path

# ============================================================================
# 300+ Instruction Templates
# ============================================================================
# {classes} will be replaced with the actual class names (e.g., "Neoplastic and Inflammatory")
# {tissue} will be replaced with the tissue type (e.g., "Breast")

TEMPLATES = [
    # === Simple Segmentation Commands (1-50) ===
    "Segment the {classes} regions.",
    "Segment {classes} in this image.",
    "Segment all {classes} areas.",
    "Segment the {classes} cells.",
    "Segment {classes} tissue.",
    "Identify and segment {classes}.",
    "Find and segment {classes} regions.",
    "Locate and segment {classes}.",
    "Detect and segment {classes} areas.",
    "Segment visible {classes}.",
    "Segment the {classes} structures.",
    "Segment {classes} nuclei.",
    "Segment all visible {classes}.",
    "Segment the {classes} components.",
    "Segment {classes} elements.",
    "Please segment {classes}.",
    "Segment the {classes} portions.",
    "Segment any {classes} present.",
    "Segment the {classes} sections.",
    "Segment {classes} zones.",
    "Segment the {classes} regions in the image.",
    "Segment all {classes} cells.",
    "Segment {classes} areas in this slide.",
    "Segment the {classes} tissue regions.",
    "Segment visible {classes} cells.",
    "Segment the {classes} nuclei.",
    "Segment all {classes} structures.",
    "Segment {classes} in this histology image.",
    "Segment the {classes} in this tissue sample.",
    "Segment any {classes} cells visible.",
    "Segment the {classes} cell populations.",
    "Segment all {classes} in the field.",
    "Segment {classes} throughout the image.",
    "Segment the entire {classes} population.",
    "Segment each {classes} region.",
    "Segment the {classes} distribution.",
    "Segment areas containing {classes}.",
    "Segment regions with {classes}.",
    "Segment tissue showing {classes}.",
    "Segment the {classes} infiltration.",
    "Segment zones of {classes}.",
    "Segment foci of {classes}.",
    "Segment clusters of {classes}.",
    "Segment patches of {classes}.",
    "Segment sites of {classes}.",
    "Segment locations of {classes}.",
    "Segment boundaries of {classes}.",
    "Segment margins of {classes}.",
    "Segment extent of {classes}.",
    "Segment the {classes} spread.",
    
    # === Highlight/Mark Commands (51-100) ===
    "Highlight the {classes} regions.",
    "Highlight {classes} in this image.",
    "Highlight all {classes} areas.",
    "Highlight {classes} cells.",
    "Highlight the {classes} tissue.",
    "Mark the {classes} regions.",
    "Mark {classes} in this slide.",
    "Mark all {classes} cells.",
    "Mark the {classes} areas.",
    "Mark visible {classes}.",
    "Outline the {classes} regions.",
    "Outline {classes} areas.",
    "Outline all {classes} structures.",
    "Outline the {classes} boundaries.",
    "Outline {classes} cells.",
    "Delineate the {classes} regions.",
    "Delineate {classes} areas.",
    "Delineate {classes} boundaries.",
    "Delineate the {classes} tissue.",
    "Delineate all {classes}.",
    "Circle the {classes} regions.",
    "Circle {classes} cells.",
    "Circle all {classes} areas.",
    "Circle the {classes} structures.",
    "Circle visible {classes}.",
    "Trace the {classes} regions.",
    "Trace {classes} boundaries.",
    "Trace all {classes} areas.",
    "Trace the {classes} outlines.",
    "Trace {classes} margins.",
    "Annotate the {classes} regions.",
    "Annotate {classes} cells.",
    "Annotate all {classes} areas.",
    "Annotate the {classes} tissue.",
    "Annotate visible {classes}.",
    "Label the {classes} regions.",
    "Label {classes} cells.",
    "Label all {classes} areas.",
    "Label the {classes} tissue.",
    "Label visible {classes}.",
    "Indicate the {classes} regions.",
    "Indicate {classes} areas.",
    "Indicate all {classes} cells.",
    "Indicate the {classes} tissue.",
    "Indicate where {classes} is present.",
    "Point out the {classes} regions.",
    "Point out {classes} cells.",
    "Point out all {classes}.",
    "Point out the {classes} areas.",
    "Point out visible {classes}.",
    
    # === Identification Commands (101-150) ===
    "Identify the {classes} regions.",
    "Identify {classes} in this image.",
    "Identify all {classes} cells.",
    "Identify the {classes} areas.",
    "Identify visible {classes}.",
    "Identify and mark {classes}.",
    "Identify {classes} tissue regions.",
    "Identify the {classes} structures.",
    "Identify all {classes} nuclei.",
    "Identify {classes} components.",
    "Locate the {classes} regions.",
    "Locate {classes} in this slide.",
    "Locate all {classes} cells.",
    "Locate the {classes} areas.",
    "Locate visible {classes}.",
    "Find the {classes} regions.",
    "Find {classes} in this image.",
    "Find all {classes} cells.",
    "Find the {classes} areas.",
    "Find visible {classes}.",
    "Detect the {classes} regions.",
    "Detect {classes} in this slide.",
    "Detect all {classes} cells.",
    "Detect the {classes} areas.",
    "Detect visible {classes}.",
    "Recognize the {classes} regions.",
    "Recognize {classes} cells.",
    "Recognize all {classes} areas.",
    "Recognize the {classes} tissue.",
    "Recognize visible {classes}.",
    "Pinpoint the {classes} regions.",
    "Pinpoint {classes} cells.",
    "Pinpoint all {classes} areas.",
    "Pinpoint the {classes} tissue.",
    "Pinpoint visible {classes}.",
    "Spot the {classes} regions.",
    "Spot {classes} cells.",
    "Spot all {classes} areas.",
    "Spot the {classes} tissue.",
    "Spot visible {classes}.",
    "Discern the {classes} regions.",
    "Discern {classes} cells.",
    "Discern all {classes} areas.",
    "Discern the {classes} from background.",
    "Discern visible {classes}.",
    "Distinguish the {classes} regions.",
    "Distinguish {classes} cells.",
    "Distinguish all {classes} areas.",
    "Distinguish {classes} from other tissue.",
    "Distinguish visible {classes}.",
    
    # === Medical/Clinical Style (151-200) ===
    "Segment the {classes} histological features.",
    "Identify histological regions showing {classes}.",
    "Mark the pathological {classes} regions.",
    "Delineate {classes} pathology in this section.",
    "Segment {classes} histopathological features.",
    "Identify {classes} cellular morphology.",
    "Mark {classes} tissue architecture.",
    "Segment microscopic {classes} features.",
    "Identify {classes} in this histology section.",
    "Segment {classes} in this tissue specimen.",
    "Mark {classes} in this biopsy sample.",
    "Identify {classes} in this pathology slide.",
    "Segment {classes} diagnostic regions.",
    "Mark {classes} lesion areas.",
    "Identify {classes} abnormalities.",
    "Segment {classes} in this microscopy image.",
    "Mark cellular {classes} regions.",
    "Identify nuclear {classes} features.",
    "Segment {classes} tissue morphology.",
    "Mark {classes} cellular infiltration.",
    "Identify {classes} in the tissue microenvironment.",
    "Segment {classes} in the tumor stroma.",
    "Mark {classes} distribution pattern.",
    "Identify {classes} spatial arrangement.",
    "Segment {classes} cellular composition.",
    "Mark regions of {classes} accumulation.",
    "Identify areas of {classes} concentration.",
    "Segment zones of {classes} presence.",
    "Mark foci of {classes} activity.",
    "Identify clusters of {classes} cells.",
    "Segment the {classes} cell population.",
    "Mark the {classes} tissue compartment.",
    "Identify the {classes} microenvironment.",
    "Segment {classes} in relation to stroma.",
    "Mark {classes} adjacent to epithelium.",
    "Identify peritumoral {classes}.",
    "Segment intratumoral {classes}.",
    "Mark stromal {classes} regions.",
    "Identify parenchymal {classes}.",
    "Segment {classes} in the tissue matrix.",
    "Annotate {classes} histological patterns.",
    "Delineate {classes} morphological features.",
    "Outline {classes} architectural patterns.",
    "Highlight {classes} cellular characteristics.",
    "Segment {classes} phenotypic features.",
    "Mark {classes} cytological features.",
    "Identify {classes} nuclear characteristics.",
    "Segment {classes} membrane patterns.",
    "Mark {classes} cytoplasmic features.",
    "Identify {classes} chromatin patterns.",
    
    # === Tissue-Specific Context (201-250) ===
    "Segment {classes} in this {tissue} tissue.",
    "Identify {classes} regions in the {tissue}.",
    "Mark {classes} cells in this {tissue} sample.",
    "Segment {classes} in the {tissue} section.",
    "Identify {classes} in {tissue} histology.",
    "Mark {classes} in this {tissue} specimen.",
    "Segment {classes} areas in {tissue} tissue.",
    "Identify {classes} features in {tissue}.",
    "Mark {classes} structures in {tissue}.",
    "Segment {classes} patterns in {tissue}.",
    "Highlight {classes} in this {tissue} slide.",
    "Delineate {classes} in {tissue} pathology.",
    "Outline {classes} in the {tissue} image.",
    "Annotate {classes} in {tissue} microscopy.",
    "Label {classes} in this {tissue} section.",
    "Locate {classes} in {tissue} tissue.",
    "Find {classes} in this {tissue} sample.",
    "Detect {classes} in {tissue} histology.",
    "Recognize {classes} in {tissue} pathology.",
    "Pinpoint {classes} in this {tissue}.",
    "In this {tissue} tissue, segment {classes}.",
    "For this {tissue} sample, mark {classes}.",
    "Within the {tissue}, identify {classes}.",
    "In the {tissue} section, locate {classes}.",
    "From this {tissue} slide, segment {classes}.",
    "On this {tissue} image, highlight {classes}.",
    "In {tissue} histology, delineate {classes}.",
    "For {tissue} pathology, outline {classes}.",
    "Within {tissue} tissue, annotate {classes}.",
    "In the {tissue} specimen, label {classes}.",
    "This is a {tissue} image. Segment {classes}.",
    "This {tissue} slide shows {classes}. Segment it.",
    "Given this {tissue} sample, mark {classes} regions.",
    "In this {tissue} histology, identify {classes} cells.",
    "For this {tissue} biopsy, segment {classes} areas.",
    "Analyze this {tissue} and segment {classes}.",
    "Examine this {tissue} tissue for {classes}.",
    "Study this {tissue} section and mark {classes}.",
    "Review this {tissue} slide and identify {classes}.",
    "Inspect this {tissue} sample for {classes}.",
    "From the {tissue} tissue, extract {classes} regions.",
    "In the {tissue} microstructure, find {classes}.",
    "Within {tissue} architecture, locate {classes}.",
    "Across the {tissue} section, segment {classes}.",
    "Throughout the {tissue} tissue, mark {classes}.",
    "Observe {classes} distribution in {tissue}.",
    "Note {classes} presence in {tissue} tissue.",
    "Document {classes} in this {tissue} sample.",
    "Record {classes} regions in {tissue}.",
    "Map {classes} distribution in {tissue}.",
    
    # === Question-Style Prompts (251-280) ===
    "Where is the {classes} in this image?",
    "Where are the {classes} regions?",
    "Where can you find {classes}?",
    "Where is {classes} located?",
    "Where are {classes} cells visible?",
    "Which regions show {classes}?",
    "Which areas contain {classes}?",
    "Which parts have {classes}?",
    "Which cells are {classes}?",
    "Which structures are {classes}?",
    "What areas show {classes}?",
    "What regions contain {classes}?",
    "What parts are {classes}?",
    "Show me the {classes} regions.",
    "Show me where {classes} is.",
    "Show me {classes} cells.",
    "Show me the {classes} areas.",
    "Show me all {classes}.",
    "Can you segment {classes}?",
    "Can you identify {classes}?",
    "Can you find {classes}?",
    "Can you locate {classes}?",
    "Can you mark {classes}?",
    "Please show {classes} regions.",
    "Please mark {classes} areas.",
    "Please identify {classes} cells.",
    "Please locate {classes}.",
    "Please find {classes}.",
    "Please highlight {classes}.",
    "Please segment {classes} tissue.",
    
    # === Action-Oriented (281-320) ===
    "Extract {classes} regions from this image.",
    "Extract all {classes} areas.",
    "Extract {classes} cell boundaries.",
    "Isolate {classes} regions.",
    "Isolate {classes} cells.",
    "Isolate {classes} from background.",
    "Separate {classes} from other tissue.",
    "Separate {classes} regions.",
    "Separate {classes} cells.",
    "Partition {classes} areas.",
    "Partition {classes} regions.",
    "Divide {classes} from surroundings.",
    "Define {classes} boundaries.",
    "Define {classes} regions.",
    "Define {classes} extent.",
    "Determine {classes} location.",
    "Determine {classes} distribution.",
    "Determine {classes} presence.",
    "Map the {classes} regions.",
    "Map {classes} distribution.",
    "Map {classes} locations.",
    "Map all {classes} areas.",
    "Chart {classes} presence.",
    "Chart {classes} distribution.",
    "Plot {classes} locations.",
    "Draw {classes} boundaries.",
    "Draw {classes} outlines.",
    "Draw around {classes}.",
    "Sketch {classes} regions.",
    "Sketch {classes} boundaries.",
    "Create mask for {classes}.",
    "Create segmentation of {classes}.",
    "Generate mask for {classes}.",
    "Generate {classes} segmentation.",
    "Produce {classes} mask.",
    "Produce segmentation of {classes}.",
    "Output {classes} regions.",
    "Output {classes} mask.",
    "Provide {classes} segmentation.",
    "Provide mask for {classes}.",
    
    # === Descriptive/Contextual (321-360) ===
    "This image contains {classes}. Segment it.",
    "There is {classes} present. Mark it.",
    "{classes} is visible. Segment the regions.",
    "{classes} can be seen. Identify and segment.",
    "The tissue shows {classes}. Mark the areas.",
    "{classes} is present in this slide. Segment it.",
    "This section has {classes}. Locate and mark.",
    "{classes} appears in this image. Segment all instances.",
    "The sample contains {classes}. Delineate the regions.",
    "{classes} is evident. Highlight the areas.",
    "Focus on {classes} in this image.",
    "Concentrate on {classes} regions.",
    "Pay attention to {classes} areas.",
    "Look for {classes} in this slide.",
    "Search for {classes} regions.",
    "Scan for {classes} cells.",
    "Examine {classes} distribution.",
    "Analyze {classes} presence.",
    "Assess {classes} regions.",
    "Evaluate {classes} areas.",
    "Segment only {classes}.",
    "Mark only {classes} regions.",
    "Identify only {classes} cells.",
    "Focus solely on {classes}.",
    "Consider only {classes} areas.",
    "Target {classes} regions.",
    "Target {classes} cells.",
    "Target {classes} in this image.",
    "Emphasize {classes} regions.",
    "Emphasize {classes} presence.",
    "The task is to segment {classes}.",
    "Your goal is to identify {classes}.",
    "The objective is to mark {classes}.",
    "You need to segment {classes}.",
    "You should identify {classes}.",
    "You must mark {classes} regions.",
    "Segmentation target: {classes}.",
    "Target class: {classes}.",
    "Classes to segment: {classes}.",
    "Required: {classes} segmentation.",
    
    # === Detailed Instructions (361-400) ===
    "Carefully segment all {classes} regions visible in this histopathology image.",
    "Precisely delineate the boundaries of {classes} cells in this tissue section.",
    "Accurately identify and mark all {classes} areas present in the slide.",
    "Thoroughly segment every instance of {classes} in this microscopy image.",
    "Meticulously outline the {classes} regions for analysis.",
    "Segment the {classes} with attention to boundary precision.",
    "Mark {classes} ensuring complete coverage of affected areas.",
    "Identify {classes} including all visible instances.",
    "Segment {classes} being careful to capture all regions.",
    "Delineate {classes} with accurate boundary detection.",
    "Provide a complete segmentation of {classes} regions.",
    "Generate a comprehensive mask for {classes}.",
    "Create an accurate segmentation of {classes} areas.",
    "Produce a detailed outline of {classes} structures.",
    "Develop a precise map of {classes} distribution.",
    "Segment {classes} for quantitative analysis.",
    "Mark {classes} for morphometric study.",
    "Identify {classes} for diagnostic purposes.",
    "Segment {classes} for pathological assessment.",
    "Delineate {classes} for clinical evaluation.",
    "Segment {classes} to enable cell counting.",
    "Mark {classes} for area measurement.",
    "Identify {classes} for spatial analysis.",
    "Segment {classes} for feature extraction.",
    "Delineate {classes} for pattern recognition.",
    "In this histopathology image, please segment {classes}.",
    "Looking at this tissue section, identify {classes}.",
    "Examining this slide, mark all {classes} regions.",
    "Analyzing this sample, segment {classes} areas.",
    "Reviewing this image, delineate {classes} boundaries.",
    "Given this pathology slide, segment {classes}.",
    "Presented with this tissue, identify {classes}.",
    "Shown this histology, mark {classes}.",
    "Viewing this section, segment {classes}.",
    "Observing this sample, delineate {classes}.",
    "Your task: segment {classes} in this image.",
    "Objective: identify {classes} regions.",
    "Goal: mark all {classes} cells.",
    "Purpose: delineate {classes} boundaries.",
    "Assignment: segment {classes} tissue.",
]

# Add more templates programmatically
VERBS = ["Segment", "Identify", "Mark", "Highlight", "Delineate", "Outline", "Locate", "Find", "Detect", "Annotate"]
OBJECTS = ["regions", "areas", "cells", "tissue", "structures", "nuclei", "components", "features", "zones", "sections"]
MODIFIERS = ["all", "visible", "present", "detected", "observed", "apparent", "discernible", "identifiable", ""]

# Generate additional templates
for verb in VERBS[:5]:
    for obj in OBJECTS[:5]:
        for mod in MODIFIERS[:3]:
            if mod:
                template = f"{verb} {mod} {{classes}} {obj}."
            else:
                template = f"{verb} the {{classes}} {obj}."
            if template not in TEMPLATES:
                TEMPLATES.append(template)

print(f"Total templates: {len(TEMPLATES)}")


def format_classes(classes_str):
    """
    Convert semicolon-separated classes to natural language.
    e.g., "Neoplastic;Inflammatory;Dead" -> "Neoplastic, Inflammatory and Dead"
    """
    if not classes_str or pd.isna(classes_str):
        return ""
    
    classes = classes_str.split(';')
    
    # Clean up class names for readability
    clean_classes = []
    for c in classes:
        # Replace underscores with spaces for readability
        clean = c.replace('_', ' ').lower()
        clean_classes.append(clean)
    
    if len(clean_classes) == 1:
        return clean_classes[0]
    elif len(clean_classes) == 2:
        return f"{clean_classes[0]} and {clean_classes[1]}"
    else:
        return ", ".join(clean_classes[:-1]) + f" and {clean_classes[-1]}"


def generate_instruction(classes_str, tissue_type):
    """Generate a random instruction for the given classes and tissue type."""
    if not classes_str or pd.isna(classes_str):
        return ""
    
    # Format classes nicely
    formatted_classes = format_classes(classes_str)
    
    # Pick a random template
    template = random.choice(TEMPLATES)
    
    # Replace placeholders
    instruction = template.replace("{classes}", formatted_classes)
    instruction = instruction.replace("{tissue}", tissue_type.replace('_', ' '))
    
    return instruction


def main():
    print("="*60)
    print("Generating Text Instructions for PanNuke")
    print("="*60)
    print(f"Number of templates: {len(TEMPLATES)}")
    
    # Load annotations
    annotations_path = "PanNuke_Preprocess/annotations.csv"
    df = pd.read_csv(annotations_path)
    print(f"Loaded {len(df)} samples")
    
    # Generate instructions
    print("Generating instructions...")
    df['instruction'] = df.apply(
        lambda row: generate_instruction(row['classes_present'], row['tissue_type']),
        axis=1
    )
    
    # Save updated annotations
    df.to_csv(annotations_path, index=False)
    print(f"Saved updated annotations to {annotations_path}")
    
    # Show some examples
    print("\n" + "="*60)
    print("Sample Generated Instructions:")
    print("="*60)
    for i in range(10):
        sample = df.iloc[i]
        print(f"\nImage: {sample['image_id']}")
        print(f"Classes: {sample['classes_present']}")
        print(f"Tissue: {sample['tissue_type']}")
        print(f"Instruction: {sample['instruction']}")
    
    # Show instruction diversity
    print("\n" + "="*60)
    print("Instruction Statistics:")
    print("="*60)
    unique_instructions = df['instruction'].nunique()
    print(f"Unique instructions: {unique_instructions}")
    print(f"Total samples: {len(df)}")
    print(f"Diversity ratio: {unique_instructions/len(df)*100:.1f}%")
    
    # Show some random examples from different tissue types
    print("\n" + "="*60)
    print("Examples from Different Tissue Types:")
    print("="*60)
    for tissue in ['Breast', 'Colon', 'Liver', 'Lung', 'Kidney']:
        sample = df[df['tissue_type'] == tissue].sample(1).iloc[0]
        print(f"\n[{tissue}] {sample['instruction']}")


if __name__ == "__main__":
    main()
