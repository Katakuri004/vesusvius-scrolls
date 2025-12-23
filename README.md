# Vesuvius Challenge: Ink Detection & Segmentation 📜

This repository contains a complete, high-performance deep learning pipeline for solving the **Vesuvius Challenge** surface detection task on an 8GB GPU.

## 🏆 Results
- **Dice Score**: `0.4398` (Validation)
- **Merge Score**: `1.0000` (Perfect Topology Baseline)
- **Visual Quality**: Clean segmentation of ink layers, successfully resolving the "blob" problem.

![Training Curves](img/training_curves.png)

## 🧠 System Architecture

### High-Level Data Pipeline
Our system processes high-resolution 3D X-ray CT volumes through a specialized pipeline designed to handle signal normalization and topological consistency.

![System Pipeline](img/graph_1.png)

The workflow consists of:
1.  **Normalization**: Auto-detection of 8-bit vs 16-bit encoding to ensure correct intensity scaling.
2.  **Patching**: Dynamic 3D patch generation `(32, 128, 128)` with positive sampling bias to handle extreme class imbalance (<1% ink).
3.  **Topology-Aware Training**: A composite loss function utilizing **clDice** to enforce structural continuity.
4.  **Inference Aggregation**: Sliding window prediction with Gaussian-weighted blending to eliminate seam artifacts.

### Model: Anisotropic 3D U-Net
We utilize a custom **Attention 3D U-Net** designed specifically for the anisotropic nature of scroll volume data.

![Model Architecture](img/graph_2.png)

**Key Innovations:**
- **Anisotropic Pooling**: Standard pooling `(2, 2, 2)` destroys Z-axis resolution. We use `(1, 2, 2)` kernels to downsample spatial dimensions while preserving depth, which is critical for resolving thin ink layers.
- **Attention Gates**: Additive attention modules at skip connections allow the decoder to focus on relevant ink features while suppressing background noise from the encoder features.
- **Lightweight Design**: Optimized for 8GB VRAM using gradient checkpointing, mixed precision (AMP), and a reduced feature base (`16` filters).

## 🛠️ Methodology
The core innovation lies in the data pipeline and loss function:

### 1. Data Rectification
- **8-Bit Normalization**: Fixed a critical bug where 8-bit volumes were treated as 16-bit, restoring signal visibility.
- **Target Labeling**: Explicitly trained on Label Value `1` (Ink) while ignoring Paper (`2`) and Void (`0`), solving the "everything is ink" issue.

### 2. Topology-Aware Training
- **clDice Loss (Centerline Dice)**: We integrated a soft skeletonization loss (`soft_cldice`) into the training.
- **Goal**: This forces the model to preserve the topological structure (skeleton) of the ink sheets, preventing mergers and breaks.
- **Composite Loss**: `Loss = 0.4*BCE + 0.4*Dice + 0.2*clDice`.

## 📸 Visual Results
Below is an example of the model's output on a validation slice. Note the clear separation of valid ink from the background noise.

![Model Output](img/model_output_showcase.png)

## 🚀 How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Trains model for 50 epochs with Cosine Annealing
# Checkpoints saved to models/
pip install jupyter
jupyter notebook Training.ipynb
```

### Submission Generation
```bash
# Generates submission.zip containing .tif masks
python submission.py
```

## 📂 File Structure
- `src/models/unet3d.py`: Model Architecture.
- `src/losses/cldice.py`: Topology Loss implementation.
- `src/data/dataset.py`: Robust 3D Datasets.
- `submission.py`: Inference pipeline.
