# Deep Learning Systems Lab

This repository serves as a centralized workspace for deep learning research, experimental prototypes, and system-level explorations. It includes implementations across multiple domains such as adversarial robustness, generative modeling, computer vision, and optimization-driven learning systems.

The focus is not just on model performance, but also on understanding internal behavior, robustness, and architectural trade-offs.

---

## Repository Structure

### Adversarial Robustness
- `CNN_training_fgsm_proof.ipynb`  
- `FGSM_proof_CNN_training.ipynb`  
  Experiments demonstrating Fast Gradient Sign Method (FGSM) attacks and their impact on CNN training dynamics, along with robustness evaluation.

---

### Generative Models
- `DCGAN_celebfaces.ipynb`  
  Implementation of Deep Convolutional GAN trained on CelebFaces dataset for high-quality face generation.

---

### EEG & Attention Mechanisms
- `EEG_band_attention_block.py`  
  Custom attention block tailored for EEG signal processing, focusing on band-specific feature learning.

---

### Neuroevolution / Optimization
- `Neural_Genetic_Algorithm.ipynb`  
  Exploration of genetic algorithms for optimizing neural network parameters and architectures.

---

### Deepfake Detection
- `RESNET_DENSENET_deepfake_detection.ipynb`  
- `VIT_Deepfake_detection.ipynb`  
- `Stress_test_deepfake_model.py`  
  Comparative study of CNN-based (ResNet, DenseNet) and Transformer-based (ViT) architectures for deepfake detection, including robustness and stress testing.

---

### Systems & Scheduling
- `Task_Scheduling_Hierarchical.py`  
  Hierarchical scheduling strategies, potentially applicable to distributed ML workloads or resource-aware training pipelines.

---

## Key Themes

- Adversarial machine learning and robustness
- Vision models (CNNs, Transformers)
- Generative modeling (GANs)
- Bio-signal processing (EEG)
- Optimization via evolutionary strategies
- Systems-level experimentation and performance considerations

---

## Design Philosophy

- **Experiment-first approach**: rapid prototyping and validation
- **System awareness**: focus on computational and architectural efficiency
- **Modularity**: reusable components where applicable
- **Exploration-driven**: includes both research-grade and exploratory work

---

## Requirements

Typical dependencies across notebooks and scripts:

- Python 3.8+
- PyTorch / TensorFlow (varies per notebook)
- NumPy, Pandas
- Matplotlib / Seaborn
- OpenCV (for vision tasks)

Install per project as needed.

---

## Usage

Each notebook/script is self-contained.  
Recommended workflow:

1. Open the relevant notebook
2. Install dependencies if missing
3. Execute sequentially
4. Modify hyperparameters or architecture for experimentation

---

## Notes

- This repository is not structured as a production package.
- Code quality and structure may vary across experiments.
- Some scripts are exploratory and may require cleanup or refactoring.

---

## Future Directions

- Standardizing experiment pipelines
- Adding benchmarking and evaluation suites
- Integrating experiment tracking (e.g., WandB)
- Expanding adversarial defense techniques
- Scaling models and datasets

---
