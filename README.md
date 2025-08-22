# SimpleBlockNAS 🧬🔍

**SimpleBlockNAS** is a *Neural Architecture Search* (NAS) framework that uses *Genetic Algorithms* (GA) to automatically discover optimal neural network architectures and tune hyperparameters.  
It provides a modular and extensible system for automated deep learning model design.

> ⚠️ **Project Status:** PAUSED/MAINTENANCE.  

---

## 🚀 Features

- **Genetic Algorithm-based NAS** — evolutionary search for neural architectures
- **Modular Block System** — pre-defined building blocks (Linear, Convolution2D, MaxPool2D, Flatten)
- **Flexible Search Settings** — configurable mutation and crossover operators
- **Multi-Dataset Support** — built-in support for EuroSAT and CIFAR10, easily extendable
- **PyTorch / PyTorch Lightning Integration**
- **Customizable Training Parameters** — learning rate, batch size, GA parameters
- **Model Weight Persistence** — save and reuse trained model weights

---

## 🏗 Project Architecture

- **GA Pipeline** — mutation, crossover, and selection implementation
- **Block System** — modular neural network building blocks
- **Population Management** — population handling and evolution
- **Network Validation** — architecture verification and parameter alignment
- **Training Pipeline** — training and evaluation

---

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.4.1+
- Torchvision 0.19.1+
- Torchmetrics 1.4.2+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Example run is available in:
```
experiments.py
```

---

## 🧩 Available Blocks

**Basic:**
- `Linear` — fully connected layers (activation, dropout)
- `Convolution2d` — 2D convolutional layers with batch norm and dropout
- `MaxPool2d` — max pooling
- `Flatten` — tensor flattening

**Configuration Features:**
- Custom activation functions (ReLU, Softmax, None)
- Dropout layers
- Batch normalization
- Parameter randomization
- Custom connections

---

## 📊 Logging & Results

The system automatically saves:
- Evolution progress
- Training metrics and validation scores
- Population statistics
- All evaluated architectures

Reports are stored in the `reports/` folder with timestamps and experiment names.

---

## 🤝 Contributing

If you have questions or suggestions, feel free to open an issue on GitHub or contact me directly.

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---
