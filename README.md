# Neural Network Handwritten Numbers

A deep learning project that implements a neural network for recognizing handwritten digits using the MNIST dataset and PyTorch.

## 🎯 Project Overview

This project demonstrates the implementation of a fully connected neural network capable of recognizing handwritten digits (0-9) with high accuracy. The model is trained on the famous MNIST dataset and achieves excellent performance on digit classification tasks.

## ✨ Features

- **Deep Neural Network**: Multi-layer perceptron with dropout for regularization
- **MNIST Dataset**: Automatic download and preprocessing of the standard MNIST dataset
- **GPU Support**: Automatic detection and utilization of CUDA-enabled GPUs
- **Model Persistence**: Save and load trained models
- **Visualization**: Generate prediction visualizations with confidence scores
- **Real-time Evaluation**: Track training progress with loss and accuracy metrics

## 🏗️ Architecture

The neural network consists of:
- **Input Layer**: 784 neurons (28×28 flattened pixels)
- **Hidden Layer 1**: 512 neurons with ReLU activation and 20% dropout
- **Hidden Layer 2**: 256 neurons with ReLU activation and 20% dropout
- **Hidden Layer 3**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9)

## 📋 Requirements

```
torch
torchvision
matplotlib
numpy
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Kalharapasan/Neural-Network-Handwritten-Numbers.git
cd Neural-Network-Handwritten-Numbers
```

2. Install required packages:
```bash
pip install torch torchvision matplotlib numpy
```

### Usage

1. **Open the Jupyter Notebook**:
```bash
jupyter notebook Neural_Network_Handwritten_Numbers.ipynb
```

2. **Run all cells** to:
   - Load and preprocess the MNIST dataset
   - Create the neural network model
   - Train the model (5 epochs by default)
   - Evaluate performance on test data
   - Generate prediction visualizations
   - Save the trained model

### Key Functions

- `load_data()`: Downloads and preprocesses MNIST dataset
- `train()`: Trains the neural network with specified parameters
- `evaluate()`: Tests the model on validation data
- `predict_digit()`: Makes predictions on individual images
- `visualize_predictions()`: Creates visual comparison of predictions vs actual labels
- `save_model()` / `load_model()`: Model persistence functions

## 📊 Performance

The model typically achieves:
- **Training Accuracy**: ~98%+
- **Test Accuracy**: ~97%+
- **Training Time**: ~2-3 minutes on GPU, ~5-10 minutes on CPU

## 📁 Project Structure

```
Neural-Network-Handwritten-Numbers/
├── Neural_Network_Handwritten_Numbers.ipynb  # Main implementation
├── digit_recognizer.pth                      # Saved model weights
├── README.md                                 # Project documentation
└── data/                                     # MNIST dataset (auto-downloaded)
    └── MNIST/
        └── raw/
            ├── train-images-idx3-ubyte
            ├── train-labels-idx1-ubyte
            ├── t10k-images-idx3-ubyte
            └── t10k-labels-idx1-ubyte
```

## 🔬 Technical Details

- **Framework**: PyTorch
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 64
- **Epochs**: 5 (configurable)
- **Normalization**: Mean=0.1307, Std=0.3081 (MNIST standard)

## 🎨 Visualization

The project generates prediction visualizations showing:
- Original handwritten digits
- Model predictions with confidence percentages
- Color-coded results (green for correct, red for incorrect)
- Side-by-side comparison with true labels

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for:
- Performance improvements
- Additional visualization features
- Code optimizations
- Documentation enhancements

## 📄 License

This project is open source and available under the [License](LICENSE.md).

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and support

---

**Built with ❤️ using PyTorch**