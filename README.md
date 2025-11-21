# Neural Network Handwritten Numbers

A deep learning project that implements a neural network for recognizing handwritten digits using the MNIST dataset and PyTorch, with interactive drawing capabilities for real-time digit recognition.

## 🎯 Project Overview

This project demonstrates the implementation of a fully connected neural network capable of recognizing handwritten digits (0-9) with high accuracy. The model is trained on the famous MNIST dataset and features an **interactive drawing interface** that allows users to draw digits on screen and get real-time predictions with confidence scores.

## ✨ Features

- **Interactive Drawing Interface**: Draw digits on screen using mouse/touch for real-time recognition
- **Real-time Predictions**: Instant digit recognition with confidence scores as you draw
- **Deep Neural Network**: Multi-layer perceptron with dropout for regularization
- **MNIST Dataset**: Automatic download and preprocessing of the standard MNIST dataset
- **GPU Support**: Automatic detection and utilization of CUDA-enabled GPUs
- **Model Persistence**: Save and load trained models
- **Visualization**: Generate prediction visualizations with confidence scores
- **Real-time Evaluation**: Track training progress with loss and accuracy metrics
- **Clear Canvas**: Reset drawing area for new digit input

## 🏗️ Architecture

The neural network consists of:
- **Input Layer**: 784 neurons (28×28 flattened pixels)
- **Hidden Layer 1**: 512 neurons with ReLU activation and 20% dropout
- **Hidden Layer 2**: 256 neurons with ReLU activation and 20% dropout
- **Hidden Layer 3**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9)

## 📋 Requirements

```text
torch
torchvision
matplotlib
numpy
tkinter (for drawing interface)
Pillow (for image processing)
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
pip install torch torchvision matplotlib numpy pillow
```

### Usage

#### Option 1: Interactive Drawing Interface

1. **Run the interactive drawing application**:

```bash
python draw_and_predict.py
```

2. **Draw digits on the canvas**:
   - Use your mouse to draw digits (0-9) on the drawing canvas
   - The model will predict the digit in real-time
   - View confidence scores for each prediction
   - Click "Clear" to reset the canvas and draw a new digit
   - Click "Predict" to get the final prediction result

#### Option 2: Train New Model (Jupyter Notebook)

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

**Core Model Functions:**
- `load_data()`: Downloads and preprocesses MNIST dataset
- `train()`: Trains the neural network with specified parameters
- `evaluate()`: Tests the model on validation data
- `predict_digit()`: Makes predictions on individual images
- `visualize_predictions()`: Creates visual comparison of predictions vs actual labels
- `save_model()` / `load_model()`: Model persistence functions

**Interactive Drawing Functions:**
- `create_drawing_interface()`: Sets up the drawing canvas GUI
- `draw_digit()`: Handles mouse drawing events on canvas
- `preprocess_canvas()`: Converts canvas drawing to model input format
- `real_time_predict()`: Provides instant predictions while drawing
- `clear_canvas()`: Resets the drawing area

## 📊 Performance

The model typically achieves:
- **Training Accuracy**: ~98%+
- **Test Accuracy**: ~97%+
- **Training Time**: ~2-3 minutes on GPU, ~5-10 minutes on CPU

## 📁 Project Structure

```text
Neural-Network-Handwritten-Numbers/
├── Neural_Network_Handwritten_Numbers.ipynb  # Main training notebook
├── draw_and_predict.py                       # Interactive drawing interface
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

**Model Architecture:**
- **Framework**: PyTorch
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 64
- **Epochs**: 5 (configurable)
- **Normalization**: Mean=0.1307, Std=0.3081 (MNIST standard)

**Interactive Interface:**
- **GUI Framework**: Tkinter (built-in with Python)
- **Canvas Size**: 280x280 pixels (scaled to 28x28 for model)
- **Real-time Processing**: Instant prediction updates while drawing
- **Image Processing**: PIL for canvas-to-tensor conversion
- **Preprocessing**: Gaussian blur and normalization for better accuracy

## 🎨 Visualization

The project provides multiple visualization modes:

**Static Analysis:**
- Original handwritten digits from test dataset
- Model predictions with confidence percentages
- Color-coded results (green for correct, red for incorrect)
- Side-by-side comparison with true labels

**Interactive Drawing:**
- Real-time canvas for drawing digits
- Live prediction updates as you draw
- Confidence score display
- Visual feedback for drawing quality

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for:

- Performance improvements
- Additional visualization features
- Enhanced drawing interface features
- Mobile/touch device support
- Code optimizations
- Documentation enhancements

## 📄 License

This project is open source and available under the [License](LICENSE.md).

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and support

---

## 🔧 Built With

- **PyTorch** - Deep learning framework
- **Tkinter** - GUI framework for drawing interface
- **PIL/Pillow** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization