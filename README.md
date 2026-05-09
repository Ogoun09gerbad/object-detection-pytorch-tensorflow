# Intel Image Scene Classification

## Project Overview

This project implements an end-to-end **image classification pipeline** for the Intel Image Classification dataset. The goal is to classify natural scenes into six categories:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

We are designed, trained and deployed :

* **PyTorch CNN** to `geraud_model.pth`
* **TensorFlow/Keras CNN** to `geraud_model.keras`

The system is deployed as a web application using **Flask** and hosted on **Hugging Face Spaces**.

## Training

### PyTorch

```bash
python main.py --framework pytorch --epochs 20 --cuda
```

### TensorFlow / Keras

```bash
python main.py --framework tensorflow --epochs 20 --cuda
```

### Optional Arguments

* `--lr` : learning rate (default: 1e-3)
* `--wd` : weight decay (default: 1e-4)
* `--epochs` : number of epochs
* `--cuda` : enable GPU

---

## Run the Web App Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

---

## Project Structure

```
├── app.py                  # Flask web application
├── main.py                 # Training script (PyTorch + TensorFlow)
├── Dockerfile              # Docker configuration
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── models/
│   ├── cnn.py              # Model architectures
│   └── train.py            # Training loop (PyTorch)
│
├── utils/
│   └── prep.py             # Data loading & augmentation
│
├── templates/
│   └── index.html          # Web interface
│
├── geraud_model.pth        # Trained PyTorch model
└── geraud_model.keras      # Trained TensorFlow model
```

---


---

## Methodology

### Data Processing

* Image resizing to **150×150**
* Normalization using ImageNet statistics
* Data augmentation:

  * Random cropping
  * Horizontal flipping
  * Rotation
  * Color jitter (brightness, contrast, saturation)

---

### Model Architectures

#### PyTorch CNN

* 4 convolutional blocks:

  * Conv --> Conv --> BatchNorm --> ReLU --> MaxPool --> Dropout
* Increasing channels: **32 --> 64 --> 128 --> 256**
* Adaptive Average Pooling
* Fully connected layers: **256 --> 128 --> 6**
* Dropout regularization

---

#### TensorFlow / Keras CNN

* Similar convolutional structure with:

  * Batch Normalization
  * Dropout (up to 0.6)
* Global Average Pooling
* Dense layers: **256 --> 6 (Softmax)**

---

###  Training Strategy

#### PyTorch

* Optimizer: **Adam**
* Scheduler: **CosineAnnealingLR**
* Loss: **CrossEntropyLoss**
* Early stopping (patience = 10)

#### TensorFlow

* Optimizer: **Adam**
* Loss: **Sparse Categorical Crossentropy**
* Callbacks:

  * ModelCheckpoint
  * ReduceLROnPlateau
  * EarlyStopping

---

## Results

| Model                | Test Accuracy | Test Loss |
| -------------------- | ------------- | --------- |
| PyTorch CNN          | **79.02%**    | 0.5726    |
| TensorFlow/Keras CNN | **85.63%**    | 0.4070    |

---

##  Key Insights

* The **TensorFlow model outperformed PyTorch** due to:

  * Adaptive learning rate scheduling (ReduceLROnPlateau)
  * Stronger regularization (higher dropout)
  * Global average pooling improving generalization

* The PyTorch model shows **stable training** but can be improved with:

  * Better learning rate scheduling
  * Mixed precision training
  * Transfer learning (ResNet / EfficientNet)

---

## Deployment

The application is deployed on **Hugging Face Spaces** using Docker:

 https://huggingface.co/spaces/OGB2000/intel_classification_image_last

---
