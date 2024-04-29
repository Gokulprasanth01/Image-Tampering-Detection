# Image Tampering Detection using Convolutional Neural Networks

This project focuses on detecting image tampering using Convolutional Neural Networks (CNNs). It provides a Python implementation to train a CNN model to classify images as real or tampered.

## Overview

Image tampering detection is a critical task in various domains including forensics and media analysis. This project aims to develop a robust solution leveraging deep learning techniques to identify manipulated images.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- OpenCV

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/image-tampering-detection.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the CASIA2 dataset and place it in the appropriate directory.

### Usage

1. Run the Jupyter notebook `Image_Tampering_Detection.ipynb` to train the model and evaluate its performance.

2. Use the trained model to detect image tampering by loading the `Model.h5` file.

## Dataset

The CASIA2 dataset is used for training and testing the model. It consists of authentic and tampered images.

## Model Architecture

The model architecture consists of convolutional layers followed by max-pooling, batch normalization, dropout, and dense layers. It is designed to extract features from images and classify them into real or tampered.

## Results

The model achieves an accuracy of [90.07] on the validation set after [15] epochs of training.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or support, please contact [gokulprasanth0104@gmail.com].

