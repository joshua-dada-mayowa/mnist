# MNIST Digit Generator with Conditional VAE

A Streamlit web application that generates handwritten digits using a Conditional Variational Autoencoder (CVAE) trained on the MNIST dataset.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional but Recommended)
```bash
python train.py --epochs 20
```
This will train the model for 20 epochs and save the weights to `models/cvae_mnist_weights.h5`.

### 3. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
├── .devcontainer/
│   └── devcontainer.json          # Dev container configuration
├── model.py                       # Conditional VAE model definition
├── train.py                       # Training script
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Model Architecture

The Conditional VAE consists of:
- **Encoder**: Compresses 28x28 images and digit labels into a 50-dimensional latent space
- **Decoder**: Reconstructs images from latent vectors and labels
- **Conditional Input**: Uses digit labels (0-9) to control generation

## Features

- **Interactive Web Interface**: Easy-to-use Streamlit interface
- **Digit Selection**: Choose which digit (0-9) to generate
- **Batch Generation**: Generate 5 unique variations at once
- **Real-time Generation**: Fast inference with trained model
- **Model Status**: Shows whether using trained or untrained model

## Training Options

```bash
# Quick training (10 epochs)
python train.py --epochs 10

# Full training with custom parameters
python train.py --epochs 50 --batch_size 256 --lr 0.0001

# Save to custom location
python train.py --model_path "my_models/cvae_weights.h5"
```

## Training Details

- **Dataset**: MNIST (60,000 training images)
- **Loss Function**: Reconstruction Loss + KL Divergence
- **Optimizer**: Adam
- **Default Learning Rate**: 1e-4
- **Default Batch Size**: 128

##  Development Container

This project includes a VS Code dev container configuration for easy setup:

1. Open the project in VS Code
2. When prompted, click "Reopen in Container"
3. The container will automatically install dependencies and start the Streamlit app

## Troubleshooting

### Model Not Generating Good Images
- **Solution**: Train the model first using `python train.py`
- The untrained model will generate random noise

### Import Errors
- **Solution**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- Check that you're using Python 3.8+ with TensorFlow 2.13+

### Memory Issues During Training
- **Solution**: Reduce batch size: `python train.py --batch_size 64`

## Expected Results

After training:
- **Epoch 1-10**: Basic digit shapes emerge
- **Epoch 10-30**: Clear, recognizable digits
- **Epoch 30+**: High-quality, varied digit generations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both trained and untrained models
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.