Polyp Segmentation Using UNet 3+ in TensorFlow
==============================================

This project implements a deep learning model for segmenting polyps in medical images using the UNet 3+ architecture in TensorFlow. The dataset used for this project is the Kvasir-SEG dataset.

Features
--------

-   **UNet 3+ Architecture**: Efficient and advanced segmentation model.

-   **Custom Loss and Metrics**: Dice loss and Dice coefficient for evaluating segmentation quality.

-   **Dataset Handling**: Automated loading, preprocessing, and splitting into training, validation, and testing sets.

-   **Callbacks**: Integrated ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and CSVLogger for robust training.

Installation
------------

1.  Clone the repository:

    ```
    git clone <repository-url>
    cd Polyp-Segmentation-using-UNet-3-Plus-in-TensorFlow-main
    ```

2.  Install dependencies:

    ```

3.   Ensure you have TensorFlow installed with macOS Metal support if using an M1/M2 Mac:

    
    pip install tensorflow-macos tensorflow-metal


Dataset
-------

The Kvasir-SEG dataset is used in this project. The dataset should be organized as follows:

```
Kvasir-SEG/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── masks/
│   ├── mask1.jpg
│   ├── mask2.jpg
│   └── ...
```

Download the dataset and place it in the project directory.

Usage
-----

### Training the Model

1.  Adjust the hyperparameters and paths in `train.py` as needed.

2.  Run the training script:

    ```
    python train.py
    ```

3.  The trained model and logs will be saved in the `files/` directory.

### Evaluating the Model

1.  Use the `test.py` script to evaluate the model on the test set.

2.  Run the evaluation:

    ```
    python test.py
    ```

Model Summary
-------------

Below is a snippet of the UNet 3+ model summary:

```
Layer (type)                  Output Shape              Param #        Connected to
------------------------------- --------------------------- ---------------- ----------------------------
input_layer (InputLayer)      (None, 256, 256, 3)       0              -
conv1_pad (ZeroPadding2D)     (None, 262, 262, 3)       0              input_layer[0][0]
conv1_conv (Conv2D)           (None, 128, 128, 64)      9,472          conv1_pad[0][0]
...
Total params: 16,463,809 (62.80 MB)
Trainable params: 16,428,097 (62.67 MB)
Non-trainable params: 35,712 (139.50 KB)
```

For the full model summary, refer to the `model_summary.txt` file in the repository.

Results
-------

-   **Training Set**: 800 images and masks

-   **Validation Set**: 100 images and masks

-   **Test Set**: 100 images and masks

Hyperparameters
---------------

-   **Image Size**: 256x256

-   **Batch Size**: 2

-   **Learning Rate**: 0.0001

-   **Epochs**: 500

Callbacks
---------

-   **ModelCheckpoint**: Saves the best model based on validation loss.

-   **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus.

-   **EarlyStopping**: Stops training if validation loss does not improve.

-   **CSVLogger**: Logs training metrics to `log.csv`.

Requirements
------------

-   Python 3.8+

-   TensorFlow 2.13+

-   NumPy

-   OpenCV

-   scikit-learn

Acknowledgements
----------------

-   **Dataset**: Kvasir-SEG Dataset

-   **Framework**: TensorFlow

-   **Architecture**: UNet 3+
