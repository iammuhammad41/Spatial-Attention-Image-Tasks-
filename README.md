# Spatial Attention Image Task: Muffin vs. Chihuahua Classification

## Overview

This repository contains a deep learning solution for classifying images of muffins and chihuahuas using **PyTorch Lightning**. The model leverages **Convolutional Neural Networks (CNNs)** along with **Spatial Attention** to enhance the focus on important regions of the image for better classification performance. The task aims to differentiate between two classes: "Muffin" and "Chihuahua" in images.

The model is trained on the **Muffin vs Chihuahua Image Classification** dataset, which contains images of muffins and chihuahuas. Spatial attention mechanisms are used to highlight important parts of the image, improving model performance.

## Model Architecture

The model architecture consists of the following layers:

1. **Convolutional Layers**: Used for feature extraction from images.
2. **Spatial Attention**: A mechanism that helps the model to focus on important regions in the image. This improves the accuracy of the model by attending to regions of the image that are relevant for classification.

   * Max pooling and average pooling are applied to extract features from different regions.
   * The outputs of these operations are concatenated and passed through a convolution layer to generate attention maps.
3. **Fully Connected Layers**: These layers are used for classification after the convolutional layers and attention mechanism.

### Spatial Attention Mechanism

* **Maxpooling** and **Average Pooling** are applied to the feature maps.
* The pooled features are concatenated and passed through a convolutional layer to generate the spatial attention map.
* The attention map is applied to the feature map, enhancing the features corresponding to the relevant regions of the image.

## Dataset

The dataset used in this task consists of images labeled as either "Muffin" or "Chihuahua". The images are preprocessed and resized to 256x256 pixels for uniformity.

### Dataset Preprocessing

1. **Resize Images**: Images are resized to 256x256 pixels.
2. **Data Augmentation**: Albumentations are used for data augmentation during training, which includes normalization and other transformations.
3. **Splitting the Data**: The dataset is split into training and validation sets.

## Installation

To use this code, you need to install the following dependencies:

```bash
pip install pytorch-lightning albumentations torchvision wandb
```

## Usage

### 1. **Data Loading and Preprocessing**

The dataset is loaded and preprocessed using the `ImageDataset` and `ImageDataLoader` classes. The images are augmented and resized during the loading process.

```python
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42, stratify=df_train['label'])
train_dm = ImageDataLoader(df_train, 16, resize=(256, 256), augment=train_augmentations)
val_dm = ImageDataLoader(df_val, 256, resize=(256, 256), augment=val_test_augmentations)
```

### 2. **Training the Model**

The model is trained using **PyTorch Lightning**. The training procedure involves:

* Using the **AdamW** optimizer with a **Cosine Annealing Learning Rate Scheduler**.
* The model is trained for a maximum of 100 epochs, with early stopping based on validation loss.

```python
trainer = pl.Trainer(devices=1, accelerator="gpu", precision=16, callbacks=[early_stop], max_epochs=100, gradient_clip_val=2)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
```

### 3. **Evaluating the Model**

After training, the model is evaluated on the test set, and the accuracy is displayed.

```python
test_acc = np.mean(test_preds == df_test['label'].values) * 100
print(f"Test Accuracy : {test_acc:.2f}%")
```

### 4. **Making Predictions**

You can make predictions on custom images using the trained model. Here's how you can predict whether an image is a "Muffin" or a "Chihuahua":

```python
print(custom_img_pred(model2, "path/to/image.jpg"))
```

### 5. **Saving and Loading the Model**

After training, the model can be saved for later use, and you can load it to make predictions on new images.

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

## Example Output

The model achieves an accuracy of approximately **90%** on the test set. The predicted label for each image is displayed as either "Muffin" or "Chihuahua".

```bash
Test Accuracy: 90.2%
```
