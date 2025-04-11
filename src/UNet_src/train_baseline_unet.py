import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import filters, measure
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse  # Added for command line arguments

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 8
EPOCHS = 50

# Cell types to process
CELL_TYPES = [
    'eosinophil_processed',
    'basophil_processed',
    'erythroblast_processed',
    'immunoglobulin_processed',
    'lymphocyte_processed',
    'monocyte_processed',
    'neutrophil_processed',
    'platelet_processed'
]

# Number of images to use per cell type
NUM_IMAGES = {
    'eosinophil_processed': 30,
    'basophil_processed': 10,
    'erythroblast_processed': 10,
    'immunoglobulin_processed': 10,
    'lymphocyte_processed': 10,
    'monocyte_processed': 10,
    'neutrophil_processed': 10,
    'platelet_processed': 10
}

def load_images(base_path="dataset_2/processed/"):
    """Load images from the specified folders."""
    images = []
    masks = []
    
    for cell_type in CELL_TYPES:
        cell_path = os.path.join(base_path, cell_type)
        
        # Check if directory exists
        if not os.path.exists(cell_path):
            print(f"Warning: Directory not found: {cell_path}")
            continue
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(cell_path) if f.endswith('.jpg')]
        image_files.sort()  # Sort to ensure we get images in order
        
        # Limit to the required number of images
        limit = NUM_IMAGES[cell_type]
        image_files = image_files[:limit]
        
        for img_file in image_files:
            img_path = os.path.join(cell_path, img_file)
            
            # Load and preprocess the image
            img = imread(img_path)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True)
            
            # Check if image is grayscale (2D) and convert to 3D if needed
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)  # Convert to RGB by duplicating the channel
            elif len(img.shape) == 3 and img.shape[2] == 4:  # Handle RGBA images
                img = img[:, :, :3]  # Take only the RGB channels
                
            img = img / 255.0  # Normalize to [0,1]
            
            # Create a simple mask by thresholding
            # Convert to grayscale first if it's RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = rgb2gray(img)
            else:
                gray = img[:,:,0]  # Just take the first channel if already grayscale-like
                
            threshold = filters.threshold_otsu(gray)
            binary = gray > threshold
            
            # Clean up the mask
            mask = measure.label(binary)
            mask = (mask > 0).astype(np.float32)
            
            images.append(img)
            masks.append(np.expand_dims(mask, axis=-1))  # Add channel dimension
    
    return np.array(images), np.array(masks)

def build_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """Build a U-Net model architecture."""
    inputs = Input(input_size)
    
    # Encoder (Contracting Path)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Decoder (Expansive Path)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1):
    """Calculate Dice coefficient for evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def train_model(model, X_train, y_train, X_val, y_val, output_model_path='baseline_unet.h5', batch_size=8, epochs=50):
    """Train the U-Net model."""
    # Define callbacks
    checkpoint = ModelCheckpoint(output_model_path, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True)
    
    early_stopping = EarlyStopping(patience=10, 
                                  verbose=1, 
                                  monitor='val_loss',
                                  restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(factor=0.1, 
                                  patience=5, 
                                  min_lr=1e-6, 
                                  verbose=1,
                                  monitor='val_loss')
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, output_dir='.'):
    """Evaluate the trained model."""
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    loss, accuracy, iou = model.evaluate(X_test, y_test)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test IoU: {iou:.4f}")
    
    # Visualize some results
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    
    for i in range(4):
        idx = np.random.randint(0, len(X_test))
        
        # Original image
        axes[i, 0].imshow(X_test[idx])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(y_test[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(predictions[idx].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_unet_results.png'))
    plt.close()
    
    return predictions

def count_cells(predictions, threshold=0.5):
    """Count the number of cells in each image based on predicted masks."""
    cell_counts = []
    
    for pred in predictions:
        # Threshold the prediction
        binary = (pred.squeeze() > threshold).astype(np.uint8)
        
        # Label connected components
        labeled = measure.label(binary)
        
        # Count regions
        regions = measure.regionprops(labeled)
        
        # Filter regions by size to remove noise
        min_size = 30  # Adjust based on your cell size
        valid_regions = [region for region in regions if region.area >= min_size]
        
        cell_counts.append(len(valid_regions))
    
    return cell_counts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a U-Net model for blood cell segmentation.')
    parser.add_argument('--dataset_path', type=str, default="dataset_2/processed/",
                        help='Path to the processed dataset')
    parser.add_argument('--output_dir', type=str, default=".",
                        help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--model_name', type=str, default="baseline_unet.h5",
                        help='Name of the output model file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set parameters from arguments
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    model_path = os.path.join(output_dir, args.model_name)
    
    # Load and preprocess data
    print(f"Loading and preprocessing images from {dataset_path}...")
    X, y = load_images(dataset_path)
    
    if len(X) == 0:
        print("Error: No images were loaded. Check the dataset path.")
        return
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Build the model
    print("Building U-Net model...")
    model = build_unet_model()
    model.summary()
    
    # Train the model
    print(f"Training the model for {epochs} epochs with batch size {batch_size}...")
    model, history = train_model(model, X_train, y_train, X_val, y_val, 
                                output_model_path=model_path, 
                                batch_size=batch_size, 
                                epochs=epochs)
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Evaluate the model
    print("Evaluating the model...")
    predictions = evaluate_model(model, X_test, y_test, output_dir=output_dir)
    
    # Count cells
    print("Counting cells in test images...")
    cell_counts = count_cells(predictions)
    
    # Display some counts
    for i in range(min(5, len(cell_counts))):
        print(f"Image {i+1}: {cell_counts[i]} cells detected")
    
    # Save results summary
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("U-Net Training Summary\n")
        f.write("=====================\n\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Model saved to: {model_path}\n")
        f.write(f"Training images: {X_train.shape[0]}\n")
        f.write(f"Validation images: {X_val.shape[0]}\n")
        f.write(f"Test images: {X_test.shape[0]}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n\n")
        
        f.write("Test Results:\n")
        f.write(f"  Loss: {model.evaluate(X_test, y_test)[0]:.4f}\n")
        f.write(f"  Accuracy: {model.evaluate(X_test, y_test)[1]:.4f}\n")
        f.write(f"  IoU: {model.evaluate(X_test, y_test)[2]:.4f}\n")
    
    print(f"Processing complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()