# Import the required dependencies
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore


# 1. Load and preprocess the dataset
def load_data(data_dir, img_size=(64, 64)):
    """Loads and preprocesses the dataset."""
    classes = sorted(os.listdir(data_dir))
    X, y = [], []
    
    # Iterate over the class folders
    for label, cls in enumerate(classes):
        cls_folder = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image {img_name}: File may be missing or corrupted.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, img_size)  # Resize to the desired input size
            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y), classes

# 2. Image visualization
def visualize_images(X, y, class_names):
    """Displays a grid of 25 images with their class names."""
    plt.figure(figsize=(10, 10))
    for i in range(25):
        idx = np.random.randint(0, len(X))
        plt.subplot(5, 5, i + 1)
        plt.imshow(X[idx])
        plt.title(class_names[y[idx]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 3. Build the CNN model
def build_model(num_classes):
    """Builds and compiles a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # First CNN block
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),  # Second CNN block
        MaxPooling2D((2, 2)),
        Flatten(),  # Flatten the output for Dense layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# 4. Train the model with data augmentation
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the model with data augmentation."""
    datagen = ImageDataGenerator(
        rotation_range=15,  # Random rotations
        width_shift_range=0.1,  # Random horizontal shift
        height_shift_range=0.1,  # Random vertical shift
    )
    
    datagen.fit(X_train)
    
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_test, y_test),
                        epochs=epochs)
    
    return history

# 5. Plot training history
def plot_history(history):
    """Plots the training and validation accuracy and loss."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')

    plt.show()


# 6. Save the trained model
def save_model(model, filename="traffic_sign_classifier.keras"):
    """Save the model to a file."""
    model.save(filename)
    print(f"Model saved to {filename}")


# 7. Load the model and classify new images
def classify_traffic_sign(image_path, model_path="traffic_sign_classifier.keras", img_size=(64, 64), class_names=None):
    """Classifies a traffic sign based on a user-provided image."""
    model = tf.keras.models.load_model(model_path)
    
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found or cannot be opened.")
    
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img / 255.0, axis=0)  # Normalize and add batch dimension
    
    # Predict and get the class with the highest probability
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    
    return class_names[class_idx]


# 8. Main function to run the entire pipeline
def main():
    data_dir = r"D:/AI_Project/tscv2/TSC\DATA2"  # Path to dataset folder
    img_size = (64, 64)  # Desired image size
    
    # Load and preprocess the dataset
    X, y, class_names = load_data(data_dir, img_size)
    X = X / 255.0  # Normalize the images to [0, 1]

    # Visualize a batch of images
    visualize_images(X, y, class_names)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_model(num_classes=len(class_names))
    
    # Summarize and plot the model
    model.summary()
    plot_model(model, to_file="summary.png", show_shapes=True, show_layer_names=True)

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50)
    
    # Save the trained model
    save_model(model, filename="traffic_sign_classifier.keras")
    print("Model saved successfully.")

    # Plot training history
    plot_history(history)
    
    # Example usage: classify a user-provided image
    user_image_path = r"D:/AI_Project/tscv2/deer.png"  # Replace with an actual image path
    traffic_sign_type = classify_traffic_sign(user_image_path, model_path="traffic_sign_classifier.keras", class_names=class_names)
    print(f"The traffic sign type is: {traffic_sign_type}")


if __name__ == "__main__":
    main()
