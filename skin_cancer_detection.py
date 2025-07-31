import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gradcam import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import cv2

class SkinCancerDetector:
    def __init__(self, img_size=(224, 224), num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        # Load VGG16 with pre-trained weights
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_size[0], self.img_size[1], 3))
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        return self.model
    
    def train(self, train_dir, val_dir, epochs=50, batch_size=32):
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[early_stopping, checkpoint]
        )
        
        return history
    
    def generate_grad_cam(self, img_path, layer_name='block5_conv3'):
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0
        
        # Get prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        
        # Create Grad-CAM
        grad_cam = GradCAM(self.model, layer_name)
        heatmap = grad_cam.compute_heatmap(img_array, predicted_class)
        
        # Superimpose heatmap on original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img, predictions
    
    def plot_results(self, img_path, layer_name='block5_conv3'):
        superimposed_img, predictions = self.generate_grad_cam(img_path, layer_name)
        
        # Plot original image and Grad-CAM visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.imread(img_path)[...,::-1])  # Convert BGR to RGB
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img[...,::-1])  # Convert BGR to RGB
        plt.title(f'Grad-CAM: {predictions[0][1]*100:.2f}% Cancer')
        plt.axis('off')
        
        plt.show()

if __name__ == "__main__":
    # Initialize detector
    detector = SkinCancerDetector()
    
    # Build model
    model = detector.build_model()
    
    # Train model (uncomment and provide paths to your dataset)
    # history = detector.train('path/to/train', 'path/to/validation')
    
    # Test model (uncomment and provide path to test image)
    # detector.plot_results('path/to/test/image.jpg')
