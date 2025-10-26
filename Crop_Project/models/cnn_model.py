import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 224/2/2/2 = 28
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class CNNClassifier:
    def __init__(self, model_path=None):
        self.model = SimpleCNN(num_classes=3)
        self.class_names = ['Healthy Crop', 'Moderate Crop', 'Soil/Non-Vegetated']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with random weights for demo
            self._initialize_demo_weights()
    
    def _initialize_demo_weights(self):
        """Initialize model with demo weights (simulating a trained model)"""
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}. Using demo weights.")
            self._initialize_demo_weights()
    
    def preprocess_image(self, image_array):
        """Preprocess image for CNN inference"""
        try:
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image_array).float()
            
            # Rearrange dimensions to (C, H, W)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Normalize with ImageNet stats (common practice)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            return image_tensor
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image for CNN: {str(e)}")
    
    def predict(self, image_array):
        """Run CNN inference on image"""
        try:
            self.model.eval()
            
            # Preprocess image
            input_tensor = self.preprocess_image(image_array)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get results
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            all_probs = probabilities.squeeze().tolist()
            class_probabilities = {
                self.class_names[i]: round(prob, 4) 
                for i, prob in enumerate(all_probs)
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': round(confidence_score, 4),
                'all_probabilities': class_probabilities,
                'processing_time': 0.15  # Simulated processing time
            }
            
        except Exception as e:
            raise ValueError(f"Error during CNN prediction: {str(e)}")
    
    def mock_training(self, epochs=3):
        """Mock training function for demonstration"""
        training_history = {
            'epochs': [],
            'accuracy': [],
            'loss': []
        }
        
        # Simulate training progress
        for epoch in range(epochs):
            # Generate mock metrics
            accuracy = 0.6 + (epoch * 0.1)  # Increasing accuracy
            loss = 1.0 - (epoch * 0.2)      # Decreasing loss
            
            training_history['epochs'].append(epoch + 1)
            training_history['accuracy'].append(round(accuracy, 3))
            training_history['loss'].append(round(loss, 3))
        
        return training_history