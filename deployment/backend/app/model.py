import tensorflow as tf
import json
import os
import numpy as np


class VideoClassifier:
    def __init__(self):
        # Load the Keras model
        model_path = os.path.join(os.path.dirname(__file__), '../model/sign_transformer.keras')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class labels
        labels_path = os.path.join(os.path.dirname(__file__), '../model/labels.json')
        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)
            self.labels = [labels_dict[str(i)] for i in range(len(labels_dict))]

        # Load normalization stats
        norm_stats_path = os.path.join(os.path.dirname(__file__), '../model/norm_stats.json')
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
            self.mean = np.array(norm_stats['mean'])
            self.variance = np.array(norm_stats['variance'])

    def normalize_landmarks(self, landmarks):
        return (landmarks - self.mean) / np.sqrt(self.variance)
    
    def predict(self, landmarks, angles):
        """
        Make prediction on processed video frames
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        norm_landmarks = self.normalize_landmarks(landmarks)
        input_data = np.concatenate([np.squeeze(norm_landmarks), angles], axis=1)
        
        predictions = self.model.predict(np.expand_dims(input_data, axis=0), verbose=0)
        
        # Get the predicted class and confidence
        predicted_class_idx = predictions.argmax()
        confidence = float(predictions.max())
        
        return {
            "label": self.labels[predicted_class_idx],
            "confidence": confidence
        }