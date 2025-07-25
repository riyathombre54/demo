# ==================== Prediction Utilities ====================
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import ipywidgets as widgets
from IPython.display import display
import tensorflow as tf
from tensorflow.keras.models import load_model

class CervixPredictor:
    def __init__(self, model_paths=None, ensemble_models=None):
        """
        Initialize predictor with either model paths or loaded models
        """
        self.models = []
        self.model_names = []
        
        if model_paths:
            for name, path in model_paths.items():
                model = load_model(path)
                self.models.append(model)
                self.model_names.append(name)
        elif ensemble_models:
            for name, model, _ in ensemble_models:
                self.models.append(model)
                self.model_names.append(name)
    
    def detect_and_crop_cervix(self, image):
        """Detect and crop cervix region"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for pink/red regions (cervix tissue)
        lower_pink = np.array([0, 50, 50])
        upper_pink = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
        
        lower_pink2 = np.array([160, 50, 50])
        upper_pink2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if the detected region is large enough to be a cervix
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Crop the image
                if w > 50 and h > 50:
                    cropped_image = image[y:y+h, x:x+w]
                    return cropped_image, True, mask
        
        return image, False, mask
    
    def preprocess_image(self, image, target_size=(512, 512)):
        """Preprocess image for prediction"""
        # Detect and crop cervix
        processed_image, is_cervix_detected, mask = self.detect_and_crop_cervix(image)
        
        # Resize
        processed_image = cv2.resize(processed_image, target_size)
        
        # Normalize
        processed_image = processed_image.astype(np.float32) / 255.0
        
        return processed_image, is_cervix_detected, mask
    
    def predict_single(self, image):
        """Make prediction on a single image"""
        processed_image, is_cervix_detected, mask = self.preprocess_image(image)
        
        # If no cervix detected, return warning
        if not is_cervix_detected:
            return {
                'warning': 'No cervix region detected. This may not be a valid cervical image.',
                'cervix_detected': False,
                'predictions': {},
                'ensemble_prediction': 0.5,
                'confidence': 0.0
            }
        
        # Expand dimensions for batch processing
        img_batch = np.expand_dims(processed_image, axis=0)
        
        # Get predictions from all models
        predictions = {}
        all_preds = []
        
        for i, model in enumerate(self.models):
            pred = model.predict(img_batch, verbose=0)[0][0]
            predictions[self.model_names[i]] = pred
            all_preds.append(pred)
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(all_preds)
        
        # Calculate confidence
        confidence = abs(ensemble_pred - 0.5) * 2  # Convert to 0-1 scale
        
        return {
            'cervix_detected': True,
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence': confidence,
            'processed_image': processed_image,
            'detection_mask': mask
        }
    
    def visualize_prediction(self, image, prediction_result):
        """Visualize prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Detection mask
        if 'detection_mask' in prediction_result:
            axes[0, 1].imshow(prediction_result['detection_mask'], cmap='gray')
            axes[0, 1].set_title('Cervix Detection Mask')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'No cervix detected', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Detection Failed')
            axes[0, 1].axis('off')
        
        # Processed image
        if 'processed_image' in prediction_result:
            axes[1, 0].imshow(prediction_result['processed_image'])
            axes[1, 0].set_title('Processed Image (for Prediction)')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].axis('off')
        
        # Prediction results
        axes[1, 1].axis('off')
        
        if prediction_result['cervix_detected']:
            ensemble_pred = prediction_result['ensemble_prediction']
            confidence = prediction_result['confidence']
            
            class_name = 'Abnormal' if ensemble_pred > 0.5 else 'Normal'
            class_prob = ensemble_pred if ensemble_pred > 0.5 else 1 - ensemble_pred
            
            result_text = f"ENSEMBLE PREDICTION\n\n"
            result_text += f"Class: {class_name}\n"
            result_text += f"Probability: {class_prob:.3f}\n"
            result_text += f"Confidence: {confidence:.3f}\n\n"
            
            result_text += "Individual Model Predictions:\n"
            for model_name, pred in prediction_result['predictions'].items():
                model_class = 'Abnormal' if pred > 0.5 else 'Normal'
                result_text += f"{model_name}: {model_class} ({pred:.3f})\n"
            
            # Color based on prediction
            color = 'red' if ensemble_pred > 0.5 else 'green'
            
        else:
            result_text = prediction_result.get('warning', 'Prediction failed')
            color = 'orange'
        
        axes[1, 1].text(0.05, 0.95, result_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.show()

def create_prediction_widget(predictor):
    """Create interactive widget for image upload and prediction"""
    
    def on_upload_change(change):
        """Handle file upload"""
        for name, file_info in upload_widget.value.items():
            print(f"\n📂 Analyzing: {name}")
            
            # Load image
            image = Image.open(io.BytesIO(file_info['content'])).convert('RGB')
            image_array = np.array(image)
            
            # Make prediction
            result = predictor.predict_single(image_array)
            
            # Visualize results
            predictor.visualize_prediction(image_array, result)
            
            # Print summary
            if result['cervix_detected']:
                ensemble_pred = result['ensemble_prediction']
                class_name = 'Abnormal' if ensemble_pred > 0.5 else 'Normal'
                confidence = result['confidence']
                print(f"🎯 Final Prediction: {class_name} (Confidence: {confidence:.3f})")
            else:
                print("⚠️  Warning: No cervix region detected in the image")
    
    # Create upload widget
    upload_widget = widgets.FileUpload(
        accept='image/*',
        multiple=True,
        description='Upload Images'
    )
    
    # Bind the event
    upload_widget.observe(on_upload_change, names='value')
    
    # Display instructions and widget
    print("🔮 Interactive Cervical Image Analyzer")
    print("=" * 50)
    print("Instructions:")
    print("1. Click 'Upload Images' button below")
    print("2. Select one or more cervical images")
    print("3. View automatic analysis results")
    print("4. The system will:")
    print("   - Detect cervix regions automatically")
    print("   - Classify as Normal or Abnormal")
    print("   - Show confidence scores")
    print("   - Display individual model predictions")
    print("\n⚠️  Note: Only upload actual cervical/colposcopic images for accurate results")
    print("=" * 50)
    
    display(upload_widget)
    
    return upload_widget

def predict_from_folder(predictor, folder_path, show_all=True):
    """Predict all images in a folder"""
    import os
    
    print(f"🔍 Analyzing all images in: {folder_path}")
    
    results = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            img_path = os.path.join(folder_path, filename)
            
            # Load image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            result = predictor.predict_single(image)
            result['filename'] = filename
            results.append(result)
            
            print(f"\n📂 Image: {filename}")
            
            if result['cervix_detected']:
                ensemble_pred = result['ensemble_prediction']
                class_name = 'Abnormal' if ensemble_pred > 0.5 else 'Normal'
                confidence = result['confidence']
                print(f"   Prediction: {class_name} (Confidence: {confidence:.3f})")
                
                if show_all:
                    predictor.visualize_prediction(image, result)
            else:
                print("   ⚠️  No cervix detected - may not be a valid cervical image")
    
    # Summary statistics
    print(f"\n📊 SUMMARY")
    print("=" * 40)
    
    valid_predictions = [r for r in results if r['cervix_detected']]
    invalid_images = [r for r in results if not r['cervix_detected']]
    
    print(f"Total images processed: {len(results)}")
    print(f"Valid cervical images: {len(valid_predictions)}")
    print(f"Invalid/unclear images: {len(invalid_images)}")
    
    if valid_predictions:
        normal_count = sum(1 for r in valid_predictions if r['ensemble_prediction'] < 0.5)
        abnormal_count = len(valid_predictions) - normal_count
        
        print(f"\nClassification Results:")
        print(f"Normal: {normal_count} ({normal_count/len(valid_predictions)*100:.1f}%)")
        print(f"Abnormal: {abnormal_count} ({abnormal_count/len(valid_predictions)*100:.1f}%)")
        
        avg_confidence = np.mean([r['confidence'] for r in valid_predictions])
        print(f"Average confidence: {avg_confidence:.3f}")
    
    return results

# ==================== Usage Example ====================
if __name__ == "__main__":
    # Example usage after training models
    print("Prediction utilities loaded!")
    print("To use these functions:")
    print("1. First train your models using the main script")
    print("2. Create a predictor: predictor = CervixPredictor(ensemble_models=trained_models)")
    print("3. Use interactive widget: create_prediction_widget(predictor)")
    print("4. Or predict from folder: predict_from_folder(predictor, '/path/to/images')")