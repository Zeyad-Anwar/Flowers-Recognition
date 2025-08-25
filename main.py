import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
import io
import requests
from tensorflow.keras.models import load_model
import os

# Configure TensorFlow to avoid CUDA issues
import os as environ_os
environ_os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Page configuration
st.set_page_config(
    page_title="Flower Recognition App",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("🌸 Flower Recognition System")
st.markdown("Upload an image of a flower and I'll predict what type it is!")

# Flower classes (adjust based on your specific dataset)
FLOWER_CLASSES = [
    'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'
]

@st.cache_resource
def load_flower_model():
    """Load the pre-trained flower recognition model"""
    try:
        # Force CPU usage to avoid CUDA errors
        with tf.device('/CPU:0'):
            # Load your trained .keras model
            model_path = "model.keras" 
            
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                st.info("Please ensure your .keras model file is in the same directory as this script")
                return None
            
            # Configure TensorFlow for CPU-only execution
            tf.config.set_visible_devices([], 'GPU')
            
            model = load_model(model_path, compile=False)  # Skip compilation to avoid GPU issues
            
            # Recompile for CPU if needed
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success(f"Model loaded successfully from {model_path} (CPU mode)")
            return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        
        # Try alternative loading method
        try:
            st.info("Trying alternative loading method...")
            # Set memory growth for GPU (if available)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            st.success("Model loaded with alternative method")
            return model
            
        except Exception as e2:
            st.error(f"Alternative loading also failed: {str(e2)}")
            return None

def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for prediction"""
    try:
        # Open and convert image
        img = Image.open(uploaded_image)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to match model input
        img = img.resize((128, 128))
        
        # Convert to array and preprocess
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def predict_flower(model, processed_image):
    """Make prediction on the processed image"""
    try:
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = FLOWER_CLASSES[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(FLOWER_CLASSES[i], float(predictions[0][i])) for i in top_3_indices]
        
        return predicted_class, confidence, top_3_predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    # Load model
    model = load_flower_model()
    
    if model is None:
        st.error("Failed to load the model. Please check your model file.")
        st.info("Make sure your .keras model file is in the correct location")
        return
    
    # Sidebar
    st.sidebar.markdown("## About")
    st.sidebar.info(
        "This app uses a Convolutional Neural Network to classify flower images. "
        "Upload an image and get instant predictions!"
    )
    
    st.sidebar.markdown("## Supported Flower Types")
    for flower in FLOWER_CLASSES:
        st.sidebar.write(f"• {flower.capitalize()}")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a flower image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a flower for best results"
        )
    
    with col2:
        if uploaded_file is not None:
            # Process and display the image
            processed_img, display_img = preprocess_image(uploaded_file)
            
            if processed_img is not None:
                st.subheader("Uploaded Image")
                st.image(display_img, caption="Uploaded Image", use_column_width=True)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, top_3 = predict_flower(model, processed_img)
                
                if predicted_class is not None:
                    st.subheader("Prediction Results")
                    
                    # Main prediction
                    st.success(f"**Predicted Flower: {predicted_class.upper()}**")
                    st.info(f"**Confidence: {confidence:.2%}**")
                    
                    # Top 3 predictions
                    st.subheader("Top 3 Predictions")
                    for i, (flower, conf) in enumerate(top_3, 1):
                        if i == 1:
                            st.write(f"🥇 **{flower.capitalize()}**: {conf:.2%}")
                        elif i == 2:
                            st.write(f"🥈 **{flower.capitalize()}**: {conf:.2%}")
                        else:
                            st.write(f"🥉 **{flower.capitalize()}**: {conf:.2%}")
                    
                    # Confidence bar chart
                    st.subheader("Confidence Distribution")
                    chart_data = {flower.capitalize(): conf for flower, conf in top_3}
                    st.bar_chart(chart_data)
        
        else:
            st.info("👆 Upload an image to get started!")
            st.image("https://via.placeholder.com/400x300/f0f0f0/999999?text=Upload+Flower+Image", 
                    caption="Waiting for image upload...")
    
    # Instructions
    st.markdown("---")
    st.subheader("How to use:")
    st.markdown("""
    1. **Upload an image** using the file uploader on the left
    2. **Wait for processing** - the app will resize and analyze your image
    3. **View results** - see the predicted flower type with confidence scores
    4. **Try different images** to test the model's accuracy
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built using Streamlit and TensorFlow | "
        "For best results, use clear, well-lit images of flowers"
    )

if __name__ == "__main__":
    main()