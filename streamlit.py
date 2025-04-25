import streamlit as st
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from neural_network import load_parameters, predict, forward_propagation

# Load the Neural Network model parameters
def load_model():
    parameters = load_parameters('model_parameters.npz')
    return parameters

# Preprocess the image for prediction
def preprocess_image(image):
    # Convert the image to grayscale (if it's not already)
    image = image.convert("L")
    
    # Resize the image to 28x28 pixels, as required by the MNIST model
    image = image.resize((28, 28))
    
    # Convert the image to a numpy array and normalize the pixel values
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    # Flatten the image to a 1D array (28x28 -> 784)
    image_array = image_array.flatten()
    
    # Apply scaling if the model was trained with StandardScaler
    scaler = StandardScaler()
    image_array = scaler.fit_transform(image_array.reshape(-1, 1)).flatten()
    
    return image_array

# Prediction function for Neural Network model
def predict_digit(image, parameters):
    image_array = preprocess_image(image)
    
    # Reshape the image for the model input (Neural Network expects column vector)
    image_array = image_array.reshape(784, 1)
    
    # Make prediction
    prediction = predict(image_array, parameters)[0]
    return prediction

def main():
    st.title('MNIST Digit Classifier (Neural Network)')
    st.write('Upload an image of a handwritten digit (0-9) and I will predict it.')
    
    # File uploader widget for uploading an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)  
        st.write("")
        
        # Load the trained Neural Network model parameters
        parameters = load_model()
        
        # Get the neural network prediction
        neural_network_prediction = predict_digit(image, parameters)
        
        # Display the preprocessed image
        preprocessed = image.convert("L").resize((28, 28))
        st.image(preprocessed, caption='Preprocessed Image (28x28)', width=150)
        
        # Get probabilities
        image_array = preprocess_image(image)
        image_array = image_array.reshape(784, 1)
        probs, _ = forward_propagation(image_array, parameters)
        probs = probs.flatten()
        
        # Find the highest probability and its index
        highest_prob_index = np.argmax(probs)
        highest_prob = probs[highest_prob_index]
        
        # Display probabilities
        st.write("Probabilities for each digit:")
        for i, prob in enumerate(probs):
            if i == highest_prob_index:
                st.write(f"Digit {i}: {prob*100:.2f}% (Highest)")
            else:
                st.write(f"Digit {i}: {prob*100:.2f}%")
        
        # Compare the predictions and use the highest probability one
        st.write("### Final Prediction Results:")
        st.write(f"Neural Network Model Output: {neural_network_prediction}")
        st.write(f"Highest Probability Digit: {highest_prob_index} (Confidence: {highest_prob*100:.2f}%)")
        
        # Determine final output based on highest probability
        final_prediction = highest_prob_index
        st.markdown(f"## Final Prediction: {final_prediction}")

# Run the app
if __name__ == "__main__":
    main()
