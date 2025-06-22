import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from cvae_model import ConditionalVAE

# --- Page Configuration ---
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model(model_path="models/cvae_mnist_weights.h5"):
    """Loads the CVAE model and its weights."""
    model = ConditionalVAE(latent_dim=50, num_classes=10)
    if os.path.exists(model_path):
        # Build the model by passing a dummy input
        dummy_input = tf.random.normal((1, 28, 28, 1))
        dummy_label = tf.constant([0], dtype=tf.int32)
        model([dummy_input, dummy_label])
        model.load_weights(model_path)
        st.sidebar.success("✅ Trained model loaded.")
    else:
        st.sidebar.warning("⚠️ Model weights not found. Using an untrained model.")
    return model

# --- Image Generation ---
class DigitGenerator:
    """A wrapper class for generating digits using the trained VAE model."""
    def __init__(self, model):
        self.model = model

    def generate_batch(self, digit: int, count: int = 5):
        """Generates a batch of unique images for a given digit."""
        generated_tensors = self.model.generate(digit, num_samples=count)
        images = [img_tensor.numpy().squeeze() for img_tensor in generated_tensors]
        return [np.clip(img, 0, 1) for img in images]


# --- Main Application ---
def main():
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Generator</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Generate unique, handwritten-style digits using a Conditional Variational Autoencoder (CVAE).</p>", unsafe_allow_html=True)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("🎛️ Controls")
        digit_to_generate = st.selectbox(
            "Select a Digit (0-9):",
            options=list(range(10)),
            index=1,
            help="Choose the digit you want to generate."
        )
        st.info("The model generates unique variations each time.")

        st.header("⚙️ Model Info")
        st.markdown("""
        - **Type:** Conditional VAE
        - **Latent Dim:** 50
        - **Training:** MNIST, 50 Epochs
        """)

    # --- Load Model and Generator ---
    cvae_model = load_model()
    generator = DigitGenerator(cvae_model)

    # --- Generate and Display Images ---
    if st.button(f"🚀 Generate 5 Images of '{digit_to_generate}'", use_container_width=True, type="primary"):
        with st.spinner("🎨 Generating..."):
            images = generator.generate_batch(digit_to_generate, count=5)

            st.subheader(f"Generated samples for digit: {digit_to_generate}")
            cols = st.columns(5)
            for i, img_array in enumerate(images):
                with cols[i]:
                    pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
                    st.image(pil_img, caption=f"Sample {i + 1}", use_column_width=True)

    # --- Expander for Technical Details ---
    with st.expander("🔧 Click here for Technical Details"):
        st.markdown("""
        This application uses a Conditional Variational Autoencoder (CVAE) built with TensorFlow.
        A CVAE is a type of generative model that can learn to produce new data similar to what it was trained on,
        with the added ability to control what it generates via a conditional input (in this case, the digit label).
        """)
        st.code("""
# Loss Function = Reconstruction Loss + KL Divergence
reconstruction_loss = binary_crossentropy(original, reconstructed)
kl_loss = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        """, language="python")

if __name__ == "__main__":
    main()