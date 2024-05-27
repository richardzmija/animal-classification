import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

# tf_keras_vis must be installed in colab first using pip

# Load model
main_path = "/content/drive/MyDrive/data"
model_path = os.path.join(main_path, "models/animal-classifier/my_model.keras")
model = tf.keras.models.load_model(model_path)

# Load images and convert them to 128x128 resolution

# Image directory relative to the data directory
imgs_path = "gradcam/animal-classification"
size = (128, 128)


def get_image_path(main_path, data_path, image_name):
    return os.path.join(main_path, data_path, image_name)


def preprocess_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=size)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Add one more dimension because the neural network expects
    # a four dimensional tensor where the first dimension is the batch size
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to be in the interval [0, 1]
    img_array = img_array / 255.0
    
    return img_array


cat_path = get_image_path(main_path, imgs_path, "cat_gradcam.jpg")
dog_path = get_image_path(main_path, imgs_path, "dog_gradcam.jpg")
horse_path = get_image_path(main_path, imgs_path, "horse_gradcam.jpg")
monkey_path = get_image_path(main_path, imgs_path, "monkey_gradcam.jpg")

cat_prep = preprocess_image(cat_path)
dog_prep = preprocess_image(dog_path)
horse_prep = preprocess_image(horse_path)
monkey_prep = preprocess_image(monkey_path)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
          return layer.name
    raise ValueError("No convolutional layer found in the model.")


# This will be the target layer for Gradcam
last_conv_layer_name = find_last_conv_layer(model)


# Change the activation function of the last layer to linear so
# that we can get the raw logits from the model and not probabilities
# calculated by softmax
def model_modifier(model):
    model.layers[-1].activation = tf.keras.activations.linear


# Select the output corresponding to the class with the highest
# predicted probability. This means that Gradcam will generate
# a heatmap for the most likely class.
def loss(output):
    return output[:, tf.argmax(output[0])]


# Create a Gradcam object, clone=True means that the model itself
# will not be modified during the computation
gradcam = Gradcam(model, model_modifier=model_modifier, clone=True)


# Define the score function for the class with the highest prediction score.
# It computes loss based on a specific classes' activations.
def get_heatmap(img_prep):
    predictions = model.predict(img_prep)
    predicted_class = np.argmax(predictions[0])
    score = CategoricalScore([predicted_class])

    heatmap = gradcam(score, cat_prep, penultimate_layer=last_conv_layer_name)
    return heatmap[0]


def overlay_heatmap(img_path, heatmap, alpha=0.6):
    """
    Put a heatmap on top of the original image.

    Parameters:
        img_path: path to the image
        heatmap: a 2D array of values
        alpha: transparency of the heatmap overlay
    """
    # Load the image
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    
    # Normalize the heatmap to the range [0, 1]
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap.astype(np.float32)
    heatmap /= np.max(heatmap)

    # Scale heatmap values to the range [0, 255] so they can
    # be used with 'jet' colormap.
    heatmap = np.uint8(255 * heatmap)

    # Color map jet maps normalized data values into RGBA values
    jet = plt.colormaps.get_cmap("jet")
    # Generate an array of 256 color values from 'jet' color map
    # using only RGB components
    jet_colors = jet(np.arange(256))[:, :3]
    # Use the heatmap as the index to the corresponding jet colors.
    jet_heatmap = jet_colors[heatmap]

    # Convert a heatmap into a PIL image
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    # Resize the heatmap to match the dimensions of the original image
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    # Convert resized heatmap image into a numpy array
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Scale the heatmap values by the transparency alpha and add it
    # to the original image. This creates the overlay effect in the image
    superimposed_img = jet_heatmap * alpha + img
    # Convert the combined array back to a PIL image for displaying
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


# Generate an image with an overlaid heatmap for all the four images
cat_heatmap_img = overlay_heatmap(cat_path, get_heatmap(cat_prep))
dog_heatmap_img = overlay_heatmap(dog_path, get_heatmap(dog_prep))
horse_heatmap_img = overlay_heatmap(horse_path, get_heatmap(horse_prep))
monkey_heatmap_img = overlay_heatmap(monkey_path, get_heatmap(monkey_prep))


def plot_side_by_side(images, titles=None):
    fig, axs = plt.subplots(1, len(images), figsize=(20, 30))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
        if titles:
            axs[i].set_title(titles[i])
    plt.show()


images = [cat_heatmap_img, dog_heatmap_img, horse_heatmap_img,
          monkey_heatmap_img]

titles = ["Cat", "Dog", "Horse", "Monkey"]

plot_side_by_side(images, titles)
