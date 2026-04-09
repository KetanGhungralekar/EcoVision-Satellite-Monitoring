import os
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, concatenate, Input, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model

# Architecture helper functions from the research notebook
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, sublayers=2):
    conv = input_tensor
    for idx in range(sublayers):
        conv = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer="he_normal", padding="same")(conv)
        if batchnorm:
            conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
    return conv

def conv2d_transpose_block(input_tensor, concatenate_tensor, n_filters, kernel_size=3, strides=2, transpose=False):
    if transpose:
        conv = Conv2DTranspose(n_filters, (kernel_size, kernel_size),
                               strides=(strides, strides), padding='same')(input_tensor)
    else:
        conv = Conv2D(n_filters, (kernel_size, kernel_size), activation = 'relu', padding = 'same',
                      kernel_initializer = 'he_normal')(UpSampling2D(size=(2, 2))(input_tensor))
    
    concatenation = concatenate([conv, concatenate_tensor])
    return concatenation

def build_unet(input_shape=(512, 512, 3), filters=[32, 64, 128, 256, 512, 1024, 2048], batchnorm=False, transpose=False):
    # Note: The filters in the notebook for model10 were [2**i for i in range(5, 12)]
    # which is [32, 64, 128, 256, 512, 1024, 2048]
    
    conv_dict = dict()
    inputs = Input(input_shape)
    
    x = inputs
    for idx, n_filters in enumerate(filters[:-1]):
        conv = conv2d_block(x, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm)
        x = MaxPooling2D((2, 2))(conv)
        conv_dict[f"conv2d_{idx+1}"] = conv
        
    conv_middle = conv2d_block(x, n_filters=filters[-1], kernel_size=3, batchnorm=batchnorm)
    
    x = conv_middle
    for idx, n_filters in enumerate(reversed(filters[:-1])):
        concatenation = conv2d_transpose_block(x,
                                               conv_dict[f"conv2d_{len(conv_dict) - idx}"],
                                               n_filters, kernel_size=2, strides=2, transpose=transpose)
        x = conv2d_block(concatenation, n_filters=n_filters, kernel_size=3, batchnorm = batchnorm)
        
    outputs = Conv2D(3, (1, 1), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class DeforestationPredictor:
    def __init__(self):
        self.model = None
        self.img_size = 512
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(base_dir, "weights", "deforestation_unet.h5")

    def load_model(self):
        if self.model is None:
            print("Loading Deforestation U-Net Model...")
            # Reconstruct the model with the exact parameters from the training notebook
            self.model = build_unet(
                input_shape=(512, 512, 3),
                filters=[32, 64, 128, 256, 512, 1024, 2048],
                batchnorm=False,
                transpose=False
            )
            self.model.load_weights(self.weights_path)
            print("Deforestation Model loaded successfully.")

    def predict(self, image_bytes):
        self.load_model()
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Pre-process
        original_size = img.shape[:2]
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_normalized = img_resized.astype("float32") / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Inference
        prediction = self.model.predict(img_batch)[0] # (512, 512, 3)

        # Post-process mask
        # Channel 0: Forest (Green), Channel 1: Deforest (Red)
        mask_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # We take the argmax to get the dominant class per pixel
        pred_class = np.argmax(prediction, axis=-1)
        
        # Color coding:
        # Class 0: Forest -> Green (0, 255, 0)
        # Class 1: Deforest -> Red (255, 0, 0)
        # Class 2: Other -> Black (0, 0, 0) or background
        
        mask_rgb[pred_class == 0] = [0, 255, 0]
        mask_rgb[pred_class == 1] = [255, 0, 0]
        
        # Create an ALPHA channel for the overlay
        mask_rgba = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2RGBA)
        mask_rgba[pred_class == 2, 3] = 0 # Transparent for 'Other'
        mask_rgba[pred_class != 2, 3] = 160 # Semi-transparent for forest/deforest

        # Resize back to original if needed? 
        # For the UI, we'll keep it square as per established pattern.
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA))
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return mask_base64

deforestation_predictor = DeforestationPredictor()
