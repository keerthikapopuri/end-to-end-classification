import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.mobilenet.MobileNet(input_shape=self.config.params_image_size,weights=self.config.params_weights,
        include_top=self.config.params_include_top
            )
        # Save the base model for future use
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till):
        # Freeze layers if necessary
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False  # Ensure layers are non-trainable
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False  # Freeze layers up to a certain point

        # Add new layers on top of the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add a fully connected dense layer with 512 units and ReLU activation

        fc_layer3 = tf.keras.layers.Dense(100, activation='relu')(flatten_in)
        
        # Add Batch Normalization for regularization
        bn_layer = tf.keras.layers.BatchNormalization()(fc_layer3)
        
        # Add a Dropout layer for reducing overfitting
        dropout_layer = tf.keras.layers.Dropout(0.5)(bn_layer)
        
        # Add another dense layer with fewer units
        fc_layer4 = tf.keras.layers.Dense(50, activation='relu')(dropout_layer)
        fc_layer5 = tf.keras.layers.Dense(15, activation='relu')(fc_layer4)
        
        # Final prediction layer with softmax activation
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(fc_layer5)

        # Combine the model inputs and outputs
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        # Update the base model by adding new layers and compiling it
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,  # Freeze all base model layers
            freeze_till=None,  # No specific freezing point
        )

        # Save the updated full model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save the trained model to the specified path
        model.save(path)
