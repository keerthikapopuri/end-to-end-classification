import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig, TrainingConfig
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Load the pre-trained model from the path specified in the config
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # Recreate the optimizer after loading the model
        new_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # Recompile the model with the newly initialized optimizer
        self.model.compile(
            optimizer=new_optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        # ImageDataGenerator arguments for training and validation
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        # Data flow parameters like image size, batch size, etc.
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # (height, width)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Create the validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Flow validation data from directory
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Apply data augmentation if enabled in config
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            # Use the same generator as validation when augmentation is off
            train_datagenerator = valid_datagenerator

        # Flow training data from directory
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save the trained model to the specified path
        model.save(path)

    def train(self):
        # Calculate steps per epoch for training and validation
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Save the trained model after training
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
