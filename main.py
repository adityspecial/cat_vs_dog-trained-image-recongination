import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Patch extraction layer
class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Patch extraction layer
class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.mlp = tf.keras.Sequential([
            layers.Dense(projection_dim * 2, activation="gelu"),
            layers.Dense(projection_dim),
        ])
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()

    def call(self, inputs, training, return_attention=False):
        attention_output, attention_scores = self.mha(inputs, inputs, return_attention_scores=True)
        x = self.layer_norm1(inputs + attention_output)
        x = self.layer_norm2(x + self.mlp(x))
        
        if return_attention:
            return x, attention_scores
        return x

# Vision Transformer Model
class ViTModel(tf.keras.Model):
    def __init__(self, patch_size, num_patches, projection_dim, transformer_layers, num_heads, mlp_head_units):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.patch_extract = PatchExtract(patch_size)
        self.projection = layers.Dense(projection_dim)  # Project patches to projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        
        self.transformer_blocks = [TransformerBlock(projection_dim, num_heads) for _ in range(transformer_layers)]
        
        self.mlp_head = tf.keras.Sequential([
            layers.Dense(mlp_head_units, activation="gelu"), 
            layers.Dense(1, activation="sigmoid")
        ])

  
    def call(self, inputs, training=False, return_attention=False):
        patches = self.patch_extract(inputs)
        patches = self.projection(patches)  # Apply projection to match position_embedding dimension
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = self.position_embedding(positions)
        x = patches + position_embeddings

        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attention = transformer_block(x, training=training, return_attention=True)
            attention_weights.append(attention)
        
        x = tf.reduce_mean(x, axis=1)
        x = self.mlp_head(x)

        if return_attention:
            return x, attention_weights
        return x

# Function to create and compile the model
def create_vit_model():
    input_shape = (224, 224, 3)
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_layers = 8
    mlp_head_units = 256

    inputs = layers.Input(shape=input_shape)
    model = ViTModel(
        patch_size,
        num_patches,
        projection_dim,
        transformer_layers,
        num_heads,
        mlp_head_units,
    )
    outputs = model(inputs)
    return tf.keras.Model(inputs, outputs)

# Data preparation
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_and_prepare_data():
    (train_data, val_data), dataset_info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True
    )

    train_data = train_data.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_data, val_data

# Training the model
def train_vit_model(model, train_data, val_data, epochs=10):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    checkpoint_filepath = "best_vit_model.keras"
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True),
        ],
    )
    return history

# Main function to train the model with TensorFlow Datasets
def main():
    # Load and prepare data
    train_data, val_data = load_and_prepare_data()

    # Create and train model
    model = create_vit_model()
    history = train_vit_model(model, train_data, val_data)

if __name__ == "__main__":
    main()
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.mlp = tf.keras.Sequential([
            layers.Dense(projection_dim * 2, activation="gelu"),
            layers.Dense(projection_dim),
        ])
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()

    def call(self, inputs, training, return_attention=False):
        attention_output, attention_scores = self.mha(inputs, inputs, return_attention_scores=True)
        x = self.layer_norm1(inputs + attention_output)
        x = self.layer_norm2(x + self.mlp(x))
        
        if return_attention:
            return x, attention_scores
        return x

# Vision Transformer Model
class ViTModel(tf.keras.Model):
    def __init__(self, patch_size, num_patches, projection_dim, transformer_layers, num_heads, mlp_head_units):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.patch_extract = PatchExtract(patch_size)
        self.projection = layers.Dense(projection_dim)  # Project patches to projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        
        self.transformer_blocks = [TransformerBlock(projection_dim, num_heads) for _ in range(transformer_layers)]
        
        self.mlp_head = tf.keras.Sequential([
            layers.Dense(mlp_head_units, activation="gelu"), 
            layers.Dense(1, activation="sigmoid")
        ])

  
    def call(self, inputs, training=False, return_attention=False):
        patches = self.patch_extract(inputs)
        patches = self.projection(patches)  # Apply projection to match position_embedding dimension
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = self.position_embedding(positions)
        x = patches + position_embeddings

        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attention = transformer_block(x, training=training, return_attention=True)
            attention_weights.append(attention)
        
        x = tf.reduce_mean(x, axis=1)
        x = self.mlp_head(x)

        if return_attention:
            return x, attention_weights
        return x

# Function to create and compile the model
def create_vit_model():
    input_shape = (224, 224, 3)
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_layers = 8
    mlp_head_units = 256

    inputs = layers.Input(shape=input_shape)
    model = ViTModel(
        patch_size,
        num_patches,
        projection_dim,
        transformer_layers,
        num_heads,
        mlp_head_units,
    )
    outputs = model(inputs)
    return tf.keras.Model(inputs, outputs)

# Data preparation
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_and_prepare_data():
    (train_data, val_data), dataset_info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True
    )

    train_data = train_data.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_data, val_data

# Training the model
def train_vit_model(model, train_data, val_data, epochs=10):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    checkpoint_filepath = "best_vit_model.keras"
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True),
        ],
    )
    return history

# Main function to train the model with TensorFlow Datasets
def main():
    # Load and prepare data
    train_data, val_data = load_and_prepare_data()

    # Create and train model
    model = create_vit_model()
    history = train_vit_model(model, train_data, val_data)

if __name__ == "__main__":
    main()
