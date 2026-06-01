import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers

# 1. SE Attention Block
def se_block(x, reduction=16):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(channels // reduction, 4), activation='gelu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])

# 2. Conv Block with Regularization
def conv_block(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", 
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    return x

# 3. Xception Block
def xception_block(x, filters):
    shortcut = x
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation("gelu")(x)
    x = se_block(x)
    return x

# 4. Residual Block (এটি মিসিং ছিল, তাই যোগ করলাম)
def residual_block(x, filters):
    shortcut = x
    x = conv_block(x, filters, 3)
    x = conv_block(x, filters, 3)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("gelu")(x)
    return x

# 5. Build Model
def build_model(input_shape=(224, 224, 3), num_classes=3):
    inputs = Input(shape=input_shape)
    
    # Input Dropout to reduce noise sensitivity
    x = layers.Dropout(0.1)(inputs) 
    
    x = conv_block(inputs, 32, 3)
    x = layers.MaxPooling2D(2)(x)
    
    x = xception_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 128) # অতিরিক্ত লেয়ার যা মডেলের গভীরতা বাড়াবে
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Dense block with higher regularization
    x = layers.Dense(256, activation="gelu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x) 
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)

if __name__ == "__main__":
    model = build_model()
    model.summary()
    print("Model built successfully with all blocks.")