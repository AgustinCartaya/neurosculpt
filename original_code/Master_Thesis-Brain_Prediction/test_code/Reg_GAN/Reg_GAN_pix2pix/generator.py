import tensorflow as tf
from tensorflow.keras.layers import Conv3D, BatchNormalization, Conv3DTranspose, ReLU
from tensorflow.keras.activations import tanh
from tensorflow_addons.layers import InstanceNormalization


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv3D(in_features, (3, 3, 3), padding='same')
        self.norm1 = InstanceNormalization(axis=-1)
        self.relu1 = ReLU()
        self.conv2 = Conv3D(in_features, (3, 3, 3), padding='same')
        self.norm2 = InstanceNormalization(axis=-1)
        # self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        # x = self.relu2(x)
        return tf.keras.layers.add([inputs, x])  # Shortcut connection



class Generator(tf.keras.Model):
    def __init__(self, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        self.conv1 = Conv3D(32, kernel_size=7, padding='same')
        self.norm1 = InstanceNormalization(axis=-1)
        self.relu1 = ReLU()

        # Downsampling
        self.conv2 = Conv3D(64, kernel_size=3, strides=(2, 2, 2), padding='same')
        self.norm2 = InstanceNormalization(axis=-1)
        self.relu2 = ReLU()

        self.conv3 = Conv3D(128, kernel_size=3, strides=(2, 2, 2), padding='same')
        self.norm3 = InstanceNormalization(axis=-1)
        self.relu3 = ReLU()

        # Residual blocks
        self.res_blocks = [ResidualBlock(128) for _ in range(n_residual_blocks)]

        # Upsampling
        self.deconv1 = Conv3DTranspose(64, kernel_size=3, strides=(2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.norm4 = InstanceNormalization(axis=-1)
        self.relu4 = ReLU()

        self.deconv2 = Conv3DTranspose(32, kernel_size=3, strides=(2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.norm5 = InstanceNormalization(axis=-1)
        self.relu5 = ReLU()

        # Output layer
        self.conv4 = Conv3D(1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = inputs

        # Initial convolution block
        x = self.relu1(self.norm1(self.conv1(x)))

        # Downsampling
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Upsampling
        x = self.relu4(self.norm4(self.deconv1(x)))
        x = self.relu5(self.norm5(self.deconv2(x)))

        # Output layer
        x = self.conv4(x)

        return x




# model = Generator()
# res = model(tf.random.normal((1, 64, 64, 64, 1)))
# print(model.summary())



