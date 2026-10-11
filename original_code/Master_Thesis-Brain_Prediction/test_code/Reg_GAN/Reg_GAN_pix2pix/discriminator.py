import tensorflow as tf
from tensorflow.keras.layers import Conv3D, LeakyReLU, BatchNormalization, Flatten
from tensorflow_addons.layers import InstanceNormalization

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv3D(32, kernel_size=4, strides=2, padding='same')
        self.leakyrelu1 = LeakyReLU(alpha=0.2)

        self.conv2 = Conv3D(64, kernel_size=4, strides=2, padding='same')
        self.instance_norm1 = InstanceNormalization(axis=-1)
        self.leakyrelu2 = LeakyReLU(alpha=0.2)

        self.conv3 = Conv3D(128, kernel_size=4, strides=2, padding='same')
        self.instance_norm2 = InstanceNormalization(axis=-1)
        self.leakyrelu3 = LeakyReLU(alpha=0.2)

        self.conv4 = Conv3D(256, kernel_size=4, strides=2, padding='same')
        self.instance_norm3 = InstanceNormalization(axis=-1)
        self.leakyrelu4 = LeakyReLU(alpha=0.2)

        # FCN classification layer
        self.conv5 = Conv3D(1, kernel_size=4, padding='same')

    def call(self, x):
        x = self.leakyrelu1(self.conv1(x))
        x = self.leakyrelu2(self.instance_norm1(self.conv2(x)))
        x = self.leakyrelu3(self.instance_norm2(self.conv3(x)))
        x = self.leakyrelu4(self.instance_norm3(self.conv4(x)))
        x = self.conv5(x)
        
        # Average pooling and flatten
        # x = tf.keras.layers.GlobalAveragePooling3D()(x)
        
        return x
    

# model = Discriminator()
# res = model(tf.random.normal((1, 64, 64, 64, 1)))
# print(model.summary())
