import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

class MetamorphicDiscriminator(Model):
    def __init__(self):
        super(MetamorphicDiscriminator, self).__init__()

        # ---- Encoder
        self.enc_conv1 = Conv3D(64, (4, 4, 4), strides=2, padding='same')
        self.enc_in1 = InstanceNormalization(axis=-1, name="enc_in1")
        self.enc_relu1 = ReLU()

        self.enc_conv2 = Conv3D(128, (4, 4, 4), strides=2, padding='same')
        self.enc_in2 = InstanceNormalization(axis=-1, name="enc_in2")
        self.enc_relu2 = ReLU()

        self.enc_conv3 = Conv3D(256, (4, 4, 4), strides=2, padding='same')
        self.enc_in3 = InstanceNormalization(axis=-1, name="enc_in3")
        self.enc_relu3 = ReLU()


        # ---- Decoder
        self.dec_deconv1 = Conv3DTranspose(128, (4, 4, 4), strides=2, padding='same')
        self.dec_in1 = InstanceNormalization(axis=-1, name="dec_in1")
        self.dec_relu1 = ReLU()

        self.dec_deconv2 = Conv3DTranspose(64, (4, 4, 4), strides=2, padding='same')
        self.dec_in2 = InstanceNormalization(axis=-1, name="dec_in2")
        self.dec_relu2 = ReLU()

        self.dec_deconv3 = Conv3DTranspose(32, (4, 4, 4), strides=2, padding='same')
        self.dec_in3 = InstanceNormalization(axis=-1, name="dec_in3")
        self.dec_relu3 = ReLU()

        # ---- Output layer
        self.output = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')


    def call(self, inputs):
        # Encoding path
        x1 = self.enc_relu1(self.enc_in1(self.enc_conv1(inputs))) # (bs, 32, 32, 32, 64)
        x2 = self.enc_relu2(self.enc_in2(self.enc_conv2(x1))) # (bs, 16, 16, 16, 128)
        x3 = self.enc_relu3(self.enc_in3(self.enc_conv3(x2))) # (bs, 8, 8, 8, 256)

        # Decoding path with skip connections
        x = self.dec_relu1(self.dec_in1(self.dec_deconv1(x3))) # (bs, 16, 16, 16, 128)
        x = Add()([x, x2])  # Skip connection
        x = self.dec_relu2(self.dec_in2(self.dec_deconv2(x))) # (bs, 32, 32, 32, 64)
        x = Add()([x, x1])  # Skip connection
        x = self.dec_relu3(self.dec_in3(self.dec_deconv3(x))) # (bs, 64, 64, 64, 32)
        x = self.output(x)

        return x



# model = MetamorphicDiscriminator()

# res = model(tf.random.normal((1, 64, 64, 64, 1)))
# print(model.summary())

# print(len(res))
# print(res[0].shape)
# print(res[1].shape)
# print(res[2].shape)