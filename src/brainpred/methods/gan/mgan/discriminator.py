"""MGAN discriminator: U-shaped voxel-wise discriminator with 3-scale outputs (16^3, 32^3, 64^3)."""
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv3D, Conv3DTranspose, ReLU
from tensorflow.keras.models import Model

from brainpred.methods.gan.layers.normalization import NORM_TFA, instance_norm


class MetamorphicDiscriminator(Model):

    def __init__(self, norm=NORM_TFA):
        super().__init__()
        # ---- Encoder
        self.enc_conv1 = Conv3D(64, (4, 4, 4), strides=2, padding='same')
        self.enc_in1 = instance_norm(norm, axis=-1, name="enc_in1")
        self.enc_relu1 = ReLU()

        self.enc_conv2 = Conv3D(128, (4, 4, 4), strides=2, padding='same')
        self.enc_in2 = instance_norm(norm, axis=-1, name="enc_in2")
        self.enc_relu2 = ReLU()

        self.enc_conv3 = Conv3D(256, (4, 4, 4), strides=2, padding='same')
        self.enc_in3 = instance_norm(norm, axis=-1, name="enc_in3")
        self.enc_relu3 = ReLU()

        # ---- Decoder
        self.dec_deconv1 = Conv3DTranspose(128, (4, 4, 4), strides=2, padding='same')
        self.dec_in1 = instance_norm(norm, axis=-1, name="dec_in1")
        self.dec_relu1 = ReLU()

        self.dec_deconv2 = Conv3DTranspose(64, (4, 4, 4), strides=2, padding='same')
        self.dec_in2 = instance_norm(norm, axis=-1, name="dec_in2")
        self.dec_relu2 = ReLU()

        self.dec_deconv3 = Conv3DTranspose(32, (4, 4, 4), strides=2, padding='same')
        self.dec_in3 = instance_norm(norm, axis=-1, name="dec_in3")
        self.dec_relu3 = ReLU()

        # ---- Output heads
        self.ls1_conv = Conv3D(1, (1, 1, 1), padding='same')
        self.ls1_sig = tf.keras.layers.Activation('sigmoid', dtype='float32')

        self.ls2_conv = Conv3D(1, (1, 1, 1), padding='same')
        self.ls2_sig = tf.keras.layers.Activation('sigmoid', dtype='float32')

        self.ls3_conv = Conv3D(1, (1, 1, 1), padding='same')
        self.ls3_sig = tf.keras.layers.Activation('sigmoid', dtype='float32')

    def call(self, inputs):
        # Encoding path
        x1 = self.enc_relu1(self.enc_in1(self.enc_conv1(inputs)))  # (bs, 32, 32, 32, 64)
        x2 = self.enc_relu2(self.enc_in2(self.enc_conv2(x1)))      # (bs, 16, 16, 16, 128)
        x3 = self.enc_relu3(self.enc_in3(self.enc_conv3(x2)))      # (bs, 8, 8, 8, 256)

        # Decoding path with skip connections
        x = self.dec_relu1(self.dec_in1(self.dec_deconv1(x3)))  # (bs, 16, 16, 16, 128)
        x_ls1 = self.ls1_sig(self.ls1_conv(x))

        x = Add()([x, x2])
        x = self.dec_relu2(self.dec_in2(self.dec_deconv2(x)))  # (bs, 32, 32, 32, 64)
        x_ls2 = self.ls2_sig(self.ls2_conv(x))

        x = Add()([x, x1])
        x = self.dec_relu3(self.dec_in3(self.dec_deconv3(x)))  # (bs, 64, 64, 64, 32)
        x_ls3 = self.ls3_sig(self.ls3_conv(x))

        return [x_ls1, x_ls2, x_ls3]
