"""MGAN generator: encoder, spatial and wavelet-frequency branches, decoder with 3-scale deep supervision.

Attribute names are kept identical to the original code so that old ``tf.train.Checkpoint`` files load.
Input: channels-last patches ``(batch, 64, 64, 64, 1)``; outputs at 16^3, 32^3 and 64^3.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dropout, ReLU

from brainpred.methods.gan.layers.normalization import NORM_TFA, instance_norm
from brainpred.methods.gan.layers.wavelet3d import DWT3D, IDWT3D


class MetamorphicGenerator(tf.keras.Model):

    def __init__(self, norm=NORM_TFA):
        super().__init__()
        # ---- Encoder
        self.enc_conv3D1 = Conv3D(32, (3, 3, 3), strides=1, padding='same', name="enc_conv3D1")
        self.enc_in1 = instance_norm(norm, axis=-1, name="enc_in1")
        self.enc_relu1 = ReLU(name="enc_relu1")

        self.enc_conv3D2 = Conv3D(64, (3, 3, 3), strides=2, padding='same', name="enc_conv3D2")
        self.enc_in2 = instance_norm(norm, axis=-1, name="enc_in2")
        self.enc_relu2 = ReLU(name="enc_relu2")

        # ---- Spatial branch
        self.spatial_branch = SpatialBranch(norm)
        self.sp_up = Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same', name="sp_up")

        # ---- Frequency branch
        self.frequency_branch = FrequencyBranch(norm)

        # ---- Concatenation
        self.conc_conv3D = Conv3D(64, (3, 3, 3), strides=1, padding='same', name="conc_conv3D")
        self.conc_in = instance_norm(norm, axis=-1, name="conc_in")
        self.conc_relu = ReLU(name="conc_relu")

        # ---- Decoder
        self.dec_deconv3D = Conv3DTranspose(32, (3, 3, 3), strides=2, padding='same', name="dec_deconv3D")
        self.dec_in1 = instance_norm(norm, axis=-1, name="dec_in")
        self.dec_relu1 = ReLU(name="dec_relu")
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

        # ---- Deep supervision heads.
        # NOTE: ls1/ls2 apply a sigmoid twice (conv activation + float32 sigmoid); kept as trained.
        self.ls1_conv3D = Conv3D(1, (1, 1, 1), strides=1, activation='sigmoid', padding='same', name="ls1_conv3D")
        self.ls1_sig = tf.keras.layers.Activation('sigmoid', dtype='float32')

        self.ls2_conv3D = Conv3D(1, (1, 1, 1), strides=1, activation='sigmoid', padding='same', name="ls2_conv3D")
        self.ls2_sig = tf.keras.layers.Activation('sigmoid', dtype='float32')

        self.ls3_conv3D = Conv3D(1, (1, 1, 1), strides=1, padding='same', name="ls3_conv3D")
        self.ls3_sig = tf.keras.layers.Activation('sigmoid', dtype='float32')

    def call(self, inputs):
        # ---- Encoder
        x = self.enc_relu1(self.enc_in1(self.enc_conv3D1(inputs)))  # (64,64,64,32)
        x = self.enc_relu2(self.enc_in2(self.enc_conv3D2(x)))       # (32,32,32,64)
        x = self.dropout1(x)

        # ---- Spatial branch
        x_sp = self.spatial_branch(x)  # (16,16,16,64)
        x_ls1 = self.ls1_conv3D(x_sp)  # (16,16,16,1)
        x_sp = self.sp_up(x_sp)        # (32,32,32,64)

        # ---- Frequency branch
        x_fr = self.frequency_branch(x)  # (32,32,32,64)

        # ---- Concatenation
        x = tf.concat([x_sp, x_fr], axis=-1)  # (32,32,32,128)
        x = self.conc_relu(self.conc_in(self.conc_conv3D(x)))  # (32,32,32,64)
        x_ls2 = self.ls2_conv3D(x)  # (32,32,32,1)
        x = self.dropout2(x)

        # ---- Decoder
        x = self.dec_relu1(self.dec_in1(self.dec_deconv3D(x)))  # (64,64,64,32)
        x_ls3 = self.ls3_conv3D(x)  # (64,64,64,1)

        return [self.ls1_sig(x_ls1), self.ls2_sig(x_ls2), self.ls3_sig(x_ls3)]


class SpatialBranch(tf.keras.layers.Layer):
    def __init__(self, norm=NORM_TFA):
        super().__init__()
        self.sp_conv = Conv3D(64, (3, 3, 3), strides=2, padding='same')
        self.sp_norm = instance_norm(norm, axis=-1)
        self.sp_relu = ReLU()
        self.transfer_block = TransferBlock(norm=norm)

    def call(self, inputs):
        x = self.sp_relu(self.sp_norm(self.sp_conv(inputs)))
        return self.transfer_block(x)


class FrequencyBranch(tf.keras.layers.Layer):
    """Per-channel 3D DWT, one transfer block per sub-band, then per-channel inverse DWT."""

    def __init__(self, norm=NORM_TFA):
        super().__init__()
        self.dwt = DWT3D(wavelet_name="bior1.3")
        self.idwt = IDWT3D(wavelet_name="bior1.3")
        self.transfer_blocks = [TransferBlock(norm=norm) for _ in range(8)]

    def call(self, inputs):
        x = inputs  # (32,32,32,64)
        n_layers = x.shape[-1]
        n_components = 8

        # DWT of each channel, grouped by sub-band
        x_dwt_layers = [[] for _ in range(n_components)]
        for layer in range(n_layers):
            x_dwt = self.dwt(tf.expand_dims(x[..., layer], axis=-1))  # (16,16,16,8)
            for i_comp in range(n_components):
                x_dwt_layers[i_comp].append(tf.expand_dims(x_dwt[..., i_comp], axis=-1))

        # Transfer block on each sub-band
        x_dwt_transfer_layers = []
        for i_comp in range(n_components):
            x_transfer = tf.concat(x_dwt_layers[i_comp], axis=-1)  # (16,16,16,64)
            x_dwt_transfer_layers.append(self.transfer_blocks[i_comp](x_transfer))

        # Inverse DWT of each channel
        x_idwt = []
        for layer in range(x_dwt_transfer_layers[0].shape[-1]):
            x_idwt_layer = [tf.expand_dims(x_dwt_transfer_layers[i_comp][..., layer], axis=-1)
                            for i_comp in range(n_components)]  # 8 x (16,16,16,1)
            x_idwt.append(self.idwt(tf.concat(x_idwt_layer, axis=-1)))  # (32,32,32,1)
        return tf.concat(x_idwt, axis=-1)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, norm=NORM_TFA):
        super().__init__()
        self.conv1 = Conv3D(64, (3, 3, 3), padding='same')
        self.norm1 = instance_norm(norm, axis=-1)
        self.relu1 = ReLU()
        self.conv2 = Conv3D(64, (3, 3, 3), padding='same')
        self.norm2 = instance_norm(norm, axis=-1)
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.relu1(self.norm1(self.conv1(inputs)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return tf.keras.layers.add([inputs, x])


class TransferBlock(tf.keras.layers.Layer):
    def __init__(self, num_residual_blocks=9, norm=NORM_TFA):
        super().__init__()
        self.residual_blocks = [ResidualBlock(norm) for _ in range(num_residual_blocks)]

    def call(self, inputs):
        x = inputs
        for block in self.residual_blocks:
            x = block(x)
        return x
