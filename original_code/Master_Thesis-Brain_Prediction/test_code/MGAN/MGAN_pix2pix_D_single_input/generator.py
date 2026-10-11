import sys
import os

current_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_path, '../../../'))
sys.path.append(base_path)

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, BatchNormalization, ReLU, Conv3DTranspose, Dropout
from tensorflow_addons.layers import InstanceNormalization
from layers.Wavelet3D import DWT3D, IDWT3D

class MetamorphicGenerator(tf.keras.Model):

    def __init__(self):
        super(MetamorphicGenerator, self).__init__()
        # ---- Encoder
        self.enc_conv3D1 = Conv3D(32, (3, 3, 3), strides=1, padding='same', name="enc_conv3D1")
        self.enc_in1 = InstanceNormalization(axis=-1, name="enc_in1")
        # self.enc_in1 = GroupNormalization(groups=-1, axis=-1, name="enc_in1")
        self.enc_relu1 = ReLU(name="enc_relu1")

        self.enc_conv3D2 = Conv3D(64, (3, 3, 3), strides=2, padding='same', name="enc_conv3D2")
        self.enc_in2 = InstanceNormalization(axis=-1, name="enc_in2")
        # self.enc_in2 = GroupNormalization(groups=-1, axis=-1, name="enc_in2")
        self.enc_relu2 = ReLU(name="enc_relu2")

        # ---- Spatial Branch
        self.spatial_branch = SpatialBranch()
        self.sp_up = Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same', name="sp_up")

        # ---- Frequency Branch
        self.frequency_branch = FrequencyBranch()

        # ---- concatenation
        self.conc_conv3D = Conv3D(64, (3, 3, 3), strides=1, padding='same', name="conc_conv3D")
        self.conc_in = InstanceNormalization(axis=-1, name="conc_in")
        # self.conc_in = GroupNormalization(groups=-1, axis=-1, name="conc_in")
        self.conc_relu = ReLU(name="conc_relu")

        # ---- Decoder
        self.dec_deconv3D = Conv3DTranspose(32, (3, 3, 3), strides=2, padding='same', name="dec_deconv3D")
        self.dec_in1 = InstanceNormalization(axis=-1, name="dec_in")
        # self.dec_in1 = GroupNormalization(groups=-1, axis=-1, name="dec_in")
        self.dec_relu1 = ReLU(name="dec_relu")

        # ---- dropout layers
        self.dropout1 = Dropout(0.1) ## added dropout layer
        self.dropout2 = Dropout(0.1) ## added dropout layer

        # deep supervision
        self.ls1_conv3D = Conv3D(1, (1, 1, 1), strides=1, activation='sigmoid', padding='same', name="ls1_conv3D")
        self.ls1_in = InstanceNormalization(axis=-1, name="ls1_in")
        # self.ls1_in = GroupNormalization(groups=-1, axis=-1, name="ls1_in")
        self.ls1_relu = ReLU(name="ls1_relu")

        self.ls2_conv3D = Conv3D(1, (1, 1, 1), strides=1, activation='sigmoid', padding='same', name="ls2_conv3D")
        self.ls2_in = InstanceNormalization(axis=-1, name="ls2_in")
        # self.ls2_in = GroupNormalization(groups=-1, axis=-1, name="ls2_in")
        self.ls2_relu = ReLU(name="ls2_relu")

        self.ls3_conv3D = Conv3D(1, (1, 1, 1), strides=1, activation='sigmoid', padding='same', name="ls3_conv3D")
        self.ls3_in = InstanceNormalization(axis=-1, name="ls3_in")
        # self.ls3_in = GroupNormalization(groups=-1, axis=-1, name="ls3_in")
        self.ls3_relu = ReLU(name="ls3_relu")

    def call(self, inputs):
        x = inputs # 64x64x64x1
        # ---- Encoder
        x = self.enc_conv3D1(inputs) # (64,64,64,32)
        x = self.enc_in1(x)
        x = self.enc_relu1(x)

        x = self.enc_conv3D2(x) # (32,32,32,64)
        x = self.enc_in2(x)
        x = self.enc_relu2(x)

        # droput layer
        x = self.dropout1(x) ## added dropout layer

        # ---- Spatial Branch
        x_sp = self.spatial_branch(x) # (16,16,16,64) 
        # -- deep supervision LS1
        x_ls1 = self.ls1_conv3D(x_sp) # (16,16,16,1)
        x_ls1 = self.ls1_in(x_ls1)
        x_ls1 = self.ls1_relu(x_ls1) # Ls1 
        # --
        x_sp = self.sp_up(x_sp) # (32,32,32,64)

        # ---- Frequency Branch
        x_fr = self.frequency_branch(x) # (32,32,32,64)

        # print(x_sp_up.shape)
        # print(x_fr.shape)
        # ---- concatenation
        x = tf.concat([x_sp, x_fr], axis=-1) # (32,32,32,128)
        x = self.conc_conv3D(x) # (32,32,32,64)
        x = self.conc_in(x)
        x = self.conc_relu(x)

        # -- deep supervision LS2
        x_ls2 = self.ls2_conv3D(x) # (32,32,32,1)
        x_ls2 = self.ls2_in(x_ls2)
        x_ls2 = self.ls2_relu(x_ls2)
        # --

        # droput layer
        x = self.dropout2(x) ## added dropout layer

        # ---- Decoder
        x = self.dec_deconv3D(x) # (64,64,64,32)
        x = self.dec_in1(x)
        x = self.dec_relu1(x)

        # -- deep supervision LS2
        x_ls3 = self.ls3_conv3D(x) # (64,64,64,1)
        x_ls3 = self.ls3_in(x_ls3)
        x_ls3 = self.ls3_relu(x_ls3)
        # --

        return [x_ls1, x_ls2, x_ls3]
        # return x_ls3



class SpatialBranch(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialBranch, self).__init__()
        self.sp_conv = Conv3D(64, (3, 3, 3), strides=2, padding='same')
        self.sp_norm = InstanceNormalization()
        # self.sp_norm = GroupNormalization(groups=-1, axis=-1)
        self.sp_relu = ReLU()

        # Spatial Transfer Block
        self.transfer_block = TransferBlock()

    def call(self, inputs):
        x = self.sp_conv(inputs)
        x = self.sp_norm(x)
        x = self.sp_relu(x)
        x = self.transfer_block(x)
        return x



class FrequencyBranch(tf.keras.layers.Layer):
    def __init__(self):
        super(FrequencyBranch, self).__init__()
        self.dwt = DWT3D()
        self.idwt = IDWT3D()
        self.transfer_blocks = [TransferBlock() for _ in range(8)]

    def call(self, inputs):
        x = inputs  # (32,32,32,64)
        n_layers = x.shape[-1]
        n_components = 8

        # obtain the dwt of each layer
        x_dwt_layers = []
        for layer in range(n_layers):
            x_dwt = self.dwt(tf.expand_dims(x[..., layer], axis=-1))  # (16,16,16,8)
            for i_comp in range(n_components):
                if len(x_dwt_layers) <= i_comp:
                    x_dwt_layers.append([])
                x_dwt_layers[i_comp].append(tf.expand_dims(x_dwt[..., i_comp], axis=-1))

        # apply transfer blocks to each component and obtain the idwt
        x_dwt_transfer_layers = []
        for i_comp in range(n_components):
            x_transfer = tf.concat(x_dwt_layers[i_comp], axis=-1)  # (16,16,16,64)
            x_idwt = self.transfer_blocks[i_comp](x_transfer) 
            x_dwt_transfer_layers.append(x_idwt)

        # print(len(x_dwt_transfer_layers))
        # print(x_dwt_transfer_layers[0].shape)
        # obtain the idwt of each layer
        x_idwt = []
        for layer in range(x_dwt_transfer_layers[0].shape[-1]):
            x_idwt_layer = [tf.expand_dims(x_dwt_transfer_layers[i_comp][..., layer], axis=-1) for i_comp in range(n_components)] #(8, 16,16,16,1)
            x_idwt.append(self.idwt(tf.concat(x_idwt_layer, axis=-1))) #(64, 16,16,16,1)
        # print(len(x_idwt))
        # print(x_idwt[0].shape)  
        return tf.concat(x_idwt, axis=-1)



class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv3D(64, (3, 3, 3), padding='same')
        self.norm1 = InstanceNormalization(axis=-1)
        # self.norm1 = GroupNormalization(groups=-1, axis=-1)
        self.relu1 = ReLU()
        self.conv2 = Conv3D(64, (3, 3, 3), padding='same')
        self.norm2 = InstanceNormalization(axis=-1)
        # self.norm2 = GroupNormalization(groups=-1, axis=-1)
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return tf.keras.layers.add([inputs, x])  # Shortcut connection


class TransferBlock(tf.keras.layers.Layer):
    def __init__(self, num_residual_blocks=4):
        super(TransferBlock, self).__init__()
        self.residual_blocks = [ResidualBlock() for _ in range(num_residual_blocks)]

    def call(self, inputs):
        x = inputs
        for block in self.residual_blocks:
            x = block(x)
        return x
    



def visualize_model():
    model = MetamorphicGenerator()
    patch_size= tf.random.normal((1, 32, 32, 32, 1), dtype=tf.float16)
    res = model(patch_size)

    print(model.summary())

    print(len(res)) 
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)

# visualize_model()
