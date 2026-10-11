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
        # self.enc_in1 = GroupNormalization(groups=-1, axis=-1, name="enc_in1")
        self.enc_relu1 = ReLU()

        self.enc_conv2 = Conv3D(128, (4, 4, 4), strides=2, padding='same')
        self.enc_in2 = InstanceNormalization(axis=-1, name="enc_in2")
        # self.enc_in2 = GroupNormalization(groups=-1, axis=-1, name="enc_in2")
        self.enc_relu2 = ReLU()

        self.enc_conv3 = Conv3D(256, (4, 4, 4), strides=2, padding='same')
        self.enc_in3 = InstanceNormalization(axis=-1, name="enc_in3")
        # self.enc_in3 = GroupNormalization(groups=-1, axis=-1, name="enc_in3")
        self.enc_relu3 = ReLU()


        # ---- Decoder
        self.dec_deconv1 = Conv3DTranspose(128, (4, 4, 4), strides=2, padding='same')
        self.dec_in1 = InstanceNormalization(axis=-1, name="dec_in1")
        # self.dec_in1 = GroupNormalization(groups=-1, axis=-1, name="dec_in1")
        self.dec_relu1 = ReLU()

        self.dec_deconv2 = Conv3DTranspose(64, (4, 4, 4), strides=2, padding='same')
        self.dec_in2 = InstanceNormalization(axis=-1, name="dec_in2")
        # self.dec_in2 = GroupNormalization(groups=-1, axis=-1, name="dec_in2")
        self.dec_relu2 = ReLU()

        self.dec_deconv3 = Conv3DTranspose(32, (4, 4, 4), strides=2, padding='same')
        self.dec_in3 = InstanceNormalization(axis=-1, name="dec_in3")
        # self.dec_in3 = GroupNormalization(groups=-1, axis=-1, name="dec_in3")
        self.dec_relu3 = ReLU()

        # ---- Output layer
        self.ls1_conv = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        # self.ls1_in = InstanceNormalization(axis=-1, name="ls1_in")
        # self.ls1_relu = ReLU()

        self.ls2_conv = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        # self.ls2_in = InstanceNormalization(axis=-1, name="ls2_in")
        # self.ls2_relu = ReLU()

        self.ls3_conv = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')
        # self.ls3_in = InstanceNormalization(axis=-1, name="ls3_in")
        # self.ls3_relu = ReLU()


    def call(self, inputs):
        # Encoding path
        x1 = self.enc_relu1(self.enc_in1(self.enc_conv1(inputs))) # (bs, 32, 32, 32, 64)
        x2 = self.enc_relu2(self.enc_in2(self.enc_conv2(x1))) # (bs, 16, 16, 16, 128)
        x3 = self.enc_relu3(self.enc_in3(self.enc_conv3(x2))) # (bs, 8, 8, 8, 256)

        # Decoding path with skip connections
        x = self.dec_relu1(self.dec_in1(self.dec_deconv1(x3))) # (bs, 16, 16, 16, 128)
        # x_ls1 = self.ls1_relu(self.ls1_in(self.ls1_conv(x)))
        x_ls1 = self.ls1_conv(x)

        x = Add()([x, x2])  # Skip connection
        x = self.dec_relu2(self.dec_in2(self.dec_deconv2(x))) # (bs, 32, 32, 32, 64)
        # x_ls2 = self.ls2_relu(self.ls2_in(self.ls2_conv(x)))
        x_ls2 = self.ls2_conv(x)
        
        x = Add()([x, x1])  # Skip connection
        x = self.dec_relu3(self.dec_in3(self.dec_deconv3(x))) # (bs, 64, 64, 64, 32)
        # x_ls3 = self.ls3_relu(self.ls3_in(self.ls3_conv(x)))
        x_ls3 = self.ls3_conv(x)

        return [x_ls1, x_ls2, x_ls3]



def visualize_model():
    model = MetamorphicDiscriminator()
    patch_size= tf.random.normal((1, 32, 32, 32, 1), dtype=tf.float16)
    res = model(patch_size)

    print(model.summary())

    print(len(res)) 
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)



# visualize_model()
# Total params: 5,380,291 (20.52 MB)
# Trainable params: 5,380,291 (20.52 MB)
# Non-trainable params: 0 (0.00 B)
