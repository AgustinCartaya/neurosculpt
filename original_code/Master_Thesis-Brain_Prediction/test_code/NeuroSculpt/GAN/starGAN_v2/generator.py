import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super(Discriminator, self).__init__()
        dim_in = 2**14 // img_size
        self.blocks = []
        self.blocks.append(tf.keras.layers.Conv2D(dim_in, (3, 3), strides=(1, 1), padding='same'))

        repeat_num = int(tf.math.log(img_size) / tf.math.log(2)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        self.blocks.append(tf.keras.layers.LeakyReLU(0.2))
        self.blocks.append(tf.keras.layers.Conv2D(dim_out, (4, 4), strides=(1, 1), padding='valid'))
        self.blocks.append(tf.keras.layers.LeakyReLU(0.2))
        self.blocks.append(tf.keras.layers.Conv2D(num_domains, (1, 1), strides=(1, 1), padding='valid'))

    def call(self, x, y):
        out = x
        for layer in self.blocks:
            out = layer(out)
        out = tf.reshape(out, (tf.shape(out)[0], -1))  # (batch, num_domains)
        idx = tf.range(tf.shape(y)[0])
        idx = tf.cast(idx, dtype=tf.int64)
        out = tf.gather(out, idx, axis=1)  # (batch)
        return out


import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))