import tensorflow as tf

class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        
    def call(self, inputs, training=None):
        def noised():
            return tf.clip_by_value(
                inputs + tf.random.normal(shape=tf.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev),
                0.,
                255.
            )
        
        return tf.keras.backend.in_train_phase(noised, inputs, training=training)
    
    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class ContrastNoiseSingle(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(ContrastNoiseSingle, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        
    def call(self, inputs, training=None):
        def noised():
            return tf.concat([
                tf.image.random_contrast(inputs[:,:,:,0:1], 1.-self.stddev, 1.+self.stddev),
                tf.image.random_contrast(inputs[:,:,:,1:2], 1.-self.stddev, 1.+self.stddev),
                tf.image.random_contrast(inputs[:,:,:,2:3], 1.-self.stddev, 1.+self.stddev)]
                ,axis=-1)
        
        return tf.keras.backend.in_train_phase(noised, inputs, training=training)
    
    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(ContrastNoiseSingle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class ContrastNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(ContrastNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        
    def call(self, inputs, training=None):
        def noised():
            return  tf.image.random_contrast(inputs, 1.-self.stddev, 1.+self.stddev)
        
        return tf.keras.backend.in_train_phase(noised, inputs, training=training)
    
    def build(self, input_shape):
        super(ContrastNoise, self).build(input_shape)
        
    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(ContrastNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class BrightnessNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(BrightnessNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        
    def call(self, inputs, training=None):
        def noised():
            return tf.clip_by_value(
                tf.image.random_brightness(inputs,self.stddev),
                0.,
                255.
            )
        
        return tf.keras.backend.in_train_phase(noised, inputs, training=training)
    
    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(BrightnessNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
