from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax, PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

class CNNVocSep(Model):

  '''
      CNN based Vocal Separation Model
      args
        input_size: tuple of shape (batch, freq_bins, w_size)
        rec_field_size: int, receptive field size for convlution
        num_freq_bins: int, number of frequency bins after STFT transform
        num_conv_blocks: int, number of convolution blocks (Default: 2)
  '''

  def __init__(self, input_size, rec_field_size, num_freq_bins, num_conv_blocks=2):
    super(CNNVocSep, self).__init__()
    
    self.input_size = input_size
    self.rec_field_size = rec_field_size
    self.num_conv_blocks = num_conv_blocks
    self.num_freq_bins = num_freq_bins

  def conv_block(self, input_data, num_filter, filter_size, dropout=0):
    
    '''
      Convolutionn block
      args
        input data: numpy array 
        num_filter: int, number of convolution filters
        filter_size: int, size of filters (eg. 3x3, 5x5)
        dropout: int, dropout rate (Default: 0.25)
    '''

    out = Conv2D(num_filter, filter_size, padding='same')(input_data)
    out = LeakyReLU()(out)
    out = MaxPool2D(filter_size)(out)
    out = Dropout(dropout)(out)

    return out

  def bottleneck(self, input_data, last_dense_nodes=128, dropout=0.25):

    '''
      Bottleneck block
      args
        input_data: numpy array
        last_dense_nodes: int, number of nodes (Default: 128)
        dropout: dropout: int, dropout rate (Default: 0.25)
    '''

    out = Flatten()(input_data)
    out = Dense(last_dense_nodes)(out)
    out = LeakyReLU()(out)
    out = Dropout(dropout)(out)
    out = Dense(self.num_freq_bins)(out)

    return out

  def call(self, x):

    for _ in range(self.num_conv_blocks):
      x = self.conv_block(x, 16, 3, 0.25)

    x = self.bottleneck(x)

    return x
