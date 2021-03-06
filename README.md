# Dynamic Filter Layers for Keras

Apply a different kernel (or matrix operation) to each datapoint in a mini batch. 

## How to install

```
git clone https://github.com/iranroman/dynfilt_layers.git

cd dynfilt_layers

pip install -e .
```

## Example with dynfilt_layers.Conv2D

```
import dynfilt_layers
import numpy as np

# the input data
# here random numbers just as a dummy example
X = np.random.randn(
		batch_size,
		256,
		256,
		nchans_in
		)

# define a kernel (could be, for example, based on 
# metadata associated with each datapoint,
# or the output of another model)
# here random numbers just as a dummy example
kernel_values = np.random.randn(
				batch_size,
				kernel_height, 
				kernel_width, 
				nchans_in,
				nchans_out
				)

# initilize the dynfilt_layers.Conv2D layer
dyn_conv = dynfilt_layers.Conv2D(
				padding=padding
				)

# convolve with dynfilt
dyn_conv(X, kernel_values)
```

WARNING: this Conv2D layer does not explicitly apply a bias or an activation function to the output of each channel. If you need to apply an activation function and/or biases, do not forget to add those!
