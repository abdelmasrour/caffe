import caffe 
import numpy as np 



proto = '/opt/caffe/inference_fpga/float.prototxt'
weights = '/opt/caffe/inference_fpga/float.caffemodel'
custom  = '/opt/caffe/inference_fpga/custom.prototxt'
custom_loss  = '/opt/caffe/inference_fpga/custom_loss.prototxt'
custom_conv_sigmoid  = '/opt/caffe/inference_fpga/custom_conv_hard_sigmoid.prototxt'

# Load the network architecture
net = caffe.Net(custom_conv_sigmoid, caffe.TEST)

# # Load the trained weights
net.copy_from(weights)

# layer_name = 'fc1'  # Replace with the name of the layer you want to retrieve
# Get a list of all supported layers
# supported_layers = list(net.layers.keys())
# print(supported_layers)
# import caffe

# # Get a list of all layer types available in caffe.layers
# layer_types = [name for name in caffe.layers if isinstance(getattr(caffe.layers, name), name)]

# # Print the list of layer types
# for layer_type in layer_types:
#     print(layer_type)
