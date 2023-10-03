import unittest
import tempfile
import os
import six

import caffe


class HardSigmoid(caffe.Layer):
    """A layer that apply the fonction sigmoid = max(0,min(1,alpha*x+beta))"""

    def setup(self, bottom, top):
        print("#########################################")
        #print(len(bottom[0]),len(bottom))
        print(bottom[0].count,"count")
        print(bottom[0].channels,"channels")
        print(bottom[0].data,"data")
        print("#########################################")
        try:
          #  self.alpha = float(self.alpha_str)
            self.alpha,self.beta  = float(self.param_str.split()[0]),float(self.param_str.split()[1])
        except ValueError:
            raise ValueError("param_alpha and param_beta  must be a legible float")
        assert len(bottom)==len(top)
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = max(0,min(1,self.alpha * bottom[0].data + self.beta))

    def backward(self, top, propagate_down, bottom):
        #bottom[0].diff[...] = self.beta * top[0].diff
        # pas besoin de back propagation ni (train ) ni QAT envisage
        pass
class Resize(caffe.Layer):
    """A layer that apply the fonction sigmoid = max(0,min(1,alpha*x+beta))"""

    def setup(self, bottom, top):
        print("#########################################")
        #print(len(bottom[0]),len(bottom))
        print("#########################################")
        try:
          #  self.alpha = float(self.alpha_str)
            self.alpha,self.beta  = float(self.param_str.split()[0]),float(self.param_str.split()[1])
        except ValueError:
            raise ValueError("param_alpha and param_beta  must be a legible float")
        assert len(bottom)==len(top)
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = max(0,min(1,self.alpha * bottom[0].data + self.beta))

    def backward(self, top, propagate_down, bottom):
        #bottom[0].diff[...] = self.beta * top[0].diff
        # pas besoin de back propagation ni (train ) ni QAT envisage
        pass


def python_param_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'mul10' bottom: 'data' top: 'mul10'
          python_param { module: 'test_python_layer_with_param_str'
                layer: 'HardSigmoid' param_str: '0.2 0.5' } }
        layer { type: 'Python' name: 'mul2' bottom: 'mul10' top: 'mul2'
          python_param { module: 'test_python_layer_with_param_str'
                layer: 'HardSigmoid' param_str: '0.2 0.5' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        x = 8
        self.net.blobs['data'].data[...] = x
        self.net.forward()
        for y in self.net.blobs['mul2'].data.flat:
            self.assertEqual(y, 2 * 10 * x)

    def test_backward(self):
        x = 7
        self.net.blobs['mul2'].diff[...] = x
        self.net.backward()
        for y in self.net.blobs['data'].diff.flat:
            self.assertEqual(y, 2 * 10 * x)