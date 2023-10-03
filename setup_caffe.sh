pip install --upgrade "pip < 21.0"
cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. &&\
mkdir build && cd build && \
cmake -DCPU_ONLY=1 .. && \
WITH_PYTHON_LAYER=1 make && make pycaffe
#make -j"$(nproc)"

export PYCAFFE_ROOT=$CAFFE_ROOT/python
export PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
export PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
