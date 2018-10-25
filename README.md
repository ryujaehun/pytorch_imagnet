# pytorch_imagnet_converter
You can use imagenet in pytorch using the weights learned in other frameworks.
This is simply an attempt to weight quantization


If you want more complete control, you can use MMdnn.

## install

```
pip install mmdnn
pip install torch
pip install torchvision
```

## usage 

```
python main.py --config config/mobilenet_v1_1.0.yaml -e
```


## references
[MMdnn](https://github.com/Microsoft/MMdnn)
[Xilinx/pytorch-quantization](https://github.com/Xilinx/pytorch-quantization)
