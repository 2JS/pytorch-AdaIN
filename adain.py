import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import coremltools as ct

import net
from function import adaptive_instance_normalization, coral

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load('models/decoder.pth'))
vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])

enc_input = torch.rand(1, 3, 256, 256)
dec_input = torch.rand(1, 512, 32, 32)
traced_vgg = torch.jit.trace(vgg, enc_input)
converted_vgg  = ct.convert(
    vgg,
    inputs = ct.TensorType(shape=enc_input.size())
)
traced_decoder = torch.jit.trace(decoder, enc_input)
converted_decoder = ct.convert(
    decoder,
    inputs = ct.TensorType(shape=dec_input.size())
)
converted_decoder.save("adain_dec.mlmodel")
converted_vgg.save("adain_vgg.mlmodel")
