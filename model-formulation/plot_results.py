import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, median

job_size_3_preds = {'efficientnet_b6-16-mobilenet_v3_small-8-vgg19-4': 1,
 'alexnet-2-resnet101-4-vgg13-16': 0,
 'densenet161-8-efficientnet_b5-8-inception_v3-32': 1,
 'densenet121-16-vgg11-8-vgg11-8': 1,
 'alexnet-2-densenet121-32-vgg16-16': 0,
 'densenet201-16-densenet201-8-vgg11-16': 1,
 'inception_v3-16-inception_v3-8-vgg16-16': 0,
 'efficientnet_b6-8-resnet34-8-squeezenet1_1-4': 0,
 'efficientnet_b6-4-resnet50-16-vgg13-16': 1,
 'densenet169-2-resnet18-16-vgg13-8': 1,
 'efficientnet_b5-4-vgg13-32-vgg16-16': 1,
 'densenet169-32-squeezenet1_0-2-squeezenet1_0-2': 0,
 'inception_v3-8-mobilenet_v3_large-8-resnet18-2': 0,
 'efficientnet_b7-32-mobilenet_v3_small-2-vgg19-16': 0,
 'squeezenet1_0-32-alexnet-4-mobilenet_v2-4': 0,
 'efficientnet_b7-2-resnet101-2-resnet101-32': 1,
 'resnet18-32-resnet50-16-resnet50-8': 0,
 'densenet161-4-mobilenet_v2-16-vgg11-2': 1,
 'vgg19-4-efficientnet_b6-16-resnet18-8': 0,
 'alexnet-2-mobilenet_v3_small-16-resnet152-4': 1,
 'efficientnet_b5-8-resnet152-8-vgg11-4': 1,
 'mobilenet_v2-8-mobilenet_v3_small-8-vgg19-32': 0,
 'mobilenet_v3_small-16-resnet50-4-vgg19-32': 0,
 'mobilenet_v2-32-squeezenet1_0-4-vgg19-8': 0,
 'alexnet-16-densenet121-8-mobilenet_v3_small-4': 0,
 'densenet201-4-inception_v3-4-squeezenet1_0-2': 0,
 'alexnet-16-resnet18-8-squeezenet1_1-8': 0,
 'densenet169-2-resnet18-4-vgg16-2': 1,
 'efficientnet_b5-4-efficientnet_b6-16-vgg11-2': 1,
 'inception_v3-2-mobilenet_v3_small-4-resnet152-8': 1,
 'densenet121-16-resnet152-2-squeezenet1_1-32': 1,
 'efficientnet_b6-16-efficientnet_b7-2-vgg13-2': 1,
 'efficientnet_b5-32-inception_v3-8-resnet18-4': 0,
 'resnet152-16-squeezenet1_1-8-vgg13-4': 1,
 'densenet121-16-efficientnet_b6-4-efficientnet_b7-32': 1,
 'efficientnet_b7-32-mobilenet_v3_large-4-squeezenet1_0-8': 0,
 'resnet50-8-efficientnet_b7-2-vgg13-8': 0,
 'efficientnet_b7-16-resnet152-2-resnet152-8': 0,
 'densenet121-32-mobilenet_v3_large-16-vgg16-2': 1,
 'alexnet-16-inception_v3-2-vgg13-8': 0,
 'efficientnet_b7-16-mobilenet_v2-4-resnet152-16': 0,
 'efficientnet_b7-16-inception_v3-16-resnet101-2': 0,
 'resnet152-4-resnet34-2-vgg19-16': 0,
 'densenet201-4-mobilenet_v2-4-mobilenet_v3_large-4': 0,
 'densenet161-16-efficientnet_b5-2-vgg19-4': 1,
 'densenet169-2-efficientnet_b5-2-squeezenet1_0-2': 1,
 'resnet152-16-resnet34-32-vgg19-2': 1,
 'efficientnet_b6-8-mobilenet_v3_small-16-squeezenet1_0-16': 1,
 'densenet201-16-resnet34-32-vgg13-16': 1,
 'densenet169-32-efficientnet_b7-16-resnet34-8': 0,
 'resnet152-2-vgg19-2-vgg19-2': 1,
 'densenet121-2-mobilenet_v3_small-8-vgg11-2': 1,
 'efficientnet_b7-16-mobilenet_v3_large-32-vgg11-16': 0,
 'densenet121-8-vgg11-16-vgg19-4': 1,
 'densenet169-32-inception_v3-2-squeezenet1_1-16': 0,
 'resnet34-2-vgg16-32-vgg16-8': 0,
 'densenet169-32-efficientnet_b7-2-vgg13-2': 1,
 'efficientnet_b6-2-resnet101-2-squeezenet1_0-32': 1,
 'densenet169-2-inception_v3-16-resnet34-4': 1,
 'efficientnet_b6-4-mobilenet_v2-2-squeezenet1_0-16': 1,
 'densenet201-16-inception_v3-16-squeezenet1_1-4': 0,
 'efficientnet_b6-2-mobilenet_v2-4-vgg13-8': 1,
 'densenet121-16-densenet161-16-densenet201-32': 1,
 'densenet169-16-mobilenet_v3_small-4-vgg19-4': 1,
 'resnet101-32-efficientnet_b5-4-resnet101-4': 0,
 'vgg19-4-resnet101-2-vgg11-2': 0,
 'efficientnet_b6-32-resnet50-16-vgg16-4': 1,
 'densenet121-32-efficientnet_b6-2-efficientnet_b7-32': 0,
 'densenet161-32-resnet50-2-vgg19-16': 0,
 'inception_v3-16-mobilenet_v2-2-resnet50-8': 1,
 'resnet152-8-resnet34-4-vgg16-32': 0,
 'inception_v3-32-mobilenet_v3_large-16-vgg16-32': 0,
 'densenet169-2-efficientnet_b7-4-resnet34-32': 1,
 'inception_v3-2-vgg11-16-vgg16-32': 0,
 'efficientnet_b5-4-efficientnet_b5-8-mobilenet_v3_small-8': 1,
 'efficientnet_b5-16-resnet34-32-vgg19-16': 0,
 'densenet201-8-resnet50-8-vgg11-2': 1,
 'densenet121-16-vgg19-16-vgg19-2': 1,
 'densenet161-32-efficientnet_b6-2-vgg19-2': 1,
 'densenet121-2-mobilenet_v2-32-mobilenet_v3_large-2': 1,
 'efficientnet_b5-2-vgg13-32-vgg19-2': 1,
 'mobilenet_v3_large-8-resnet152-4-resnet152-8': 1,
 'efficientnet_b6-32-mobilenet_v2-32-mobilenet_v3_large-16': 0,
 'densenet169-8-efficientnet_b7-16-resnet18-8': 1,
 'densenet121-4-squeezenet1_1-4-vgg19-2': 1,
 'alexnet-8-efficientnet_b5-4-resnet34-2': 0,
 'alexnet-32-densenet169-2-efficientnet_b6-32': 1,
 'densenet161-4-mobilenet_v3_small-32-resnet152-16': 1,
 'squeezenet1_0-32-resnet152-8-squeezenet1_1-32': 0,
 'mobilenet_v3_small-2-squeezenet1_1-4-vgg16-2': 1,
 'densenet169-16-efficientnet_b5-32-vgg11-32': 1,
 'efficientnet_b5-16-efficientnet_b5-32-squeezenet1_0-8': 0,
 'resnet152-16-resnet152-2-vgg19-16': 0,
 'inception_v3-16-mobilenet_v2-4-mobilenet_v3_small-8': 0,
 'alexnet-8-efficientnet_b5-16-vgg11-4': 0,
 'inception_v3-32-inception_v3-16-resnet34-2': 0,
 'densenet169-2-mobilenet_v2-8-resnet34-8': 1,
 'densenet169-8-efficientnet_b7-8-resnet101-4': 0,
 'squeezenet1_1-2-vgg11-8-vgg19-32': 0,
 'densenet121-4-densenet169-4-vgg11-4': 1,
 'densenet161-8-efficientnet_b7-32-resnet152-8': 1,
 'efficientnet_b6-4-inception_v3-4-vgg19-16': 1,
 'alexnet-8-efficientnet_b6-4-squeezenet1_0-4': 0,
 'inception_v3-8-resnet18-32-vgg19-16': 0,
 'vgg16-4-densenet169-16-mobilenet_v2-8': 1,
 'mobilenet_v2-2-resnet18-2-vgg19-2': 1,
 'densenet121-8-resnet18-8-vgg16-2': 1,
 'densenet169-4-mobilenet_v2-8-resnet18-4': 1,
 'densenet121-16-mobilenet_v2-8-vgg19-16': 0,
 'vgg19-8-densenet201-8-resnet34-2': 1,
 'mobilenet_v2-4-mobilenet_v3_small-32-vgg16-2': 1,
 'mobilenet_v3_small-16-vgg16-2-vgg19-32': 0,
 'squeezenet1_1-16-squeezenet1_1-16-vgg13-4': 1,
 'densenet161-32-vgg11-32-vgg19-32': 0,
 'densenet201-8-resnet18-4-squeezenet1_0-32': 1,
 'densenet161-8-resnet101-32-resnet18-16': 1,
 'densenet169-16-densenet201-32-vgg19-2': 1,
 'vgg16-2-vgg19-16-vgg19-4': 1,
 'efficientnet_b7-4-mobilenet_v3_large-32-vgg16-16': 1,
 'alexnet-8-inception_v3-4-mobilenet_v3_large-8': 0,
 'resnet50-32-vgg11-2-vgg13-8': 1,
 'densenet169-4-resnet101-16-squeezenet1_0-4': 1,
 'squeezenet1_1-4-vgg13-4-vgg13-8': 0,
 'mobilenet_v3_small-32-resnet50-8-vgg13-16': 0,
 'densenet169-32-efficientnet_b6-8-mobilenet_v3_large-8': 0,
 'densenet121-8-densenet201-16-vgg16-4': 1,
 'densenet169-32-resnet152-8-vgg16-8': 0,
 'alexnet-32-densenet201-8-efficientnet_b5-32': 1,
 'densenet161-8-mobilenet_v3_small-32-resnet101-4': 0,
 'alexnet-4-squeezenet1_1-32-vgg13-32': 0,
 'densenet121-4-efficientnet_b7-4-vgg16-4': 1,
 'densenet169-8-efficientnet_b6-8-mobilenet_v3_small-32': 1,
 'densenet161-8-densenet169-4-efficientnet_b6-8': 0,
 'densenet201-8-resnet152-16-resnet34-16': 1,
 'resnet101-16-resnet152-16-squeezenet1_1-2': 0,
 'efficientnet_b6-2-squeezenet1_0-32-vgg19-2': 1,
 'densenet169-4-mobilenet_v2-8-squeezenet1_0-8': 1,
 'vgg13-2-densenet169-2-squeezenet1_1-8': 1,
 'densenet169-4-efficientnet_b7-8-squeezenet1_1-16': 1,
 'efficientnet_b7-4-squeezenet1_0-16-vgg19-16': 1,
 'efficientnet_b6-16-efficientnet_b6-2-inception_v3-4': 0,
 'inception_v3-32-resnet18-16-vgg16-8': 0,
 'inception_v3-4-mobilenet_v3_small-4-squeezenet1_1-4': 0,
 'resnet50-2-squeezenet1_0-8-vgg13-16': 1,
 'densenet169-16-mobilenet_v3_large-2-vgg16-32': 0,
 'squeezenet1_1-2-vgg16-4-vgg19-8': 1,
 'efficientnet_b7-4-vgg13-32-vgg16-4': 1,
 'vgg16-32-squeezenet1_0-32-vgg11-2': 0,
 'squeezenet1_0-4-vgg16-32-vgg16-4': 1,
 'mobilenet_v2-2-resnet152-2-vgg11-8': 1,
 'mobilenet_v3_small-16-squeezenet1_0-2-squeezenet1_1-16': 1,
 'densenet161-2-densenet201-16-efficientnet_b6-8': 1,
 'efficientnet_b5-8-resnet50-4-vgg13-8': 1,
 'densenet161-2-densenet201-4-efficientnet_b7-2': 1,
 'densenet121-4-resnet18-2-resnet18-8': 1,
 'densenet161-16-densenet201-4-vgg13-4': 1,
 'mobilenet_v3_large-16-mobilenet_v3_large-8-squeezenet1_1-2': 0,
 'alexnet-32-densenet161-32-mobilenet_v3_large-16': 0,
 'densenet121-8-mobilenet_v3_large-4-squeezenet1_1-16': 1,
 'alexnet-16-densenet121-2-mobilenet_v3_small-32': 1,
 'inception_v3-2-vgg11-4-vgg13-2': 1,
 'densenet121-4-efficientnet_b6-4-mobilenet_v3_large-32': 1,
 'resnet34-16-vgg16-2-vgg16-8': 1,
 'densenet161-8-densenet201-8-mobilenet_v3_small-8': 0,
 'efficientnet_b6-16-inception_v3-8-resnet18-32': 0,
 'efficientnet_b5-16-resnet101-4-vgg11-4': 1,
 'densenet121-8-inception_v3-16-resnet50-16': 1,
 'mobilenet_v3_large-4-vgg13-32-vgg19-16': 0,
 'densenet121-16-densenet169-8-mobilenet_v2-2': 0,
 'inception_v3-16-resnet18-8-vgg11-4': 1,
 'densenet201-16-efficientnet_b7-16-mobilenet_v3_large-8': 0,
 'densenet201-8-resnet101-8-resnet101-8': 1,
 'resnet18-2-resnet18-4-resnet50-8': 1,
 'mobilenet_v3_large-4-mobilenet_v3_small-4-resnet50-8': 1,
 'alexnet-8-mobilenet_v2-16-mobilenet_v3_small-2': 1,
 'mobilenet_v3_large-4-resnet101-2-vgg11-4': 1,
 'mobilenet_v3_large-4-resnet50-8-squeezenet1_1-4': 1,
 'densenet169-8-resnet34-8-vgg13-2': 1,
 'efficientnet_b7-8-inception_v3-4-vgg13-8': 1,
 'alexnet-16-mobilenet_v3_small-2-resnet152-2': 1,
 'vgg19-4-squeezenet1_0-8-squeezenet1_1-16': 1,
 'densenet161-8-resnet18-32-resnet50-4': 1,
 'densenet121-4-efficientnet_b6-4-vgg13-8': 1,
 'resnet18-32-resnet34-16-squeezenet1_0-2': 0,
 'efficientnet_b5-8-resnet50-2-vgg13-8': 1,
 'densenet169-4-densenet201-32-resnet152-2': 1,
 'mobilenet_v2-2-mobilenet_v2-32-resnet152-2': 1,
 'resnet34-16-resnet50-4-vgg11-32': 0,
 'alexnet-2-resnet152-16-vgg13-2': 1,
 'inception_v3-4-vgg16-8-vgg19-16': 0,
 'densenet121-16-resnet101-4-resnet18-8': 0,
 'efficientnet_b5-2-mobilenet_v3_small-8-resnet18-4': 1,
 'densenet121-2-resnet101-8-squeezenet1_0-16': 1,
 'densenet121-2-squeezenet1_1-4-vgg13-2': 1,
 'efficientnet_b7-2-resnet34-2-resnet50-8': 1,
 'densenet169-4-inception_v3-16-resnet152-32': 1,
 'densenet121-2-resnet101-4-vgg11-2': 1,
 'mobilenet_v3_large-32-squeezenet1_1-2-vgg13-16': 0,
 'densenet121-2-mobilenet_v3_large-32-squeezenet1_0-8': 1,
 'densenet121-4-efficientnet_b7-8-mobilenet_v3_large-32': 1,
 'mobilenet_v3_small-4-resnet18-4-squeezenet1_1-2': 0,
 'vgg16-4-alexnet-16-resnet18-32': 1,
 'densenet161-8-resnet101-16-vgg13-4': 1,
 'resnet101-8-squeezenet1_0-32-squeezenet1_0-4': 1,
 'densenet161-16-densenet169-32-densenet169-8': 1,
 'densenet161-2-densenet169-4-densenet201-2': 1,
 'efficientnet_b5-2-resnet101-16-resnet50-8': 1,
 'resnet18-32-squeezenet1_0-2-vgg16-16': 0,
 'vgg19-32-vgg19-4-inception_v3-8': 0,
 'densenet169-32-mobilenet_v3_large-16-vgg19-2': 1,
 'mobilenet_v3_large-8-mobilenet_v3_small-16-resnet18-4': 0,
 'densenet121-4-resnet101-32-resnet34-8': 1,
 'densenet201-4-resnet34-4-squeezenet1_0-2': 0,
 'efficientnet_b5-16-vgg11-16-vgg16-2': 1,
 'alexnet-4-densenet169-4-resnet50-16': 1,
 'densenet201-16-mobilenet_v3_small-32-squeezenet1_1-4': 0,
 'densenet169-32-resnet152-16-squeezenet1_1-2': 0,
 'resnet18-32-vgg11-16-vgg19-16': 0,
 'densenet201-32-mobilenet_v3_large-32-resnet152-16': 0,
 'mobilenet_v2-32-vgg16-16-vgg16-16': 0,
 'mobilenet_v2-8-resnet50-4-squeezenet1_0-16': 1,
 'densenet121-8-densenet201-32-densenet201-4': 1,
 'densenet121-32-densenet201-8-mobilenet_v3_small-4': 0,
 'densenet161-32-efficientnet_b5-8-vgg11-32': 1,
 'efficientnet_b6-32-densenet169-4-resnet50-16': 0,
 'densenet161-16-densenet201-2-efficientnet_b7-8': 0,
 'densenet201-2-efficientnet_b6-2-resnet101-2': 1,
 'densenet169-32-mobilenet_v3_small-4-vgg13-8': 0,
 'densenet169-4-inception_v3-16-vgg16-16': 1,
 'inception_v3-32-vgg11-32-vgg19-2': 1,
 'resnet152-16-inception_v3-16-vgg11-8': 0,
 'resnet101-16-densenet201-16-inception_v3-2': 0,
 'efficientnet_b6-2-resnet18-8-squeezenet1_0-4': 1,
 'densenet121-16-densenet201-2-vgg16-32': 1,
 'densenet169-8-resnet34-16-squeezenet1_0-2': 0,
 'alexnet-2-resnet152-16-resnet152-8': 1,
 'densenet121-32-mobilenet_v3_large-16-mobilenet_v3_large-2': 0,
 'efficientnet_b6-2-resnet50-16-squeezenet1_1-16': 1,
 'resnet152-4-resnet34-32-resnet34-32': 1,
 'efficientnet_b6-2-resnet18-2-vgg16-2': 1,
 'densenet121-2-inception_v3-16-vgg19-4': 1,
 'resnet50-8-vgg16-4-vgg19-8': 1,
 'densenet121-8-resnet152-8-squeezenet1_0-32': 1,
 'densenet201-2-inception_v3-8-resnet101-2': 1,
 'mobilenet_v3_small-32-mobilenet_v3_small-32-vgg19-16': 0,
 'mobilenet_v3_small-2-resnet18-32-vgg11-8': 1,
 'mobilenet_v3_large-16-squeezenet1_1-32-vgg19-8': 0,
 'vgg16-8-densenet169-16-resnet101-8': 1,
 'resnet34-32-resnet34-32-squeezenet1_0-16': 0,
 'densenet121-2-efficientnet_b5-8-vgg19-2': 1,
 'resnet18-32-vgg11-4-vgg16-32': 0,
 'resnet18-4-squeezenet1_0-32-squeezenet1_0-8': 1,
 'mobilenet_v3_large-32-mobilenet_v3_large-4-resnet50-8': 0,
 'mobilenet_v2-2-mobilenet_v2-4-resnet152-2': 1,
 'densenet121-2-densenet201-4-vgg13-4': 1,
 'resnet101-16-resnet101-8-vgg13-16': 0,
 'densenet121-8-squeezenet1_1-4-vgg19-16': 0,
 'densenet169-32-inception_v3-32-resnet50-16': 0,
 'resnet101-8-resnet18-4-vgg13-2': 1,
 'densenet121-2-mobilenet_v2-8-vgg13-32': 1,
 'mobilenet_v2-32-resnet101-32-vgg19-32': 0,
 'squeezenet1_1-32-resnet18-8-resnet34-2': 0,
 'efficientnet_b6-8-resnet18-32-resnet34-16': 1,
 'alexnet-8-resnet152-2-resnet152-8': 1,
 'mobilenet_v3_large-2-vgg11-32-vgg16-2': 1,
 'efficientnet_b7-16-squeezenet1_1-32-vgg16-2': 1,
 'efficientnet_b5-8-mobilenet_v2-16-vgg19-2': 1,
 'mobilenet_v2-16-resnet34-16-resnet50-4': 1,
 'mobilenet_v2-8-resnet34-8-vgg16-2': 1,
 'densenet121-32-efficientnet_b7-4-efficientnet_b7-8': 0,
 'efficientnet_b7-2-resnet152-32-resnet34-4': 1,
 'squeezenet1_1-4-squeezenet1_1-8-vgg19-16': 0,
 'densenet161-16-efficientnet_b6-2-squeezenet1_0-16': 0,
 'densenet201-2-inception_v3-8-mobilenet_v3_small-2': 1,
 'alexnet-2-densenet161-2-efficientnet_b6-16': 1,
 'efficientnet_b5-2-mobilenet_v3_large-32-resnet101-8': 1,
 'inception_v3-8-resnet101-4-vgg13-8': 0,
 'densenet121-32-efficientnet_b6-32-resnet152-4': 0,
 'resnet34-16-resnet50-8-squeezenet1_1-32': 1,
 'mobilenet_v3_large-8-resnet18-16-vgg19-32': 0,
 'efficientnet_b7-16-mobilenet_v3_large-16-mobilenet_v3_small-8': 0,
 'efficientnet_b6-32-resnet152-16-resnet18-8': 0,
 'resnet34-16-vgg19-2-vgg19-8': 1,
 'resnet101-32-resnet34-8-vgg19-2': 1,
 'resnet152-32-efficientnet_b7-2-vgg16-8': 0,
 'vgg19-2-mobilenet_v3_small-4-resnet34-4': 1,
 'inception_v3-8-resnet152-8-vgg16-32': 0,
 'densenet169-16-efficientnet_b5-16-resnet152-4': 1,
 'alexnet-32-mobilenet_v3_small-32-squeezenet1_1-4': 0,
 'resnet34-32-densenet201-4-squeezenet1_1-8': 0,
 'mobilenet_v2-4-mobilenet_v3_large-32-resnet101-2': 1,
 'densenet169-16-resnet18-16-vgg16-32': 0,
 'inception_v3-4-resnet50-4-vgg19-32': 1,
 'densenet161-16-densenet169-2-efficientnet_b6-8': 0,
 'inception_v3-2-mobilenet_v3_large-32-vgg16-2': 1,
 'inception_v3-32-mobilenet_v3_small-4-vgg16-8': 0,
 'densenet201-4-inception_v3-2-squeezenet1_1-32': 1,
 'densenet121-2-efficientnet_b6-8-resnet101-4': 1,
 'efficientnet_b7-4-squeezenet1_0-8-vgg13-4': 1,
 'alexnet-32-densenet201-16-resnet101-16': 1,
 'densenet161-8-mobilenet_v3_large-32-squeezenet1_0-4': 1,
 'resnet101-8-vgg19-4-vgg19-4': 1,
 'densenet201-8-mobilenet_v3_small-16-squeezenet1_0-2': 0,
 'inception_v3-8-vgg13-2-vgg13-2': 1,
 'efficientnet_b6-8-densenet161-4-resnet152-8': 1,
 'inception_v3-2-inception_v3-2-vgg16-2': 1,
 'densenet161-4-resnet152-32-squeezenet1_0-2': 0,
 'densenet201-2-vgg11-2-vgg11-8': 1,
 'efficientnet_b5-4-efficientnet_b6-16-resnet34-16': 1,
 'efficientnet_b6-2-resnet18-32-resnet50-16': 1,
 'alexnet-8-efficientnet_b5-4-vgg16-2': 1,
 'efficientnet_b5-32-mobilenet_v3_large-2-resnet50-8': 0,
 'vgg19-32-densenet201-2-vgg11-8': 0,
 'densenet169-2-inception_v3-32-vgg16-8': 1,
 'densenet161-4-mobilenet_v3_small-8-vgg19-4': 1,
 'efficientnet_b6-4-resnet50-4-squeezenet1_1-32': 1,
 'resnet101-4-squeezenet1_0-2-vgg16-32': 0,
 'efficientnet_b5-32-squeezenet1_0-32-vgg19-8': 0,
 'densenet121-16-mobilenet_v3_small-8-vgg11-16': 0,
 'mobilenet_v3_large-4-mobilenet_v3_small-4-squeezenet1_1-8': 1,
 'mobilenet_v2-32-resnet34-2-vgg19-32': 0,
 'mobilenet_v2-16-mobilenet_v2-8-mobilenet_v3_large-32': 1,
 'densenet161-8-densenet161-8-densenet201-2': 1,
 'squeezenet1_1-2-vgg11-4-vgg19-2': 1,
 'alexnet-2-efficientnet_b6-8-vgg16-2': 1,
 'densenet169-2-resnet50-8-squeezenet1_0-8': 1,
 'densenet161-2-efficientnet_b6-8-resnet50-8': 1,
 'alexnet-8-mobilenet_v3_large-8-vgg16-2': 0,
 'inception_v3-8-resnet101-8-vgg13-16': 0,
 'efficientnet_b6-32-resnet18-4-vgg13-16': 0,
 'efficientnet_b5-16-mobilenet_v3_small-4-squeezenet1_0-32': 0,
 'mobilenet_v2-16-mobilenet_v3_large-4-mobilenet_v3_small-2': 0,
 'densenet169-8-efficientnet_b6-4-mobilenet_v3_small-2': 0,
 'inception_v3-32-squeezenet1_0-2-vgg19-16': 0,
 'alexnet-2-efficientnet_b5-8-resnet152-8': 1,
 'resnet101-16-vgg11-2-vgg19-32': 0,
 'densenet161-8-densenet201-16-densenet201-2': 1,
 'squeezenet1_0-2-squeezenet1_1-2-vgg16-2': 0,
 'efficientnet_b7-4-resnet18-2-vgg13-2': 0,
 'inception_v3-32-densenet201-16-resnet101-4': 0,
 'alexnet-32-mobilenet_v3_large-4-resnet18-4': 0,
 'mobilenet_v3_large-8-vgg11-32-vgg16-4': 1,
 'densenet201-16-mobilenet_v3_small-4-vgg16-8': 0,
 'resnet101-16-resnet34-8-squeezenet1_1-32': 1,
 'efficientnet_b6-8-mobilenet_v3_large-2-vgg13-4': 1,
 'alexnet-4-resnet34-16-vgg19-2': 1,
 'inception_v3-32-mobilenet_v2-2-resnet18-16': 0,
 'efficientnet_b7-16-mobilenet_v2-4-resnet50-8': 0,
 'densenet161-4-mobilenet_v3_large-2-squeezenet1_0-8': 1,
 'efficientnet_b7-8-mobilenet_v2-32-resnet50-2': 1,
 'efficientnet_b6-32-resnet18-32-resnet34-32': 0,
 'densenet201-2-efficientnet_b6-8-vgg13-16': 1,
 'resnet152-2-resnet34-16-resnet50-2': 1,
 'alexnet-32-efficientnet_b6-16-resnet50-32': 0,
 'densenet161-2-resnet152-2-resnet34-32': 1,
 'efficientnet_b6-4-resnet50-4-vgg11-2': 1,
 'efficientnet_b5-2-mobilenet_v3_small-32-resnet50-8': 1,
 'inception_v3-8-mobilenet_v3_large-8-resnet152-16': 1,
 'resnet101-32-squeezenet1_1-32-vgg16-32': 0,
 'resnet101-2-resnet152-8-resnet50-4': 1,
 'densenet169-8-densenet201-2-mobilenet_v2-16': 1,
 'efficientnet_b5-8-inception_v3-8-mobilenet_v3_large-16': 1,
 'mobilenet_v2-4-resnet50-8-squeezenet1_1-16': 1,
 'densenet201-16-mobilenet_v3_small-16-resnet101-2': 0,
 'densenet161-4-resnet101-16-resnet50-16': 1,
 'densenet201-16-squeezenet1_1-4-vgg13-32': 0,
 'densenet121-32-efficientnet_b6-16-mobilenet_v3_large-16': 0,
 'mobilenet_v2-2-resnet152-32-vgg11-16': 0,
 'mobilenet_v3_large-4-resnet18-2-vgg19-8': 0,
 'vgg16-2-mobilenet_v3_large-16-resnet18-2': 1,
 'alexnet-4-alexnet-4-densenet201-16': 1,
 'densenet169-16-densenet201-2-inception_v3-32': 1,
 'alexnet-32-efficientnet_b5-16-squeezenet1_1-2': 0,
 'squeezenet1_1-32-vgg11-8-vgg19-8': 0,
 'inception_v3-2-mobilenet_v3_large-16-vgg19-32': 1,
 'resnet18-16-squeezenet1_0-16-vgg16-4': 1,
 'inception_v3-4-squeezenet1_0-4-vgg19-4': 1,
 'vgg19-32-resnet18-2-squeezenet1_1-8': 0,
 'densenet169-8-mobilenet_v3_large-2-resnet50-32': 1,
 'densenet161-8-vgg11-32-vgg13-2': 1,
 'efficientnet_b7-2-vgg11-4-vgg19-8': 1,
 'densenet121-4-mobilenet_v3_large-16-resnet101-8': 1,
 'resnet101-4-resnet152-2-resnet50-8': 1,
 'mobilenet_v3_large-4-resnet101-32-vgg13-32': 0,
 'resnet101-4-densenet201-2-mobilenet_v3_large-2': 0,
 'efficientnet_b6-2-mobilenet_v2-2-resnet50-4': 1,
 'densenet121-8-mobilenet_v3_small-8-vgg11-8': 0,
 'densenet121-32-resnet152-8-squeezenet1_0-2': 0,
 'inception_v3-16-inception_v3-32-mobilenet_v3_large-2': 0,
 'densenet161-8-resnet152-8-resnet18-8': 1,
 'alexnet-8-resnet152-32-squeezenet1_0-2': 0,
 'efficientnet_b6-32-efficientnet_b6-2-squeezenet1_0-16': 0,
 'densenet169-16-efficientnet_b6-2-vgg13-16': 1,
 'densenet201-4-efficientnet_b7-8-vgg13-4': 1,
 'alexnet-16-mobilenet_v3_large-8-resnet18-2': 0,
 'densenet121-4-resnet152-2-resnet34-8': 1,
 'squeezenet1_0-16-squeezenet1_1-16-squeezenet1_1-8': 0,
 'mobilenet_v3_small-32-resnet152-32-vgg11-16': 0,
 'densenet169-8-densenet201-4-squeezenet1_0-4': 0,
 'densenet169-16-mobilenet_v3_large-2-resnet34-8': 0,
 'alexnet-8-resnet50-16-squeezenet1_0-4': 0,
 'densenet161-2-densenet169-4-resnet152-16': 1,
 'densenet169-2-efficientnet_b6-8-resnet34-8': 1,
 'efficientnet_b6-8-mobilenet_v3_small-16-vgg13-4': 1,
 'efficientnet_b5-16-mobilenet_v3_large-2-resnet18-32': 1,
 'densenet121-8-densenet161-4-squeezenet1_1-32': 1,
 'efficientnet_b5-32-mobilenet_v2-16-resnet18-4': 0,
 'densenet169-4-efficientnet_b7-32-efficientnet_b7-4': 1,
 'densenet161-32-efficientnet_b5-4-vgg16-2': 1,
 'densenet121-4-squeezenet1_0-16-vgg13-4': 1,
 'alexnet-16-efficientnet_b5-4-mobilenet_v3_large-8': 1,
 'densenet121-2-densenet169-8-efficientnet_b7-32': 1,
 'densenet161-2-efficientnet_b7-4-vgg19-16': 1,
 'densenet169-2-efficientnet_b5-4-resnet101-16': 1,
 'alexnet-32-efficientnet_b6-2-resnet18-2': 1,
 'efficientnet_b6-16-mobilenet_v2-2-vgg16-8': 0,
 'mobilenet_v3_large-8-resnet34-8-squeezenet1_0-4': 0,
 'resnet18-32-vgg11-16-vgg16-8': 0,
 'mobilenet_v3_large-2-vgg11-16-vgg13-32': 0,
 'densenet169-2-inception_v3-8-squeezenet1_1-32': 1,
 'efficientnet_b5-4-efficientnet_b6-8-vgg19-2': 1,
 'inception_v3-8-resnet101-8-squeezenet1_0-8': 1,
 'mobilenet_v3_small-4-resnet50-4-vgg13-32': 0,
 'mobilenet_v3_small-16-resnet34-32-vgg19-32': 0,
 'efficientnet_b7-16-mobilenet_v2-8-vgg19-16': 0,
 'densenet201-8-densenet201-8-resnet34-8': 0,
 'alexnet-8-efficientnet_b6-32-squeezenet1_1-4': 0,
 'densenet169-8-efficientnet_b5-4-resnet101-2': 0,
 'efficientnet_b7-16-mobilenet_v3_large-32-mobilenet_v3_small-2': 0,
 'densenet169-2-vgg11-16-vgg13-8': 1,
 'densenet201-32-mobilenet_v2-32-resnet101-32': 0,
 'efficientnet_b7-8-resnet18-16-vgg13-32': 0,
 'alexnet-8-resnet152-8-vgg13-8': 0,
 'alexnet-8-densenet121-32-vgg19-4': 1,
 'efficientnet_b7-2-inception_v3-4-resnet34-32': 1,
 'squeezenet1_1-8-vgg11-32-vgg19-4': 1,
 'densenet169-4-inception_v3-4-resnet34-8': 1,
 'vgg13-32-vgg11-2-vgg16-2': 1,
 'densenet201-4-inception_v3-16-squeezenet1_1-32': 1,
 'densenet201-4-resnet50-8-squeezenet1_0-16': 1,
 'densenet161-32-mobilenet_v3_large-32-resnet101-16': 0,
 'alexnet-8-efficientnet_b6-8-mobilenet_v3_large-8': 1,
 'efficientnet_b5-8-resnet50-2-squeezenet1_1-8': 0,
 'resnet101-16-resnet18-16-vgg13-2': 1,
 'efficientnet_b6-4-squeezenet1_0-32-vgg13-4': 1,
 'densenet161-32-efficientnet_b6-2-mobilenet_v3_small-2': 0,
 'densenet161-2-densenet169-4-resnet101-8': 1,
 'efficientnet_b6-2-mobilenet_v3_large-16-vgg16-16': 1,
 'efficientnet_b7-32-mobilenet_v3_large-4-vgg11-8': 0,
 'mobilenet_v2-32-densenet121-8-resnet18-4': 0,
 'densenet169-32-inception_v3-2-mobilenet_v3_large-16': 0,
 'densenet169-4-resnet152-16-resnet18-8': 1,
 'densenet169-32-resnet101-8-vgg16-4': 1,
 'densenet121-4-densenet169-32-vgg13-8': 1,
 'densenet161-32-mobilenet_v3_large-4-vgg11-8': 0,
 'densenet161-2-resnet152-4-vgg11-4': 1,
 'densenet121-16-efficientnet_b7-16-mobilenet_v2-4': 0,
 'alexnet-16-alexnet-4-mobilenet_v3_small-8': 0,
 'resnet18-32-resnet34-2-vgg13-4': 1,
 'inception_v3-16-mobilenet_v2-4-squeezenet1_0-16': 1,
 'densenet161-2-resnet18-4-squeezenet1_0-4': 1,
 'resnet34-16-resnet50-8-vgg11-16': 0,
 'inception_v3-8-resnet152-4-resnet50-8': 1,
 'mobilenet_v2-8-resnet18-32-vgg11-8': 0,
 'efficientnet_b5-16-mobilenet_v3_small-8-resnet34-2': 0,
 'vgg19-2-densenet161-2-densenet169-4': 1,
 'densenet121-32-resnet18-16-squeezenet1_0-16': 0,
 'alexnet-4-efficientnet_b6-32-resnet101-4': 0,
 'densenet201-4-efficientnet_b5-2-inception_v3-4': 0,
 'efficientnet_b7-4-mobilenet_v3_small-4-squeezenet1_1-4': 0,
 'squeezenet1_1-8-vgg11-8-vgg13-16': 0,
 'inception_v3-2-resnet152-2-vgg13-8': 1,
 'resnet18-8-resnet34-16-vgg16-8': 0,
 'efficientnet_b5-8-vgg13-4-vgg19-8': 1,
 'densenet161-2-densenet201-2-resnet18-4': 1,
 'vgg16-2-efficientnet_b6-2-squeezenet1_1-4': 1,
 'densenet121-8-mobilenet_v3_large-32-squeezenet1_0-8': 1,
 'efficientnet_b5-16-efficientnet_b7-16-mobilenet_v3_small-32': 0,
 'densenet201-8-efficientnet_b6-2-mobilenet_v2-32': 1,
 'squeezenet1_0-16-efficientnet_b5-4-squeezenet1_0-4': 0,
 'alexnet-2-mobilenet_v3_small-32-squeezenet1_1-8': 1,
 'densenet161-4-mobilenet_v3_large-32-vgg19-32': 1,
 'densenet161-8-resnet101-4-squeezenet1_1-2': 0,
 'squeezenet1_0-32-vgg11-16-vgg19-8': 0,
 'inception_v3-32-resnet152-32-resnet152-32': 0,
 'densenet121-8-densenet169-16-resnet50-16': 1,
 'densenet161-32-efficientnet_b5-32-efficientnet_b6-2': 0,
 'inception_v3-16-efficientnet_b7-8-resnet101-8': 1,
 'efficientnet_b7-4-efficientnet_b7-4-vgg11-2': 1,
 'inception_v3-8-squeezenet1_1-32-vgg11-16': 1,
 'inception_v3-4-resnet34-2-squeezenet1_0-8': 1,
 'densenet121-2-densenet161-16-vgg11-16': 1,
 'densenet201-32-mobilenet_v2-16-resnet50-8': 0,
 'densenet121-16-efficientnet_b6-32-resnet101-32': 1,
 'mobilenet_v3_small-32-resnet152-32-squeezenet1_1-2': 0,
 'resnet34-16-resnet34-16-resnet34-8': 1,
 'densenet121-2-densenet161-8-resnet50-16': 1,
 'vgg11-32-vgg13-2-vgg19-2': 1,
 'mobilenet_v3_small-16-squeezenet1_0-16-vgg19-8': 0,
 'resnet34-4-squeezenet1_1-8-vgg16-16': 0,
 'vgg13-4-densenet169-16-efficientnet_b6-4': 1,
 'efficientnet_b6-8-efficientnet_b7-4-squeezenet1_1-8': 0,
 'resnet18-32-squeezenet1_0-2-vgg11-4': 1,
 'efficientnet_b5-32-efficientnet_b6-32-inception_v3-4': 0,
 'densenet121-8-mobilenet_v3_large-16-vgg19-8': 1,
 'mobilenet_v2-4-mobilenet_v3_large-8-resnet152-16': 1,
 'resnet101-8-resnet18-32-squeezenet1_0-8': 1,
 'mobilenet_v3_large-32-alexnet-4-squeezenet1_1-8': 0,
 'efficientnet_b6-2-resnet152-4-squeezenet1_1-4': 1,
 'mobilenet_v3_small-2-resnet101-2-squeezenet1_0-8': 1,
 'densenet169-2-vgg13-16-vgg13-8': 1,
 'efficientnet_b7-2-resnet18-8-squeezenet1_0-8': 1,
 'densenet161-4-inception_v3-4-vgg19-16': 1,
 'efficientnet_b6-32-resnet152-16-vgg16-8': 0,
 'efficientnet_b5-32-efficientnet_b5-32-squeezenet1_0-4': 0,
 'vgg16-4-inception_v3-16-squeezenet1_1-8': 0,
 'densenet121-2-densenet161-2-resnet34-8': 1,
 'efficientnet_b5-2-resnet50-32-vgg19-32': 1,
 'resnet34-32-densenet161-2-vgg16-2': 1,
 'efficientnet_b7-16-mobilenet_v3_large-2-resnet18-16': 0,
 'densenet161-4-densenet169-4-squeezenet1_0-32': 1,
 'densenet161-8-densenet169-4-resnet152-4': 0,
 'alexnet-16-densenet169-2-vgg16-32': 0,
 'mobilenet_v2-8-resnet34-16-squeezenet1_0-16': 1,
 'vgg16-4-densenet169-4-resnet34-32': 1,
 'vgg11-4-vgg13-4-vgg19-16': 0,
 'alexnet-8-efficientnet_b5-2-resnet101-2': 1,
 'mobilenet_v2-4-mobilenet_v3_large-2-vgg11-4': 0,
 'efficientnet_b7-2-mobilenet_v3_large-8-vgg19-2': 1,
 'densenet169-16-resnet50-2-vgg19-2': 1,
 'densenet161-2-resnet101-4-resnet101-4': 1,
 'inception_v3-2-vgg13-2-vgg16-16': 1,
 'mobilenet_v3_large-16-mobilenet_v3_small-32-vgg16-32': 0,
 'efficientnet_b5-8-mobilenet_v2-2-vgg11-16': 0,
 'densenet121-4-densenet121-8-efficientnet_b5-16': 1,
 'densenet169-16-resnet152-2-resnet50-8': 1,
 'alexnet-32-densenet201-16-resnet18-16': 1,
 'densenet161-4-efficientnet_b5-4-vgg13-16': 1,
 'mobilenet_v3_large-2-mobilenet_v3_large-8-vgg11-2': 1,
 'densenet121-4-efficientnet_b7-8-mobilenet_v2-32': 1,
 'efficientnet_b5-2-resnet18-32-vgg16-4': 1,
 'densenet201-8-mobilenet_v3_large-16-vgg13-16': 1,
 'alexnet-8-efficientnet_b5-4-mobilenet_v2-4': 0,
 'mobilenet_v3_small-2-resnet50-16-squeezenet1_0-4': 1,
 'resnet101-16-densenet161-4-efficientnet_b5-4': 1,
 'densenet201-4-mobilenet_v3_large-16-squeezenet1_0-32': 1,
 'alexnet-8-inception_v3-2-resnet18-16': 1,
 'densenet121-32-mobilenet_v3_large-8-squeezenet1_0-16': 0,
 'mobilenet_v3_large-32-squeezenet1_0-16-vgg13-16': 0,
 'efficientnet_b5-32-efficientnet_b7-16-inception_v3-4': 0,
 'resnet101-4-resnet101-4-vgg11-16': 1,
 'alexnet-8-densenet169-16-efficientnet_b5-8': 1,
 'resnet101-4-resnet34-8-vgg11-32': 1,
 'resnet152-16-resnet152-16-vgg16-8': 0,
 'efficientnet_b5-2-inception_v3-32-vgg11-8': 1,
 'densenet161-16-efficientnet_b5-32-resnet18-4': 0,
 'alexnet-16-resnet34-8-vgg19-2': 1,
 'densenet201-8-efficientnet_b5-8-mobilenet_v3_large-8': 0,
 'densenet121-8-squeezenet1_1-8-squeezenet1_1-8': 0,
 'efficientnet_b5-8-squeezenet1_1-32-vgg19-32': 0,
 'resnet50-32-densenet201-4-resnet101-8': 0,
 'resnet152-8-resnet34-4-vgg13-2': 1,
 'densenet169-4-squeezenet1_0-32-squeezenet1_1-4': 1,
 'efficientnet_b6-32-vgg16-2-vgg19-16': 0,
 'mobilenet_v3_large-4-mobilenet_v3_large-4-mobilenet_v3_small-16': 1,
 'densenet169-16-densenet201-8-inception_v3-16': 1,
 'efficientnet_b7-16-resnet152-2-vgg13-16': 0,
 'densenet201-2-mobilenet_v2-32-resnet34-8': 1,
 'vgg11-16-alexnet-4-efficientnet_b6-2': 0,
 'densenet161-8-mobilenet_v2-8-mobilenet_v2-8': 1,
 'alexnet-4-resnet101-32-vgg16-32': 0,
 'densenet121-16-efficientnet_b5-4-squeezenet1_1-2': 0,
 'densenet201-32-efficientnet_b5-8-vgg19-16': 0,
 'resnet34-8-resnet50-4-vgg13-2': 1,
 'efficientnet_b6-8-inception_v3-32-squeezenet1_0-4': 0,
 'densenet161-4-efficientnet_b5-32-resnet18-2': 0,
 'efficientnet_b5-32-squeezenet1_0-16-vgg11-32': 0,
 'densenet201-4-resnet50-8-vgg16-4': 1,
 'efficientnet_b7-8-resnet101-8-vgg19-16': 1,
 'resnet34-16-vgg11-4-vgg13-16': 0,
 'densenet121-8-resnet18-16-resnet34-8': 1,
 'alexnet-32-resnet18-8-squeezenet1_1-2': 0,
 'efficientnet_b7-2-resnet34-16-resnet34-32': 1,
 'resnet152-32-resnet50-2-squeezenet1_0-4': 0,
 'densenet121-8-densenet161-2-vgg19-16': 0,
 'efficientnet_b6-8-mobilenet_v3_small-32-squeezenet1_0-8': 1,
 'efficientnet_b5-32-mobilenet_v3_small-4-resnet34-8': 0,
 'densenet161-4-vgg13-2-vgg16-16': 1,
 'mobilenet_v3_small-2-vgg16-4-vgg16-8': 1,
 'efficientnet_b6-16-resnet50-16-vgg13-4': 1,
 'densenet201-8-inception_v3-2-resnet34-16': 1,
 'efficientnet_b6-4-efficientnet_b6-8-vgg16-4': 1,
 'densenet121-4-squeezenet1_0-32-vgg13-32': 0,
 'inception_v3-4-squeezenet1_1-2-vgg19-4': 0,
 'alexnet-32-densenet161-16-resnet34-16': 1,
 'efficientnet_b7-4-resnet34-2-vgg11-8': 1,
 'densenet169-2-efficientnet_b7-8-vgg19-4': 1,
 'densenet169-32-resnet101-4-vgg13-32': 0,
 'efficientnet_b5-2-efficientnet_b7-2-resnet152-2': 1,
 'densenet169-16-inception_v3-8-resnet50-4': 0,
 'densenet169-32-resnet34-32-vgg19-2': 1,
 'densenet161-2-inception_v3-16-resnet50-2': 1,
 'densenet121-16-resnet101-2-squeezenet1_1-8': 0,
 'resnet152-16-squeezenet1_1-32-squeezenet1_1-8': 0,
 'densenet169-8-mobilenet_v2-4-resnet18-4': 0,
 'densenet169-8-efficientnet_b5-8-vgg11-4': 1,
 'densenet121-8-mobilenet_v3_small-16-squeezenet1_0-4': 0,
 'resnet50-16-resnet50-8-squeezenet1_1-32': 1,
 'mobilenet_v2-4-resnet18-8-vgg19-8': 0,
 'densenet121-4-densenet201-2-resnet34-2': 0,
 'densenet121-2-mobilenet_v3_small-32-squeezenet1_0-4': 1,
 'densenet201-32-inception_v3-2-resnet18-2': 0,
 'densenet161-32-densenet169-8-vgg16-4': 1,
 'mobilenet_v3_small-8-efficientnet_b7-2-inception_v3-4': 0,
 'efficientnet_b5-4-efficientnet_b7-4-resnet152-16': 1,
 'efficientnet_b5-16-mobilenet_v3_small-32-vgg11-32': 0,
 'efficientnet_b7-32-resnet50-16-squeezenet1_0-2': 0,
 'efficientnet_b5-4-efficientnet_b7-8-vgg11-2': 1,
 'densenet121-16-densenet201-4-efficientnet_b6-32': 1,
 'densenet169-2-resnet34-32-squeezenet1_0-16': 1,
 'densenet161-16-efficientnet_b5-8-mobilenet_v3_small-16': 0,
 'densenet161-32-mobilenet_v3_large-32-mobilenet_v3_large-32': 0,
 'densenet201-8-mobilenet_v2-2-squeezenet1_1-8': 0,
 'efficientnet_b6-4-efficientnet_b7-8-resnet50-2': 0,
 'densenet121-8-inception_v3-16-vgg11-32': 1,
 'densenet121-32-densenet169-32-inception_v3-4': 0,
 'resnet101-2-resnet18-2-vgg13-8': 1,
 'efficientnet_b6-8-vgg11-32-vgg16-4': 1,
 'densenet121-32-efficientnet_b6-32-vgg11-32': 0,
 'efficientnet_b6-2-inception_v3-32-resnet18-16': 1,
 'vgg19-32-densenet201-32-resnet34-4': 0,
 'squeezenet1_0-32-vgg11-8-vgg16-2': 1,
 'efficientnet_b6-16-efficientnet_b7-8-vgg13-16': 1,
 'inception_v3-16-mobilenet_v2-8-resnet34-16': 1,
 'densenet161-32-densenet169-16-efficientnet_b5-2': 0,
 'mobilenet_v2-32-densenet201-16-mobilenet_v3_small-32': 0,
 'resnet101-8-resnet50-8-squeezenet1_1-8': 0,
 'efficientnet_b5-32-resnet101-8-resnet50-16': 0,
 'alexnet-4-squeezenet1_0-8-vgg13-2': 1,
 'vgg11-32-vgg16-32-vgg19-16': 0,
 'efficientnet_b7-32-mobilenet_v3_large-2-vgg11-32': 0,
 'densenet169-16-efficientnet_b6-4-resnet50-16': 1,
 'efficientnet_b6-16-resnet50-16-squeezenet1_0-8': 0,
 'densenet169-4-mobilenet_v3_large-16-squeezenet1_0-8': 1,
 'densenet201-2-mobilenet_v2-4-mobilenet_v3_large-8': 1,
 'mobilenet_v3_large-2-resnet50-2-vgg13-16': 1,
 'densenet169-32-inception_v3-8-squeezenet1_0-8': 0,
 'resnet152-2-resnet18-16-vgg11-32': 1,
 'inception_v3-16-resnet18-2-vgg16-2': 1,
 'densenet169-16-densenet169-2-vgg11-4': 1,
 'efficientnet_b7-4-mobilenet_v3_large-16-vgg16-16': 1,
 'efficientnet_b7-8-mobilenet_v3_small-32-squeezenet1_1-16': 1,
 'mobilenet_v2-2-resnet18-2-vgg13-8': 0,
 'efficientnet_b6-32-resnet101-32-squeezenet1_0-2': 0,
 'alexnet-4-resnet18-16-vgg11-16': 0,
 'efficientnet_b5-2-inception_v3-2-mobilenet_v2-4': 1,
 'efficientnet_b6-2-mobilenet_v3_large-4-vgg16-8': 1,
 'mobilenet_v2-16-resnet152-4-vgg19-2': 1,
 'inception_v3-8-mobilenet_v3_large-16-vgg16-2': 1,
 'efficientnet_b6-8-mobilenet_v3_large-4-resnet34-2': 0,
 'alexnet-32-efficientnet_b6-8-resnet101-16': 1,
 'resnet152-4-resnet18-32-squeezenet1_0-16': 1,
 'resnet101-32-resnet50-8-squeezenet1_0-16': 0,
 'alexnet-16-alexnet-2-efficientnet_b6-16': 0,
 'alexnet-2-inception_v3-16-squeezenet1_1-4': 1,
 'mobilenet_v2-2-resnet34-8-vgg19-8': 1,
 'densenet169-8-vgg19-32-vgg19-4': 1,
 'densenet161-16-resnet34-8-vgg11-8': 0,
 'efficientnet_b6-4-inception_v3-4-vgg11-32': 1,
 'densenet201-2-efficientnet_b6-4-resnet18-32': 1,
 'mobilenet_v2-4-resnet152-4-vgg11-8': 1,
 'alexnet-8-resnet18-32-vgg11-8': 0,
 'efficientnet_b6-2-inception_v3-16-resnet50-4': 1,
 'resnet152-16-resnet34-16-squeezenet1_0-8': 0,
 'mobilenet_v2-4-mobilenet_v3_small-16-resnet34-32': 1,
 'mobilenet_v3_small-32-resnet34-2-vgg19-8': 0,
 'resnet101-32-efficientnet_b7-16-squeezenet1_0-4': 0,
 'densenet201-2-resnet101-4-vgg13-2': 1,
 'densenet121-2-inception_v3-16-vgg13-32': 1,
 'squeezenet1_0-16-vgg11-8-vgg13-4': 1,
 'densenet161-16-densenet169-8-vgg11-16': 1,
 'inception_v3-8-vgg13-2-vgg13-32': 0,
 'densenet161-2-efficientnet_b7-32-resnet101-2': 1,
 'alexnet-8-densenet161-4-vgg11-4': 1,
 'resnet152-8-vgg11-8-vgg19-16': 0,
 'efficientnet_b7-16-mobilenet_v3_large-8-vgg16-32': 0,
 'resnet101-4-squeezenet1_0-32-vgg19-16': 0,
 'densenet161-4-densenet201-2-vgg13-2': 1,
 'alexnet-16-inception_v3-2-squeezenet1_1-16': 1,
 'efficientnet_b6-16-mobilenet_v3_small-8-resnet152-2': 0,
 'alexnet-32-inception_v3-2-vgg11-32': 0,
 'resnet50-2-squeezenet1_0-16-squeezenet1_1-32': 1,
 'densenet201-16-inception_v3-32-resnet18-8': 1,
 'densenet121-32-densenet161-8-resnet18-2': 0,
 'densenet161-32-vgg13-4-vgg16-32': 1,
 'densenet121-2-densenet169-8-efficientnet_b7-4': 1,
 'alexnet-2-densenet121-8-efficientnet_b5-2': 1,
 'densenet161-16-mobilenet_v3_large-4-vgg19-8': 0,
 'resnet101-16-resnet18-16-vgg11-16': 0,
 'resnet50-16-densenet121-8-resnet34-4': 0,
 'inception_v3-8-resnet34-16-resnet50-2': 1,
 'densenet169-32-resnet101-4-resnet34-2': 0,
 'inception_v3-4-inception_v3-8-resnet18-8': 1,
 'densenet121-8-mobilenet_v3_small-16-squeezenet1_0-16': 1,
 'mobilenet_v2-32-mobilenet_v2-4-vgg13-2': 1,
 'densenet169-8-vgg16-16-vgg19-2': 1,
 'resnet101-32-inception_v3-2-resnet34-8': 0,
 'densenet169-16-mobilenet_v3_small-8-resnet34-8': 0,
 'mobilenet_v2-16-resnet152-32-squeezenet1_0-8': 1,
 'mobilenet_v3_small-16-mobilenet_v3_small-32-vgg13-8': 0,
 'densenet121-16-resnet34-16-squeezenet1_1-16': 1,
 'densenet161-4-mobilenet_v2-32-resnet152-16': 1,
 'efficientnet_b5-32-squeezenet1_1-16-vgg13-32': 0,
 'resnet101-8-resnet152-32-vgg16-32': 1,
 'densenet161-2-resnet152-2-resnet18-4': 1,
 'densenet169-32-densenet169-32-vgg13-4': 1,
 'resnet101-2-resnet18-16-squeezenet1_0-8': 1,
 'densenet161-16-densenet121-4-inception_v3-2': 0,
 'densenet121-32-resnet50-8-squeezenet1_1-16': 0,
 'efficientnet_b5-4-resnet50-16-vgg16-2': 1,
 'densenet121-32-efficientnet_b5-8-resnet18-16': 0,
 'inception_v3-2-squeezenet1_0-32-vgg16-32': 0,
 'efficientnet_b6-2-resnet101-2-vgg16-32': 1,
 'vgg13-8-alexnet-2-resnet34-8': 0,
 'alexnet-2-densenet161-16-resnet34-2': 1,
 'alexnet-8-efficientnet_b5-2-resnet101-8': 1,
 'densenet161-16-mobilenet_v3_large-4-resnet34-2': 0,
 'inception_v3-4-resnet101-32-resnet34-32': 1,
 'alexnet-2-densenet161-4-densenet169-8': 1,
 'densenet161-2-densenet161-32-efficientnet_b7-16': 1,
 'alexnet-2-resnet101-8-squeezenet1_1-32': 1,
 'densenet121-32-mobilenet_v3_large-32-resnet50-32': 0,
 'mobilenet_v3_large-2-resnet152-4-vgg19-2': 1,
 'mobilenet_v2-4-mobilenet_v3_large-16-vgg11-2': 1,
 'efficientnet_b7-16-squeezenet1_0-4-vgg19-2': 1,
 'efficientnet_b6-4-resnet50-8-squeezenet1_0-16': 1,
 'resnet18-2-squeezenet1_1-4-vgg13-2': 1,
 'resnet18-32-resnet34-4-squeezenet1_1-32': 0,
 'densenet201-8-squeezenet1_0-32-vgg13-8': 1,
 'densenet161-2-resnet18-2-vgg13-32': 0,
 'alexnet-4-mobilenet_v3_large-8-squeezenet1_1-2': 0,
 'densenet201-32-densenet121-8-vgg11-16': 0,
 'densenet169-2-squeezenet1_0-32-vgg13-16': 1,
 'alexnet-32-mobilenet_v3_small-2-resnet18-2': 0,
 'inception_v3-2-mobilenet_v3_large-4-squeezenet1_1-32': 1,
 'densenet121-4-efficientnet_b5-2-resnet101-8': 1,
 'mobilenet_v2-8-resnet18-32-resnet50-8': 1,
 'densenet161-8-densenet169-4-mobilenet_v3_small-16': 0,
 'densenet201-4-efficientnet_b5-2-vgg16-8': 1,
 'densenet201-2-densenet201-8-squeezenet1_0-16': 1,
 'efficientnet_b6-16-efficientnet_b7-2-squeezenet1_0-2': 0,
 'inception_v3-16-squeezenet1_1-2-squeezenet1_1-4': 0,
 'mobilenet_v3_large-8-mobilenet_v3_small-2-resnet34-2': 0,
 'mobilenet_v2-32-resnet50-16-squeezenet1_1-2': 0,
 'inception_v3-8-mobilenet_v3_small-8-vgg16-16': 0,
 'densenet121-4-squeezenet1_0-16-vgg11-16': 1,
 'alexnet-16-vgg11-32-vgg19-4': 1,
 'efficientnet_b5-32-resnet101-2-resnet34-4': 0,
 'squeezenet1_0-8-vgg11-32-vgg11-4': 0,
 'efficientnet_b7-4-efficientnet_b5-2-resnet101-2': 0,
 'densenet161-32-resnet18-32-resnet18-4': 0,
 'mobilenet_v3_small-32-resnet152-4-squeezenet1_0-8': 1,
 'resnet50-32-alexnet-2-vgg13-4': 1,
 'densenet201-2-resnet101-8-resnet50-4': 1,
 'densenet201-2-resnet152-16-vgg16-32': 1,
 'inception_v3-16-squeezenet1_1-2-vgg16-8': 0,
 'densenet161-8-densenet169-16-resnet18-4': 0,
 'efficientnet_b7-32-resnet152-8-resnet34-2': 0,
 'efficientnet_b7-4-resnet18-4-resnet34-2': 0,
 'mobilenet_v3_large-8-resnet18-4-squeezenet1_0-16': 1,
 'densenet161-2-efficientnet_b5-4-resnet152-16': 1,
 'alexnet-8-densenet121-4-vgg13-8': 0,
 'densenet161-4-squeezenet1_0-4-vgg11-2': 1,
 'densenet169-32-efficientnet_b6-4-mobilenet_v3_large-2': 0,
 'resnet101-2-squeezenet1_0-2-vgg11-32': 0,
 'inception_v3-4-squeezenet1_0-32-squeezenet1_0-4': 1,
 'inception_v3-4-squeezenet1_0-16-vgg11-4': 1,
 'densenet169-8-efficientnet_b5-2-resnet50-2': 0,
 'efficientnet_b7-4-efficientnet_b7-4-vgg16-16': 1,
 'densenet161-32-resnet101-2-resnet152-32': 0,
 'densenet121-4-resnet18-8-vgg13-4': 1,
 'inception_v3-8-mobilenet_v3_large-16-mobilenet_v3_small-32': 1,
 'alexnet-32-vgg13-4-vgg19-4': 1,
 'densenet161-16-mobilenet_v2-2-vgg19-32': 0,
 'resnet18-4-squeezenet1_0-8-vgg16-32': 0,
 'efficientnet_b7-4-resnet152-4-squeezenet1_1-4': 0,
 'densenet169-32-efficientnet_b6-2-squeezenet1_1-4': 0,
 'resnet18-4-resnet34-8-vgg19-32': 0,
 'resnet18-16-resnet18-8-vgg19-16': 0,
 'vgg19-4-resnet101-8-resnet34-8': 1,
 'densenet201-8-squeezenet1_0-16-vgg13-16': 1,
 'densenet161-32-squeezenet1_1-4-vgg19-16': 0,
 'densenet121-8-densenet169-2-vgg16-2': 1,
 'densenet201-2-mobilenet_v3_small-32-resnet101-4': 1,
 'alexnet-32-efficientnet_b6-2-efficientnet_b7-2': 1}
job_size_3_targets = {'efficientnet_b6-16-mobilenet_v3_small-8-vgg19-4': 1,
 'alexnet-2-resnet101-4-vgg13-16': 0,
 'densenet161-8-efficientnet_b5-8-inception_v3-32': 1,
 'densenet121-16-vgg11-8-vgg11-8': 1,
 'alexnet-2-densenet121-32-vgg16-16': 0,
 'densenet201-16-densenet201-8-vgg11-16': 1,
 'inception_v3-16-inception_v3-8-vgg16-16': 0,
 'efficientnet_b6-8-resnet34-8-squeezenet1_1-4': 0,
 'efficientnet_b6-4-resnet50-16-vgg13-16': 1,
 'densenet169-2-resnet18-16-vgg13-8': 1,
 'efficientnet_b5-4-vgg13-32-vgg16-16': 0,
 'densenet169-32-squeezenet1_0-2-squeezenet1_0-2': 0,
 'inception_v3-8-mobilenet_v3_large-8-resnet18-2': 0,
 'efficientnet_b7-32-mobilenet_v3_small-2-vgg19-16': 0,
 'squeezenet1_0-32-alexnet-4-mobilenet_v2-4': 1,
 'efficientnet_b7-2-resnet101-2-resnet101-32': 1,
 'resnet18-32-resnet50-16-resnet50-8': 1,
 'densenet161-4-mobilenet_v2-16-vgg11-2': 1,
 'vgg19-4-efficientnet_b6-16-resnet18-8': 1,
 'alexnet-2-mobilenet_v3_small-16-resnet152-4': 1,
 'efficientnet_b5-8-resnet152-8-vgg11-4': 1,
 'mobilenet_v2-8-mobilenet_v3_small-8-vgg19-32': 0,
 'mobilenet_v3_small-16-resnet50-4-vgg19-32': 0,
 'mobilenet_v2-32-squeezenet1_0-4-vgg19-8': 0,
 'alexnet-16-densenet121-8-mobilenet_v3_small-4': 0,
 'densenet201-4-inception_v3-4-squeezenet1_0-2': 0,
 'alexnet-16-resnet18-8-squeezenet1_1-8': 0,
 'densenet169-2-resnet18-4-vgg16-2': 1,
 'efficientnet_b5-4-efficientnet_b6-16-vgg11-2': 1,
 'inception_v3-2-mobilenet_v3_small-4-resnet152-8': 1,
 'densenet121-16-resnet152-2-squeezenet1_1-32': 1,
 'efficientnet_b6-16-efficientnet_b7-2-vgg13-2': 1,
 'efficientnet_b5-32-inception_v3-8-resnet18-4': 0,
 'resnet152-16-squeezenet1_1-8-vgg13-4': 0,
 'densenet121-16-efficientnet_b6-4-efficientnet_b7-32': 0,
 'efficientnet_b7-32-mobilenet_v3_large-4-squeezenet1_0-8': 0,
 'resnet50-8-efficientnet_b7-2-vgg13-8': 1,
 'efficientnet_b7-16-resnet152-2-resnet152-8': 0,
 'densenet121-32-mobilenet_v3_large-16-vgg16-2': 1,
 'alexnet-16-inception_v3-2-vgg13-8': 0,
 'efficientnet_b7-16-mobilenet_v2-4-resnet152-16': 0,
 'efficientnet_b7-16-inception_v3-16-resnet101-2': 0,
 'resnet152-4-resnet34-2-vgg19-16': 0,
 'densenet201-4-mobilenet_v2-4-mobilenet_v3_large-4': 0,
 'densenet161-16-efficientnet_b5-2-vgg19-4': 1,
 'densenet169-2-efficientnet_b5-2-squeezenet1_0-2': 0,
 'resnet152-16-resnet34-32-vgg19-2': 1,
 'efficientnet_b6-8-mobilenet_v3_small-16-squeezenet1_0-16': 1,
 'densenet201-16-resnet34-32-vgg13-16': 1,
 'densenet169-32-efficientnet_b7-16-resnet34-8': 0,
 'resnet152-2-vgg19-2-vgg19-2': 1,
 'densenet121-2-mobilenet_v3_small-8-vgg11-2': 1,
 'efficientnet_b7-16-mobilenet_v3_large-32-vgg11-16': 1,
 'densenet121-8-vgg11-16-vgg19-4': 1,
 'densenet169-32-inception_v3-2-squeezenet1_1-16': 0,
 'resnet34-2-vgg16-32-vgg16-8': 0,
 'densenet169-32-efficientnet_b7-2-vgg13-2': 1,
 'efficientnet_b6-2-resnet101-2-squeezenet1_0-32': 1,
 'densenet169-2-inception_v3-16-resnet34-4': 1,
 'efficientnet_b6-4-mobilenet_v2-2-squeezenet1_0-16': 1,
 'densenet201-16-inception_v3-16-squeezenet1_1-4': 0,
 'efficientnet_b6-2-mobilenet_v2-4-vgg13-8': 1,
 'densenet121-16-densenet161-16-densenet201-32': 1,
 'densenet169-16-mobilenet_v3_small-4-vgg19-4': 1,
 'resnet101-32-efficientnet_b5-4-resnet101-4': 1,
 'vgg19-4-resnet101-2-vgg11-2': 0,
 'efficientnet_b6-32-resnet50-16-vgg16-4': 1,
 'densenet121-32-efficientnet_b6-2-efficientnet_b7-32': 0,
 'densenet161-32-resnet50-2-vgg19-16': 0,
 'inception_v3-16-mobilenet_v2-2-resnet50-8': 0,
 'resnet152-8-resnet34-4-vgg16-32': 0,
 'inception_v3-32-mobilenet_v3_large-16-vgg16-32': 0,
 'densenet169-2-efficientnet_b7-4-resnet34-32': 1,
 'inception_v3-2-vgg11-16-vgg16-32': 0,
 'efficientnet_b5-4-efficientnet_b5-8-mobilenet_v3_small-8': 1,
 'efficientnet_b5-16-resnet34-32-vgg19-16': 1,
 'densenet201-8-resnet50-8-vgg11-2': 1,
 'densenet121-16-vgg19-16-vgg19-2': 1,
 'densenet161-32-efficientnet_b6-2-vgg19-2': 1,
 'densenet121-2-mobilenet_v2-32-mobilenet_v3_large-2': 1,
 'efficientnet_b5-2-vgg13-32-vgg19-2': 0,
 'mobilenet_v3_large-8-resnet152-4-resnet152-8': 0,
 'efficientnet_b6-32-mobilenet_v2-32-mobilenet_v3_large-16': 0,
 'densenet169-8-efficientnet_b7-16-resnet18-8': 0,
 'densenet121-4-squeezenet1_1-4-vgg19-2': 0,
 'alexnet-8-efficientnet_b5-4-resnet34-2': 0,
 'alexnet-32-densenet169-2-efficientnet_b6-32': 0,
 'densenet161-4-mobilenet_v3_small-32-resnet152-16': 1,
 'squeezenet1_0-32-resnet152-8-squeezenet1_1-32': 1,
 'mobilenet_v3_small-2-squeezenet1_1-4-vgg16-2': 1,
 'densenet169-16-efficientnet_b5-32-vgg11-32': 1,
 'efficientnet_b5-16-efficientnet_b5-32-squeezenet1_0-8': 0,
 'resnet152-16-resnet152-2-vgg19-16': 0,
 'inception_v3-16-mobilenet_v2-4-mobilenet_v3_small-8': 0,
 'alexnet-8-efficientnet_b5-16-vgg11-4': 0,
 'inception_v3-32-inception_v3-16-resnet34-2': 1,
 'densenet169-2-mobilenet_v2-8-resnet34-8': 1,
 'densenet169-8-efficientnet_b7-8-resnet101-4': 0,
 'squeezenet1_1-2-vgg11-8-vgg19-32': 0,
 'densenet121-4-densenet169-4-vgg11-4': 1,
 'densenet161-8-efficientnet_b7-32-resnet152-8': 1,
 'efficientnet_b6-4-inception_v3-4-vgg19-16': 1,
 'alexnet-8-efficientnet_b6-4-squeezenet1_0-4': 0,
 'inception_v3-8-resnet18-32-vgg19-16': 0,
 'vgg16-4-densenet169-16-mobilenet_v2-8': 0,
 'mobilenet_v2-2-resnet18-2-vgg19-2': 1,
 'densenet121-8-resnet18-8-vgg16-2': 1,
 'densenet169-4-mobilenet_v2-8-resnet18-4': 1,
 'densenet121-16-mobilenet_v2-8-vgg19-16': 0,
 'vgg19-8-densenet201-8-resnet34-2': 0,
 'mobilenet_v2-4-mobilenet_v3_small-32-vgg16-2': 1,
 'mobilenet_v3_small-16-vgg16-2-vgg19-32': 1,
 'squeezenet1_1-16-squeezenet1_1-16-vgg13-4': 0,
 'densenet161-32-vgg11-32-vgg19-32': 1,
 'densenet201-8-resnet18-4-squeezenet1_0-32': 1,
 'densenet161-8-resnet101-32-resnet18-16': 1,
 'densenet169-16-densenet201-32-vgg19-2': 1,
 'vgg16-2-vgg19-16-vgg19-4': 1,
 'efficientnet_b7-4-mobilenet_v3_large-32-vgg16-16': 1,
 'alexnet-8-inception_v3-4-mobilenet_v3_large-8': 0,
 'resnet50-32-vgg11-2-vgg13-8': 1,
 'densenet169-4-resnet101-16-squeezenet1_0-4': 1,
 'squeezenet1_1-4-vgg13-4-vgg13-8': 0,
 'mobilenet_v3_small-32-resnet50-8-vgg13-16': 0,
 'densenet169-32-efficientnet_b6-8-mobilenet_v3_large-8': 0,
 'densenet121-8-densenet201-16-vgg16-4': 1,
 'densenet169-32-resnet152-8-vgg16-8': 0,
 'alexnet-32-densenet201-8-efficientnet_b5-32': 1,
 'densenet161-8-mobilenet_v3_small-32-resnet101-4': 1,
 'alexnet-4-squeezenet1_1-32-vgg13-32': 0,
 'densenet121-4-efficientnet_b7-4-vgg16-4': 1,
 'densenet169-8-efficientnet_b6-8-mobilenet_v3_small-32': 1,
 'densenet161-8-densenet169-4-efficientnet_b6-8': 0,
 'densenet201-8-resnet152-16-resnet34-16': 1,
 'resnet101-16-resnet152-16-squeezenet1_1-2': 0,
 'efficientnet_b6-2-squeezenet1_0-32-vgg19-2': 1,
 'densenet169-4-mobilenet_v2-8-squeezenet1_0-8': 1,
 'vgg13-2-densenet169-2-squeezenet1_1-8': 1,
 'densenet169-4-efficientnet_b7-8-squeezenet1_1-16': 1,
 'efficientnet_b7-4-squeezenet1_0-16-vgg19-16': 1,
 'efficientnet_b6-16-efficientnet_b6-2-inception_v3-4': 0,
 'inception_v3-32-resnet18-16-vgg16-8': 0,
 'inception_v3-4-mobilenet_v3_small-4-squeezenet1_1-4': 0,
 'resnet50-2-squeezenet1_0-8-vgg13-16': 1,
 'densenet169-16-mobilenet_v3_large-2-vgg16-32': 0,
 'squeezenet1_1-2-vgg16-4-vgg19-8': 0,
 'efficientnet_b7-4-vgg13-32-vgg16-4': 1,
 'vgg16-32-squeezenet1_0-32-vgg11-2': 0,
 'squeezenet1_0-4-vgg16-32-vgg16-4': 0,
 'mobilenet_v2-2-resnet152-2-vgg11-8': 1,
 'mobilenet_v3_small-16-squeezenet1_0-2-squeezenet1_1-16': 1,
 'densenet161-2-densenet201-16-efficientnet_b6-8': 1,
 'efficientnet_b5-8-resnet50-4-vgg13-8': 1,
 'densenet161-2-densenet201-4-efficientnet_b7-2': 1,
 'densenet121-4-resnet18-2-resnet18-8': 1,
 'densenet161-16-densenet201-4-vgg13-4': 0,
 'mobilenet_v3_large-16-mobilenet_v3_large-8-squeezenet1_1-2': 0,
 'alexnet-32-densenet161-32-mobilenet_v3_large-16': 0,
 'densenet121-8-mobilenet_v3_large-4-squeezenet1_1-16': 1,
 'alexnet-16-densenet121-2-mobilenet_v3_small-32': 1,
 'inception_v3-2-vgg11-4-vgg13-2': 1,
 'densenet121-4-efficientnet_b6-4-mobilenet_v3_large-32': 1,
 'resnet34-16-vgg16-2-vgg16-8': 1,
 'densenet161-8-densenet201-8-mobilenet_v3_small-8': 0,
 'efficientnet_b6-16-inception_v3-8-resnet18-32': 1,
 'efficientnet_b5-16-resnet101-4-vgg11-4': 0,
 'densenet121-8-inception_v3-16-resnet50-16': 1,
 'mobilenet_v3_large-4-vgg13-32-vgg19-16': 0,
 'densenet121-16-densenet169-8-mobilenet_v2-2': 0,
 'inception_v3-16-resnet18-8-vgg11-4': 1,
 'densenet201-16-efficientnet_b7-16-mobilenet_v3_large-8': 0,
 'densenet201-8-resnet101-8-resnet101-8': 0,
 'resnet18-2-resnet18-4-resnet50-8': 1,
 'mobilenet_v3_large-4-mobilenet_v3_small-4-resnet50-8': 1,
 'alexnet-8-mobilenet_v2-16-mobilenet_v3_small-2': 0,
 'mobilenet_v3_large-4-resnet101-2-vgg11-4': 1,
 'mobilenet_v3_large-4-resnet50-8-squeezenet1_1-4': 1,
 'densenet169-8-resnet34-8-vgg13-2': 1,
 'efficientnet_b7-8-inception_v3-4-vgg13-8': 1,
 'alexnet-16-mobilenet_v3_small-2-resnet152-2': 0,
 'vgg19-4-squeezenet1_0-8-squeezenet1_1-16': 1,
 'densenet161-8-resnet18-32-resnet50-4': 1,
 'densenet121-4-efficientnet_b6-4-vgg13-8': 1,
 'resnet18-32-resnet34-16-squeezenet1_0-2': 0,
 'efficientnet_b5-8-resnet50-2-vgg13-8': 1,
 'densenet169-4-densenet201-32-resnet152-2': 1,
 'mobilenet_v2-2-mobilenet_v2-32-resnet152-2': 0,
 'resnet34-16-resnet50-4-vgg11-32': 1,
 'alexnet-2-resnet152-16-vgg13-2': 1,
 'inception_v3-4-vgg16-8-vgg19-16': 0,
 'densenet121-16-resnet101-4-resnet18-8': 0,
 'efficientnet_b5-2-mobilenet_v3_small-8-resnet18-4': 1,
 'densenet121-2-resnet101-8-squeezenet1_0-16': 1,
 'densenet121-2-squeezenet1_1-4-vgg13-2': 1,
 'efficientnet_b7-2-resnet34-2-resnet50-8': 1,
 'densenet169-4-inception_v3-16-resnet152-32': 1,
 'densenet121-2-resnet101-4-vgg11-2': 1,
 'mobilenet_v3_large-32-squeezenet1_1-2-vgg13-16': 0,
 'densenet121-2-mobilenet_v3_large-32-squeezenet1_0-8': 1,
 'densenet121-4-efficientnet_b7-8-mobilenet_v3_large-32': 1,
 'mobilenet_v3_small-4-resnet18-4-squeezenet1_1-2': 0,
 'vgg16-4-alexnet-16-resnet18-32': 1,
 'densenet161-8-resnet101-16-vgg13-4': 1,
 'resnet101-8-squeezenet1_0-32-squeezenet1_0-4': 1,
 'densenet161-16-densenet169-32-densenet169-8': 1,
 'densenet161-2-densenet169-4-densenet201-2': 1,
 'efficientnet_b5-2-resnet101-16-resnet50-8': 1,
 'resnet18-32-squeezenet1_0-2-vgg16-16': 0,
 'vgg19-32-vgg19-4-inception_v3-8': 0,
 'densenet169-32-mobilenet_v3_large-16-vgg19-2': 1,
 'mobilenet_v3_large-8-mobilenet_v3_small-16-resnet18-4': 1,
 'densenet121-4-resnet101-32-resnet34-8': 1,
 'densenet201-4-resnet34-4-squeezenet1_0-2': 0,
 'efficientnet_b5-16-vgg11-16-vgg16-2': 1,
 'alexnet-4-densenet169-4-resnet50-16': 1,
 'densenet201-16-mobilenet_v3_small-32-squeezenet1_1-4': 1,
 'densenet169-32-resnet152-16-squeezenet1_1-2': 0,
 'resnet18-32-vgg11-16-vgg19-16': 0,
 'densenet201-32-mobilenet_v3_large-32-resnet152-16': 0,
 'mobilenet_v2-32-vgg16-16-vgg16-16': 0,
 'mobilenet_v2-8-resnet50-4-squeezenet1_0-16': 1,
 'densenet121-8-densenet201-32-densenet201-4': 1,
 'densenet121-32-densenet201-8-mobilenet_v3_small-4': 0,
 'densenet161-32-efficientnet_b5-8-vgg11-32': 1,
 'efficientnet_b6-32-densenet169-4-resnet50-16': 1,
 'densenet161-16-densenet201-2-efficientnet_b7-8': 0,
 'densenet201-2-efficientnet_b6-2-resnet101-2': 0,
 'densenet169-32-mobilenet_v3_small-4-vgg13-8': 0,
 'densenet169-4-inception_v3-16-vgg16-16': 1,
 'inception_v3-32-vgg11-32-vgg19-2': 0,
 'resnet152-16-inception_v3-16-vgg11-8': 1,
 'resnet101-16-densenet201-16-inception_v3-2': 1,
 'efficientnet_b6-2-resnet18-8-squeezenet1_0-4': 1,
 'densenet121-16-densenet201-2-vgg16-32': 1,
 'densenet169-8-resnet34-16-squeezenet1_0-2': 1,
 'alexnet-2-resnet152-16-resnet152-8': 1,
 'densenet121-32-mobilenet_v3_large-16-mobilenet_v3_large-2': 0,
 'efficientnet_b6-2-resnet50-16-squeezenet1_1-16': 1,
 'resnet152-4-resnet34-32-resnet34-32': 1,
 'efficientnet_b6-2-resnet18-2-vgg16-2': 1,
 'densenet121-2-inception_v3-16-vgg19-4': 1,
 'resnet50-8-vgg16-4-vgg19-8': 1,
 'densenet121-8-resnet152-8-squeezenet1_0-32': 1,
 'densenet201-2-inception_v3-8-resnet101-2': 1,
 'mobilenet_v3_small-32-mobilenet_v3_small-32-vgg19-16': 0,
 'mobilenet_v3_small-2-resnet18-32-vgg11-8': 1,
 'mobilenet_v3_large-16-squeezenet1_1-32-vgg19-8': 0,
 'vgg16-8-densenet169-16-resnet101-8': 1,
 'resnet34-32-resnet34-32-squeezenet1_0-16': 1,
 'densenet121-2-efficientnet_b5-8-vgg19-2': 1,
 'resnet18-32-vgg11-4-vgg16-32': 1,
 'resnet18-4-squeezenet1_0-32-squeezenet1_0-8': 1,
 'mobilenet_v3_large-32-mobilenet_v3_large-4-resnet50-8': 0,
 'mobilenet_v2-2-mobilenet_v2-4-resnet152-2': 1,
 'densenet121-2-densenet201-4-vgg13-4': 1,
 'resnet101-16-resnet101-8-vgg13-16': 0,
 'densenet121-8-squeezenet1_1-4-vgg19-16': 0,
 'densenet169-32-inception_v3-32-resnet50-16': 0,
 'resnet101-8-resnet18-4-vgg13-2': 1,
 'densenet121-2-mobilenet_v2-8-vgg13-32': 1,
 'mobilenet_v2-32-resnet101-32-vgg19-32': 0,
 'squeezenet1_1-32-resnet18-8-resnet34-2': 0,
 'efficientnet_b6-8-resnet18-32-resnet34-16': 1,
 'alexnet-8-resnet152-2-resnet152-8': 1,
 'mobilenet_v3_large-2-vgg11-32-vgg16-2': 1,
 'efficientnet_b7-16-squeezenet1_1-32-vgg16-2': 1,
 'efficientnet_b5-8-mobilenet_v2-16-vgg19-2': 1,
 'mobilenet_v2-16-resnet34-16-resnet50-4': 1,
 'mobilenet_v2-8-resnet34-8-vgg16-2': 1,
 'densenet121-32-efficientnet_b7-4-efficientnet_b7-8': 0,
 'efficientnet_b7-2-resnet152-32-resnet34-4': 1,
 'squeezenet1_1-4-squeezenet1_1-8-vgg19-16': 0,
 'densenet161-16-efficientnet_b6-2-squeezenet1_0-16': 0,
 'densenet201-2-inception_v3-8-mobilenet_v3_small-2': 1,
 'alexnet-2-densenet161-2-efficientnet_b6-16': 1,
 'efficientnet_b5-2-mobilenet_v3_large-32-resnet101-8': 1,
 'inception_v3-8-resnet101-4-vgg13-8': 1,
 'densenet121-32-efficientnet_b6-32-resnet152-4': 0,
 'resnet34-16-resnet50-8-squeezenet1_1-32': 1,
 'mobilenet_v3_large-8-resnet18-16-vgg19-32': 0,
 'efficientnet_b7-16-mobilenet_v3_large-16-mobilenet_v3_small-8': 0,
 'efficientnet_b6-32-resnet152-16-resnet18-8': 0,
 'resnet34-16-vgg19-2-vgg19-8': 1,
 'resnet101-32-resnet34-8-vgg19-2': 1,
 'resnet152-32-efficientnet_b7-2-vgg16-8': 1,
 'vgg19-2-mobilenet_v3_small-4-resnet34-4': 1,
 'inception_v3-8-resnet152-8-vgg16-32': 0,
 'densenet169-16-efficientnet_b5-16-resnet152-4': 1,
 'alexnet-32-mobilenet_v3_small-32-squeezenet1_1-4': 1,
 'resnet34-32-densenet201-4-squeezenet1_1-8': 1,
 'mobilenet_v2-4-mobilenet_v3_large-32-resnet101-2': 1,
 'densenet169-16-resnet18-16-vgg16-32': 0,
 'inception_v3-4-resnet50-4-vgg19-32': 1,
 'densenet161-16-densenet169-2-efficientnet_b6-8': 0,
 'inception_v3-2-mobilenet_v3_large-32-vgg16-2': 1,
 'inception_v3-32-mobilenet_v3_small-4-vgg16-8': 0,
 'densenet201-4-inception_v3-2-squeezenet1_1-32': 1,
 'densenet121-2-efficientnet_b6-8-resnet101-4': 1,
 'efficientnet_b7-4-squeezenet1_0-8-vgg13-4': 1,
 'alexnet-32-densenet201-16-resnet101-16': 1,
 'densenet161-8-mobilenet_v3_large-32-squeezenet1_0-4': 1,
 'resnet101-8-vgg19-4-vgg19-4': 1,
 'densenet201-8-mobilenet_v3_small-16-squeezenet1_0-2': 1,
 'inception_v3-8-vgg13-2-vgg13-2': 1,
 'efficientnet_b6-8-densenet161-4-resnet152-8': 1,
 'inception_v3-2-inception_v3-2-vgg16-2': 1,
 'densenet161-4-resnet152-32-squeezenet1_0-2': 0,
 'densenet201-2-vgg11-2-vgg11-8': 1,
 'efficientnet_b5-4-efficientnet_b6-16-resnet34-16': 1,
 'efficientnet_b6-2-resnet18-32-resnet50-16': 1,
 'alexnet-8-efficientnet_b5-4-vgg16-2': 0,
 'efficientnet_b5-32-mobilenet_v3_large-2-resnet50-8': 0,
 'vgg19-32-densenet201-2-vgg11-8': 0,
 'densenet169-2-inception_v3-32-vgg16-8': 1,
 'densenet161-4-mobilenet_v3_small-8-vgg19-4': 1,
 'efficientnet_b6-4-resnet50-4-squeezenet1_1-32': 1,
 'resnet101-4-squeezenet1_0-2-vgg16-32': 0,
 'efficientnet_b5-32-squeezenet1_0-32-vgg19-8': 0,
 'densenet121-16-mobilenet_v3_small-8-vgg11-16': 0,
 'mobilenet_v3_large-4-mobilenet_v3_small-4-squeezenet1_1-8': 1,
 'mobilenet_v2-32-resnet34-2-vgg19-32': 0,
 'mobilenet_v2-16-mobilenet_v2-8-mobilenet_v3_large-32': 1,
 'densenet161-8-densenet161-8-densenet201-2': 1,
 'squeezenet1_1-2-vgg11-4-vgg19-2': 0,
 'alexnet-2-efficientnet_b6-8-vgg16-2': 1,
 'densenet169-2-resnet50-8-squeezenet1_0-8': 1,
 'densenet161-2-efficientnet_b6-8-resnet50-8': 1,
 'alexnet-8-mobilenet_v3_large-8-vgg16-2': 0,
 'inception_v3-8-resnet101-8-vgg13-16': 1,
 'efficientnet_b6-32-resnet18-4-vgg13-16': 0,
 'efficientnet_b5-16-mobilenet_v3_small-4-squeezenet1_0-32': 0,
 'mobilenet_v2-16-mobilenet_v3_large-4-mobilenet_v3_small-2': 0,
 'densenet169-8-efficientnet_b6-4-mobilenet_v3_small-2': 0,
 'inception_v3-32-squeezenet1_0-2-vgg19-16': 0,
 'alexnet-2-efficientnet_b5-8-resnet152-8': 0,
 'resnet101-16-vgg11-2-vgg19-32': 0,
 'densenet161-8-densenet201-16-densenet201-2': 1,
 'squeezenet1_0-2-squeezenet1_1-2-vgg16-2': 0,
 'efficientnet_b7-4-resnet18-2-vgg13-2': 0,
 'inception_v3-32-densenet201-16-resnet101-4': 1,
 'alexnet-32-mobilenet_v3_large-4-resnet18-4': 0,
 'mobilenet_v3_large-8-vgg11-32-vgg16-4': 0,
 'densenet201-16-mobilenet_v3_small-4-vgg16-8': 0,
 'resnet101-16-resnet34-8-squeezenet1_1-32': 1,
 'efficientnet_b6-8-mobilenet_v3_large-2-vgg13-4': 0,
 'alexnet-4-resnet34-16-vgg19-2': 1,
 'inception_v3-32-mobilenet_v2-2-resnet18-16': 0,
 'efficientnet_b7-16-mobilenet_v2-4-resnet50-8': 0,
 'densenet161-4-mobilenet_v3_large-2-squeezenet1_0-8': 1,
 'efficientnet_b7-8-mobilenet_v2-32-resnet50-2': 1,
 'efficientnet_b6-32-resnet18-32-resnet34-32': 0,
 'densenet201-2-efficientnet_b6-8-vgg13-16': 1,
 'resnet152-2-resnet34-16-resnet50-2': 1,
 'alexnet-32-efficientnet_b6-16-resnet50-32': 0,
 'densenet161-2-resnet152-2-resnet34-32': 1,
 'efficientnet_b6-4-resnet50-4-vgg11-2': 1,
 'efficientnet_b5-2-mobilenet_v3_small-32-resnet50-8': 1,
 'inception_v3-8-mobilenet_v3_large-8-resnet152-16': 0,
 'resnet101-32-squeezenet1_1-32-vgg16-32': 0,
 'resnet101-2-resnet152-8-resnet50-4': 1,
 'densenet169-8-densenet201-2-mobilenet_v2-16': 1,
 'efficientnet_b5-8-inception_v3-8-mobilenet_v3_large-16': 1,
 'mobilenet_v2-4-resnet50-8-squeezenet1_1-16': 1,
 'densenet201-16-mobilenet_v3_small-16-resnet101-2': 0,
 'densenet161-4-resnet101-16-resnet50-16': 1,
 'densenet201-16-squeezenet1_1-4-vgg13-32': 0,
 'densenet121-32-efficientnet_b6-16-mobilenet_v3_large-16': 0,
 'mobilenet_v2-2-resnet152-32-vgg11-16': 1,
 'mobilenet_v3_large-4-resnet18-2-vgg19-8': 0,
 'vgg16-2-mobilenet_v3_large-16-resnet18-2': 1,
 'alexnet-4-alexnet-4-densenet201-16': 1,
 'densenet169-16-densenet201-2-inception_v3-32': 1,
 'alexnet-32-efficientnet_b5-16-squeezenet1_1-2': 0,
 'squeezenet1_1-32-vgg11-8-vgg19-8': 0,
 'inception_v3-2-mobilenet_v3_large-16-vgg19-32': 0,
 'resnet18-16-squeezenet1_0-16-vgg16-4': 1,
 'inception_v3-4-squeezenet1_0-4-vgg19-4': 0,
 'vgg19-32-resnet18-2-squeezenet1_1-8': 1,
 'densenet169-8-mobilenet_v3_large-2-resnet50-32': 0,
 'densenet161-8-vgg11-32-vgg13-2': 1,
 'efficientnet_b7-2-vgg11-4-vgg19-8': 1,
 'densenet121-4-mobilenet_v3_large-16-resnet101-8': 1,
 'resnet101-4-resnet152-2-resnet50-8': 1,
 'mobilenet_v3_large-4-resnet101-32-vgg13-32': 0,
 'resnet101-4-densenet201-2-mobilenet_v3_large-2': 1,
 'efficientnet_b6-2-mobilenet_v2-2-resnet50-4': 1,
 'densenet121-8-mobilenet_v3_small-8-vgg11-8': 0,
 'densenet121-32-resnet152-8-squeezenet1_0-2': 0,
 'inception_v3-16-inception_v3-32-mobilenet_v3_large-2': 0,
 'densenet161-8-resnet152-8-resnet18-8': 0,
 'alexnet-8-resnet152-32-squeezenet1_0-2': 0,
 'efficientnet_b6-32-efficientnet_b6-2-squeezenet1_0-16': 1,
 'densenet169-16-efficientnet_b6-2-vgg13-16': 1,
 'densenet201-4-efficientnet_b7-8-vgg13-4': 1,
 'alexnet-16-mobilenet_v3_large-8-resnet18-2': 0,
 'densenet121-4-resnet152-2-resnet34-8': 1,
 'squeezenet1_0-16-squeezenet1_1-16-squeezenet1_1-8': 0,
 'mobilenet_v3_small-32-resnet152-32-vgg11-16': 0,
 'densenet169-8-densenet201-4-squeezenet1_0-4': 0,
 'densenet169-16-mobilenet_v3_large-2-resnet34-8': 0,
 'alexnet-8-resnet50-16-squeezenet1_0-4': 1,
 'densenet161-2-densenet169-4-resnet152-16': 1,
 'densenet169-2-efficientnet_b6-8-resnet34-8': 1,
 'efficientnet_b6-8-mobilenet_v3_small-16-vgg13-4': 0,
 'efficientnet_b5-16-mobilenet_v3_large-2-resnet18-32': 1,
 'densenet121-8-densenet161-4-squeezenet1_1-32': 1,
 'efficientnet_b5-32-mobilenet_v2-16-resnet18-4': 0,
 'densenet169-4-efficientnet_b7-32-efficientnet_b7-4': 1,
 'densenet161-32-efficientnet_b5-4-vgg16-2': 1,
 'densenet121-4-squeezenet1_0-16-vgg13-4': 1,
 'alexnet-16-efficientnet_b5-4-mobilenet_v3_large-8': 0,
 'densenet121-2-densenet169-8-efficientnet_b7-32': 1,
 'densenet161-2-efficientnet_b7-4-vgg19-16': 1,
 'densenet169-2-efficientnet_b5-4-resnet101-16': 1,
 'alexnet-32-efficientnet_b6-2-resnet18-2': 1,
 'efficientnet_b6-16-mobilenet_v2-2-vgg16-8': 0,
 'mobilenet_v3_large-8-resnet34-8-squeezenet1_0-4': 0,
 'resnet18-32-vgg11-16-vgg16-8': 0,
 'mobilenet_v3_large-2-vgg11-16-vgg13-32': 0,
 'densenet169-2-inception_v3-8-squeezenet1_1-32': 1,
 'efficientnet_b5-4-efficientnet_b6-8-vgg19-2': 1,
 'inception_v3-8-resnet101-8-squeezenet1_0-8': 1,
 'mobilenet_v3_small-4-resnet50-4-vgg13-32': 0,
 'mobilenet_v3_small-16-resnet34-32-vgg19-32': 0,
 'efficientnet_b7-16-mobilenet_v2-8-vgg19-16': 0,
 'densenet201-8-densenet201-8-resnet34-8': 1,
 'alexnet-8-efficientnet_b6-32-squeezenet1_1-4': 0,
 'densenet169-8-efficientnet_b5-4-resnet101-2': 0,
 'efficientnet_b7-16-mobilenet_v3_large-32-mobilenet_v3_small-2': 0,
 'densenet169-2-vgg11-16-vgg13-8': 1,
 'densenet201-32-mobilenet_v2-32-resnet101-32': 0,
 'efficientnet_b7-8-resnet18-16-vgg13-32': 0,
 'alexnet-8-resnet152-8-vgg13-8': 0,
 'alexnet-8-densenet121-32-vgg19-4': 0,
 'efficientnet_b7-2-inception_v3-4-resnet34-32': 1,
 'squeezenet1_1-8-vgg11-32-vgg19-4': 0,
 'densenet169-4-inception_v3-4-resnet34-8': 1,
 'vgg13-32-vgg11-2-vgg16-2': 1,
 'densenet201-4-inception_v3-16-squeezenet1_1-32': 1,
 'densenet201-4-resnet50-8-squeezenet1_0-16': 1,
 'densenet161-32-mobilenet_v3_large-32-resnet101-16': 0,
 'alexnet-8-efficientnet_b6-8-mobilenet_v3_large-8': 0,
 'efficientnet_b5-8-resnet50-2-squeezenet1_1-8': 0,
 'resnet101-16-resnet18-16-vgg13-2': 1,
 'efficientnet_b6-4-squeezenet1_0-32-vgg13-4': 1,
 'densenet161-32-efficientnet_b6-2-mobilenet_v3_small-2': 0,
 'densenet161-2-densenet169-4-resnet101-8': 1,
 'efficientnet_b6-2-mobilenet_v3_large-16-vgg16-16': 1,
 'efficientnet_b7-32-mobilenet_v3_large-4-vgg11-8': 0,
 'mobilenet_v2-32-densenet121-8-resnet18-4': 0,
 'densenet169-32-inception_v3-2-mobilenet_v3_large-16': 0,
 'densenet169-4-resnet152-16-resnet18-8': 1,
 'densenet169-32-resnet101-8-vgg16-4': 1,
 'densenet121-4-densenet169-32-vgg13-8': 1,
 'densenet161-32-mobilenet_v3_large-4-vgg11-8': 0,
 'densenet161-2-resnet152-4-vgg11-4': 1,
 'densenet121-16-efficientnet_b7-16-mobilenet_v2-4': 0,
 'alexnet-16-alexnet-4-mobilenet_v3_small-8': 0,
 'resnet18-32-resnet34-2-vgg13-4': 1,
 'inception_v3-16-mobilenet_v2-4-squeezenet1_0-16': 1,
 'densenet161-2-resnet18-4-squeezenet1_0-4': 1,
 'resnet34-16-resnet50-8-vgg11-16': 0,
 'inception_v3-8-resnet152-4-resnet50-8': 1,
 'mobilenet_v2-8-resnet18-32-vgg11-8': 1,
 'efficientnet_b5-16-mobilenet_v3_small-8-resnet34-2': 0,
 'vgg19-2-densenet161-2-densenet169-4': 0,
 'densenet121-32-resnet18-16-squeezenet1_0-16': 0,
 'alexnet-4-efficientnet_b6-32-resnet101-4': 0,
 'densenet201-4-efficientnet_b5-2-inception_v3-4': 0,
 'efficientnet_b7-4-mobilenet_v3_small-4-squeezenet1_1-4': 0,
 'squeezenet1_1-8-vgg11-8-vgg13-16': 0,
 'inception_v3-2-resnet152-2-vgg13-8': 1,
 'resnet18-8-resnet34-16-vgg16-8': 0,
 'efficientnet_b5-8-vgg13-4-vgg19-8': 1,
 'densenet161-2-densenet201-2-resnet18-4': 1,
 'vgg16-2-efficientnet_b6-2-squeezenet1_1-4': 0,
 'densenet121-8-mobilenet_v3_large-32-squeezenet1_0-8': 1,
 'efficientnet_b5-16-efficientnet_b7-16-mobilenet_v3_small-32': 0,
 'densenet201-8-efficientnet_b6-2-mobilenet_v2-32': 1,
 'squeezenet1_0-16-efficientnet_b5-4-squeezenet1_0-4': 1,
 'alexnet-2-mobilenet_v3_small-32-squeezenet1_1-8': 1,
 'densenet161-4-mobilenet_v3_large-32-vgg19-32': 0,
 'densenet161-8-resnet101-4-squeezenet1_1-2': 0,
 'squeezenet1_0-32-vgg11-16-vgg19-8': 0,
 'inception_v3-32-resnet152-32-resnet152-32': 0,
 'densenet121-8-densenet169-16-resnet50-16': 1,
 'densenet161-32-efficientnet_b5-32-efficientnet_b6-2': 0,
 'inception_v3-16-efficientnet_b7-8-resnet101-8': 0,
 'efficientnet_b7-4-efficientnet_b7-4-vgg11-2': 1,
 'inception_v3-8-squeezenet1_1-32-vgg11-16': 1,
 'inception_v3-4-resnet34-2-squeezenet1_0-8': 1,
 'densenet121-2-densenet161-16-vgg11-16': 1,
 'densenet201-32-mobilenet_v2-16-resnet50-8': 0,
 'densenet121-16-efficientnet_b6-32-resnet101-32': 1,
 'mobilenet_v3_small-32-resnet152-32-squeezenet1_1-2': 0,
 'resnet34-16-resnet34-16-resnet34-8': 1,
 'densenet121-2-densenet161-8-resnet50-16': 1,
 'vgg11-32-vgg13-2-vgg19-2': 1,
 'mobilenet_v3_small-16-squeezenet1_0-16-vgg19-8': 0,
 'resnet34-4-squeezenet1_1-8-vgg16-16': 0,
 'vgg13-4-densenet169-16-efficientnet_b6-4': 1,
 'efficientnet_b6-8-efficientnet_b7-4-squeezenet1_1-8': 0,
 'resnet18-32-squeezenet1_0-2-vgg11-4': 1,
 'efficientnet_b5-32-efficientnet_b6-32-inception_v3-4': 0,
 'densenet121-8-mobilenet_v3_large-16-vgg19-8': 0,
 'mobilenet_v2-4-mobilenet_v3_large-8-resnet152-16': 1,
 'resnet101-8-resnet18-32-squeezenet1_0-8': 1,
 'mobilenet_v3_large-32-alexnet-4-squeezenet1_1-8': 1,
 'efficientnet_b6-2-resnet152-4-squeezenet1_1-4': 1,
 'mobilenet_v3_small-2-resnet101-2-squeezenet1_0-8': 1,
 'densenet169-2-vgg13-16-vgg13-8': 0,
 'efficientnet_b7-2-resnet18-8-squeezenet1_0-8': 1,
 'densenet161-4-inception_v3-4-vgg19-16': 1,
 'efficientnet_b6-32-resnet152-16-vgg16-8': 0,
 'efficientnet_b5-32-efficientnet_b5-32-squeezenet1_0-4': 1,
 'vgg16-4-inception_v3-16-squeezenet1_1-8': 0,
 'densenet121-2-densenet161-2-resnet34-8': 1,
 'efficientnet_b5-2-resnet50-32-vgg19-32': 1,
 'resnet34-32-densenet161-2-vgg16-2': 1,
 'efficientnet_b7-16-mobilenet_v3_large-2-resnet18-16': 0,
 'densenet161-4-densenet169-4-squeezenet1_0-32': 1,
 'densenet161-8-densenet169-4-resnet152-4': 0,
 'alexnet-16-densenet169-2-vgg16-32': 0,
 'mobilenet_v2-8-resnet34-16-squeezenet1_0-16': 1,
 'vgg16-4-densenet169-4-resnet34-32': 1,
 'vgg11-4-vgg13-4-vgg19-16': 1,
 'alexnet-8-efficientnet_b5-2-resnet101-2': 1,
 'mobilenet_v2-4-mobilenet_v3_large-2-vgg11-4': 0,
 'efficientnet_b7-2-mobilenet_v3_large-8-vgg19-2': 1,
 'densenet169-16-resnet50-2-vgg19-2': 1,
 'densenet161-2-resnet101-4-resnet101-4': 1,
 'inception_v3-2-vgg13-2-vgg16-16': 1,
 'mobilenet_v3_large-16-mobilenet_v3_small-32-vgg16-32': 0,
 'efficientnet_b5-8-mobilenet_v2-2-vgg11-16': 1,
 'densenet121-4-densenet121-8-efficientnet_b5-16': 1,
 'densenet169-16-resnet152-2-resnet50-8': 1,
 'alexnet-32-densenet201-16-resnet18-16': 1,
 'densenet161-4-efficientnet_b5-4-vgg13-16': 1,
 'mobilenet_v3_large-2-mobilenet_v3_large-8-vgg11-2': 1,
 'densenet121-4-efficientnet_b7-8-mobilenet_v2-32': 1,
 'efficientnet_b5-2-resnet18-32-vgg16-4': 1,
 'densenet201-8-mobilenet_v3_large-16-vgg13-16': 0,
 'alexnet-8-efficientnet_b5-4-mobilenet_v2-4': 0,
 'mobilenet_v3_small-2-resnet50-16-squeezenet1_0-4': 1,
 'resnet101-16-densenet161-4-efficientnet_b5-4': 1,
 'densenet201-4-mobilenet_v3_large-16-squeezenet1_0-32': 1,
 'alexnet-8-inception_v3-2-resnet18-16': 1,
 'densenet121-32-mobilenet_v3_large-8-squeezenet1_0-16': 0,
 'mobilenet_v3_large-32-squeezenet1_0-16-vgg13-16': 0,
 'efficientnet_b5-32-efficientnet_b7-16-inception_v3-4': 0,
 'resnet101-4-resnet101-4-vgg11-16': 1,
 'alexnet-8-densenet169-16-efficientnet_b5-8': 0,
 'resnet101-4-resnet34-8-vgg11-32': 0,
 'resnet152-16-resnet152-16-vgg16-8': 1,
 'efficientnet_b5-2-inception_v3-32-vgg11-8': 1,
 'densenet161-16-efficientnet_b5-32-resnet18-4': 0,
 'alexnet-16-resnet34-8-vgg19-2': 1,
 'densenet201-8-efficientnet_b5-8-mobilenet_v3_large-8': 0,
 'densenet121-8-squeezenet1_1-8-squeezenet1_1-8': 1,
 'efficientnet_b5-8-squeezenet1_1-32-vgg19-32': 0,
 'resnet50-32-densenet201-4-resnet101-8': 1,
 'resnet152-8-resnet34-4-vgg13-2': 1,
 'densenet169-4-squeezenet1_0-32-squeezenet1_1-4': 1,
 'efficientnet_b6-32-vgg16-2-vgg19-16': 0,
 'mobilenet_v3_large-4-mobilenet_v3_large-4-mobilenet_v3_small-16': 1,
 'densenet169-16-densenet201-8-inception_v3-16': 0,
 'efficientnet_b7-16-resnet152-2-vgg13-16': 0,
 'densenet201-2-mobilenet_v2-32-resnet34-8': 1,
 'vgg11-16-alexnet-4-efficientnet_b6-2': 0,
 'densenet161-8-mobilenet_v2-8-mobilenet_v2-8': 0,
 'alexnet-4-resnet101-32-vgg16-32': 0,
 'densenet121-16-efficientnet_b5-4-squeezenet1_1-2': 0,
 'densenet201-32-efficientnet_b5-8-vgg19-16': 0,
 'resnet34-8-resnet50-4-vgg13-2': 1,
 'efficientnet_b6-8-inception_v3-32-squeezenet1_0-4': 1,
 'densenet161-4-efficientnet_b5-32-resnet18-2': 0,
 'efficientnet_b5-32-squeezenet1_0-16-vgg11-32': 0,
 'densenet201-4-resnet50-8-vgg16-4': 1,
 'efficientnet_b7-8-resnet101-8-vgg19-16': 1,
 'resnet34-16-vgg11-4-vgg13-16': 0,
 'densenet121-8-resnet18-16-resnet34-8': 1,
 'alexnet-32-resnet18-8-squeezenet1_1-2': 0,
 'efficientnet_b7-2-resnet34-16-resnet34-32': 1,
 'resnet152-32-resnet50-2-squeezenet1_0-4': 0,
 'densenet121-8-densenet161-2-vgg19-16': 1,
 'efficientnet_b6-8-mobilenet_v3_small-32-squeezenet1_0-8': 1,
 'efficientnet_b5-32-mobilenet_v3_small-4-resnet34-8': 0,
 'densenet161-4-vgg13-2-vgg16-16': 1,
 'mobilenet_v3_small-2-vgg16-4-vgg16-8': 1,
 'efficientnet_b6-16-resnet50-16-vgg13-4': 0,
 'densenet201-8-inception_v3-2-resnet34-16': 1,
 'efficientnet_b6-4-efficientnet_b6-8-vgg16-4': 1,
 'densenet121-4-squeezenet1_0-32-vgg13-32': 1,
 'inception_v3-4-squeezenet1_1-2-vgg19-4': 0,
 'alexnet-32-densenet161-16-resnet34-16': 0,
 'efficientnet_b7-4-resnet34-2-vgg11-8': 1,
 'densenet169-2-efficientnet_b7-8-vgg19-4': 1,
 'densenet169-32-resnet101-4-vgg13-32': 1,
 'efficientnet_b5-2-efficientnet_b7-2-resnet152-2': 0,
 'densenet169-16-inception_v3-8-resnet50-4': 0,
 'densenet169-32-resnet34-32-vgg19-2': 1,
 'densenet161-2-inception_v3-16-resnet50-2': 1,
 'densenet121-16-resnet101-2-squeezenet1_1-8': 0,
 'resnet152-16-squeezenet1_1-32-squeezenet1_1-8': 1,
 'densenet169-8-mobilenet_v2-4-resnet18-4': 0,
 'densenet169-8-efficientnet_b5-8-vgg11-4': 1,
 'densenet121-8-mobilenet_v3_small-16-squeezenet1_0-4': 1,
 'resnet50-16-resnet50-8-squeezenet1_1-32': 1,
 'mobilenet_v2-4-resnet18-8-vgg19-8': 0,
 'densenet121-4-densenet201-2-resnet34-2': 0,
 'densenet121-2-mobilenet_v3_small-32-squeezenet1_0-4': 1,
 'densenet201-32-inception_v3-2-resnet18-2': 0,
 'densenet161-32-densenet169-8-vgg16-4': 1,
 'mobilenet_v3_small-8-efficientnet_b7-2-inception_v3-4': 0,
 'efficientnet_b5-4-efficientnet_b7-4-resnet152-16': 1,
 'efficientnet_b5-16-mobilenet_v3_small-32-vgg11-32': 0,
 'efficientnet_b7-32-resnet50-16-squeezenet1_0-2': 0,
 'efficientnet_b5-4-efficientnet_b7-8-vgg11-2': 1,
 'densenet121-16-densenet201-4-efficientnet_b6-32': 0,
 'densenet169-2-resnet34-32-squeezenet1_0-16': 1,
 'densenet161-16-efficientnet_b5-8-mobilenet_v3_small-16': 0,
 'densenet161-32-mobilenet_v3_large-32-mobilenet_v3_large-32': 0,
 'densenet201-8-mobilenet_v2-2-squeezenet1_1-8': 0,
 'efficientnet_b6-4-efficientnet_b7-8-resnet50-2': 0,
 'densenet121-8-inception_v3-16-vgg11-32': 1,
 'densenet121-32-densenet169-32-inception_v3-4': 0,
 'resnet101-2-resnet18-2-vgg13-8': 1,
 'efficientnet_b6-8-vgg11-32-vgg16-4': 1,
 'densenet121-32-efficientnet_b6-32-vgg11-32': 0,
 'efficientnet_b6-2-inception_v3-32-resnet18-16': 1,
 'vgg19-32-densenet201-32-resnet34-4': 0,
 'squeezenet1_0-32-vgg11-8-vgg16-2': 1,
 'efficientnet_b6-16-efficientnet_b7-8-vgg13-16': 1,
 'inception_v3-16-mobilenet_v2-8-resnet34-16': 1,
 'densenet161-32-densenet169-16-efficientnet_b5-2': 0,
 'mobilenet_v2-32-densenet201-16-mobilenet_v3_small-32': 0,
 'resnet101-8-resnet50-8-squeezenet1_1-8': 1,
 'efficientnet_b5-32-resnet101-8-resnet50-16': 0,
 'alexnet-4-squeezenet1_0-8-vgg13-2': 1,
 'vgg11-32-vgg16-32-vgg19-16': 1,
 'efficientnet_b7-32-mobilenet_v3_large-2-vgg11-32': 0,
 'densenet169-16-efficientnet_b6-4-resnet50-16': 1,
 'efficientnet_b6-16-resnet50-16-squeezenet1_0-8': 0,
 'densenet169-4-mobilenet_v3_large-16-squeezenet1_0-8': 1,
 'densenet201-2-mobilenet_v2-4-mobilenet_v3_large-8': 1,
 'mobilenet_v3_large-2-resnet50-2-vgg13-16': 1,
 'densenet169-32-inception_v3-8-squeezenet1_0-8': 0,
 'resnet152-2-resnet18-16-vgg11-32': 1,
 'inception_v3-16-resnet18-2-vgg16-2': 1,
 'densenet169-16-densenet169-2-vgg11-4': 1,
 'efficientnet_b7-4-mobilenet_v3_large-16-vgg16-16': 1,
 'efficientnet_b7-8-mobilenet_v3_small-32-squeezenet1_1-16': 1,
 'mobilenet_v2-2-resnet18-2-vgg13-8': 0,
 'efficientnet_b6-32-resnet101-32-squeezenet1_0-2': 0,
 'alexnet-4-resnet18-16-vgg11-16': 1,
 'efficientnet_b5-2-inception_v3-2-mobilenet_v2-4': 1,
 'efficientnet_b6-2-mobilenet_v3_large-4-vgg16-8': 1,
 'mobilenet_v2-16-resnet152-4-vgg19-2': 1,
 'inception_v3-8-mobilenet_v3_large-16-vgg16-2': 1,
 'efficientnet_b6-8-mobilenet_v3_large-4-resnet34-2': 0,
 'alexnet-32-efficientnet_b6-8-resnet101-16': 1,
 'resnet152-4-resnet18-32-squeezenet1_0-16': 1,
 'resnet101-32-resnet50-8-squeezenet1_0-16': 0,
 'alexnet-16-alexnet-2-efficientnet_b6-16': 0,
 'alexnet-2-inception_v3-16-squeezenet1_1-4': 1,
 'mobilenet_v2-2-resnet34-8-vgg19-8': 1,
 'densenet169-8-vgg19-32-vgg19-4': 0,
 'densenet161-16-resnet34-8-vgg11-8': 0,
 'efficientnet_b6-4-inception_v3-4-vgg11-32': 1,
 'densenet201-2-efficientnet_b6-4-resnet18-32': 1,
 'mobilenet_v2-4-resnet152-4-vgg11-8': 1,
 'alexnet-8-resnet18-32-vgg11-8': 1,
 'efficientnet_b6-2-inception_v3-16-resnet50-4': 1,
 'resnet152-16-resnet34-16-squeezenet1_0-8': 0,
 'mobilenet_v2-4-mobilenet_v3_small-16-resnet34-32': 1,
 'mobilenet_v3_small-32-resnet34-2-vgg19-8': 0,
 'resnet101-32-efficientnet_b7-16-squeezenet1_0-4': 0,
 'densenet201-2-resnet101-4-vgg13-2': 1,
 'densenet121-2-inception_v3-16-vgg13-32': 1,
 'squeezenet1_0-16-vgg11-8-vgg13-4': 0,
 'densenet161-16-densenet169-8-vgg11-16': 0,
 'inception_v3-8-vgg13-2-vgg13-32': 0,
 'densenet161-2-efficientnet_b7-32-resnet101-2': 1,
 'alexnet-8-densenet161-4-vgg11-4': 0,
 'resnet152-8-vgg11-8-vgg19-16': 1,
 'efficientnet_b7-16-mobilenet_v3_large-8-vgg16-32': 0,
 'resnet101-4-squeezenet1_0-32-vgg19-16': 1,
 'densenet161-4-densenet201-2-vgg13-2': 1,
 'alexnet-16-inception_v3-2-squeezenet1_1-16': 1,
 'efficientnet_b6-16-mobilenet_v3_small-8-resnet152-2': 0,
 'alexnet-32-inception_v3-2-vgg11-32': 0,
 'resnet50-2-squeezenet1_0-16-squeezenet1_1-32': 1,
 'densenet201-16-inception_v3-32-resnet18-8': 1,
 'densenet121-32-densenet161-8-resnet18-2': 0,
 'densenet161-32-vgg13-4-vgg16-32': 0,
 'densenet121-2-densenet169-8-efficientnet_b7-4': 1,
 'alexnet-2-densenet121-8-efficientnet_b5-2': 1,
 'densenet161-16-mobilenet_v3_large-4-vgg19-8': 0,
 'resnet101-16-resnet18-16-vgg11-16': 0,
 'resnet50-16-densenet121-8-resnet34-4': 1,
 'inception_v3-8-resnet34-16-resnet50-2': 1,
 'densenet169-32-resnet101-4-resnet34-2': 0,
 'inception_v3-4-inception_v3-8-resnet18-8': 1,
 'densenet121-8-mobilenet_v3_small-16-squeezenet1_0-16': 1,
 'mobilenet_v2-32-mobilenet_v2-4-vgg13-2': 1,
 'densenet169-8-vgg16-16-vgg19-2': 1,
 'resnet101-32-inception_v3-2-resnet34-8': 1,
 'densenet169-16-mobilenet_v3_small-8-resnet34-8': 0,
 'mobilenet_v2-16-resnet152-32-squeezenet1_0-8': 0,
 'mobilenet_v3_small-16-mobilenet_v3_small-32-vgg13-8': 0,
 'densenet121-16-resnet34-16-squeezenet1_1-16': 1,
 'densenet161-4-mobilenet_v2-32-resnet152-16': 1,
 'efficientnet_b5-32-squeezenet1_1-16-vgg13-32': 0,
 'resnet101-8-resnet152-32-vgg16-32': 1,
 'densenet161-2-resnet152-2-resnet18-4': 1,
 'densenet169-32-densenet169-32-vgg13-4': 1,
 'resnet101-2-resnet18-16-squeezenet1_0-8': 1,
 'densenet161-16-densenet121-4-inception_v3-2': 1,
 'densenet121-32-resnet50-8-squeezenet1_1-16': 0,
 'efficientnet_b5-4-resnet50-16-vgg16-2': 1,
 'densenet121-32-efficientnet_b5-8-resnet18-16': 0,
 'inception_v3-2-squeezenet1_0-32-vgg16-32': 0,
 'efficientnet_b6-2-resnet101-2-vgg16-32': 1,
 'vgg13-8-alexnet-2-resnet34-8': 1,
 'alexnet-2-densenet161-16-resnet34-2': 1,
 'alexnet-8-efficientnet_b5-2-resnet101-8': 0,
 'densenet161-16-mobilenet_v3_large-4-resnet34-2': 0,
 'inception_v3-4-resnet101-32-resnet34-32': 1,
 'alexnet-2-densenet161-4-densenet169-8': 1,
 'densenet161-2-densenet161-32-efficientnet_b7-16': 1,
 'alexnet-2-resnet101-8-squeezenet1_1-32': 1,
 'densenet121-32-mobilenet_v3_large-32-resnet50-32': 0,
 'mobilenet_v3_large-2-resnet152-4-vgg19-2': 1,
 'mobilenet_v2-4-mobilenet_v3_large-16-vgg11-2': 1,
 'efficientnet_b7-16-squeezenet1_0-4-vgg19-2': 1,
 'efficientnet_b6-4-resnet50-8-squeezenet1_0-16': 1,
 'resnet18-2-squeezenet1_1-4-vgg13-2': 1,
 'resnet18-32-resnet34-4-squeezenet1_1-32': 0,
 'densenet201-8-squeezenet1_0-32-vgg13-8': 1,
 'densenet161-2-resnet18-2-vgg13-32': 0,
 'alexnet-4-mobilenet_v3_large-8-squeezenet1_1-2': 0,
 'densenet201-32-densenet121-8-vgg11-16': 1,
 'densenet169-2-squeezenet1_0-32-vgg13-16': 1,
 'alexnet-32-mobilenet_v3_small-2-resnet18-2': 0,
 'inception_v3-2-mobilenet_v3_large-4-squeezenet1_1-32': 1,
 'densenet121-4-efficientnet_b5-2-resnet101-8': 1,
 'mobilenet_v2-8-resnet18-32-resnet50-8': 1,
 'densenet161-8-densenet169-4-mobilenet_v3_small-16': 1,
 'densenet201-4-efficientnet_b5-2-vgg16-8': 1,
 'densenet201-2-densenet201-8-squeezenet1_0-16': 1,
 'efficientnet_b6-16-efficientnet_b7-2-squeezenet1_0-2': 0,
 'inception_v3-16-squeezenet1_1-2-squeezenet1_1-4': 0,
 'mobilenet_v3_large-8-mobilenet_v3_small-2-resnet34-2': 0,
 'mobilenet_v2-32-resnet50-16-squeezenet1_1-2': 0,
 'inception_v3-8-mobilenet_v3_small-8-vgg16-16': 0,
 'densenet121-4-squeezenet1_0-16-vgg11-16': 1,
 'alexnet-16-vgg11-32-vgg19-4': 1,
 'efficientnet_b5-32-resnet101-2-resnet34-4': 0,
 'squeezenet1_0-8-vgg11-32-vgg11-4': 0,
 'efficientnet_b7-4-efficientnet_b5-2-resnet101-2': 0,
 'densenet161-32-resnet18-32-resnet18-4': 0,
 'mobilenet_v3_small-32-resnet152-4-squeezenet1_0-8': 1,
 'resnet50-32-alexnet-2-vgg13-4': 0,
 'densenet201-2-resnet101-8-resnet50-4': 1,
 'densenet201-2-resnet152-16-vgg16-32': 1,
 'inception_v3-16-squeezenet1_1-2-vgg16-8': 0,
 'densenet161-8-densenet169-16-resnet18-4': 0,
 'efficientnet_b7-32-resnet152-8-resnet34-2': 0,
 'efficientnet_b7-4-resnet18-4-resnet34-2': 0,
 'mobilenet_v3_large-8-resnet18-4-squeezenet1_0-16': 1,
 'densenet161-2-efficientnet_b5-4-resnet152-16': 1,
 'alexnet-8-densenet121-4-vgg13-8': 0,
 'densenet161-4-squeezenet1_0-4-vgg11-2': 1,
 'densenet169-32-efficientnet_b6-4-mobilenet_v3_large-2': 0,
 'resnet101-2-squeezenet1_0-2-vgg11-32': 1,
 'inception_v3-4-squeezenet1_0-32-squeezenet1_0-4': 1,
 'inception_v3-4-squeezenet1_0-16-vgg11-4': 1,
 'densenet169-8-efficientnet_b5-2-resnet50-2': 0,
 'efficientnet_b7-4-efficientnet_b7-4-vgg16-16': 1,
 'densenet161-32-resnet101-2-resnet152-32': 0,
 'densenet121-4-resnet18-8-vgg13-4': 1,
 'inception_v3-8-mobilenet_v3_large-16-mobilenet_v3_small-32': 1,
 'alexnet-32-vgg13-4-vgg19-4': 1,
 'densenet161-16-mobilenet_v2-2-vgg19-32': 0,
 'resnet18-4-squeezenet1_0-8-vgg16-32': 0,
 'efficientnet_b7-4-resnet152-4-squeezenet1_1-4': 0,
 'densenet169-32-efficientnet_b6-2-squeezenet1_1-4': 0,
 'resnet18-4-resnet34-8-vgg19-32': 0,
 'resnet18-16-resnet18-8-vgg19-16': 0,
 'vgg19-4-resnet101-8-resnet34-8': 0,
 'densenet201-8-squeezenet1_0-16-vgg13-16': 1,
 'densenet161-32-squeezenet1_1-4-vgg19-16': 0,
 'densenet121-8-densenet169-2-vgg16-2': 0,
 'densenet201-2-mobilenet_v3_small-32-resnet101-4': 1,
 'alexnet-32-efficientnet_b6-2-efficientnet_b7-2': 1}


def preprocess_data(pattern: str, size: int):
    if pattern == 'closed':
        df = pd.read_csv(f'./data/raw-data/tb-{pattern}-observed-size-{size}.csv')
        if size == 3:
            filtered_df = df[df['job_mix'].isin(job_size_3_preds)]
            filtered_df['total_tput'] = filtered_df['tput_m1'] + filtered_df['tput_m2'] + filtered_df['tput_m3']
            filtered_df['max_total_p90_latency'] = filtered_df[['total_p90_m1', 'total_p90_m2', 'total_p90_m3']].apply(max, axis=1)
            filtered_df['max_total_p99_latency'] = filtered_df[['total_p99_m1', 'total_p99_m2', 'total_p99_m3']].apply(max, axis=1)
            filtered_df['system'] = filtered_df['mode']

            new_rows = []
            count = 0
            for key in job_size_3_preds:
                if job_size_3_preds[key] == 1:
                    row = filtered_df[(filtered_df['job_mix'] == key) & (filtered_df['mode'] == 'mps-uncap')].iloc[0].to_list()
                    row[-1] = 'tiebreaker'
                else:
                    row = filtered_df[(filtered_df['job_mix'] == key) & (filtered_df['mode'] == 'mig')].iloc[0].to_list()
                    row[-1] = 'tiebreaker'
                new_rows.append(row)

            for key in job_size_3_targets:
                if job_size_3_targets[key] == 1:
                    row = filtered_df[(filtered_df['job_mix'] == key) & (filtered_df['mode'] == 'mps-uncap')].iloc[0].to_list()
                    row[-1] = 'oracle'
                else:
                    row = filtered_df[(filtered_df['job_mix'] == key) & (filtered_df['mode'] == 'mig')].iloc[0].to_list()
                    row[-1] = 'oracle'
                new_rows.append(row)

            append_df = pd.DataFrame(new_rows, columns=filtered_df.columns.tolist())
            df_final = pd.concat([filtered_df, append_df])
    
    # Drop job mixes that have SLO violations in both MPS and MIG
    groups = df_final.groupby('job_mix')
    job_mixes_to_drop = []
    for name, group in groups:
        if all(group['slo_violation_no'] > 0):
            job_mixes_to_drop.append(name)
    df_final = df_final[~df_final['job_mix'].isin(job_mixes_to_drop)].reset_index(drop=True)

    # Sort data: (1) mig winner first, then mps, and (2) sort mig winners in increasing throughput, decreasing for mps winner
    # df_oracle = df_final[df_final['system'] == 'oracle']
    # df_mig = df_oracle[df_oracle['mode'] == 'mig'].sort_values(by='total_tput')
    # df_mps = df_oracle[df_oracle['mode'] == 'mps-uncap'].sort_values(by='total_tput', ascending=False)
    # df_merged = pd.concat([df_mig, df_mps])
    # df_merged['job_mix_id'] = range(1, len(df_merged) + 1)
    # job_mix_id_dict = dict(zip(df_merged['job_mix'], df_merged['job_mix_id']))
    # df_final['job_mix_id'] = df_final['job_mix'].map(job_mix_id_dict)
    return df_final

def e2e_plot_split(df: pd.DataFrame):
    df = df.sort_values(by=['job_mix_id'])
    job_mix_id = df[df['system'] == 'oracle']['job_mix_id'].tolist()
    
    # Get metrics as lists to manually create seaborn lines
    # oracle_tput = df[df['system'] == 'oracle']['total_tput'].tolist()
    tiebreaker_tput = df[df['system'] == 'tiebreaker']['total_tput'].tolist()
    mig_tput = df[df['system'] == 'mig']['total_tput'].tolist()
    mps_tput = df[df['system'] == 'mps-uncap']['total_tput'].tolist()
    
    # oracle_p99_lat = df[df['system'] == 'oracle']['max_total_p99_latency'].tolist()
    tiebreaker_p99_lat = df[df['system'] == 'tiebreaker']['max_total_p99_latency'].tolist()
    mig_p99_lat = df[df['system'] == 'mig']['max_total_p99_latency'].tolist()
    mps_p99_lat = df[df['system'] == 'mps-uncap']['max_total_p99_latency'].tolist()    

    # oracle_p90_lat = df[df['system'] == 'oracle']['max_total_p90_latency'].tolist()
    tiebreaker_p90_lat = df[df['system'] == 'tiebreaker']['max_total_p90_latency'].tolist()
    mig_p90_lat = df[df['system'] == 'mig']['max_total_p90_latency'].tolist()
    mps_p90_lat = df[df['system'] == 'mps-uncap']['max_total_p90_latency'].tolist()
    
    plt.figure(figsize=(6, 2)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p1 = sns.lineplot(x=job_mix_id, y=mps_tput, label='MPS', alpha=0.8, zorder=1)
    sns.lineplot(x=job_mix_id, y=mig_tput, label='MISOServe++', alpha=0.8, zorder=1)
    sns.lineplot(x=job_mix_id, y=tiebreaker_tput, label='TieBreaker', alpha=1, zorder=3)
    p1.grid(axis = "x")
    # sns.lineplot(x=job_mix_id, y=oracle_tput, label='Oracle', alpha=1, zorder=4, linestyle='dashed')
    plt.xlabel('Job Mix')
    plt.ylabel('Total GPU Throughput\n(images/second)')
    plt.axvline(x=334, color='black', linestyle='--')
    plt.text(320, plt.ylim()[1] * 0.85, "Oracle Chose:", ha='right', color='black')
    plt.text(240, plt.ylim()[1] * 0.75, "MIG", ha='right', color='black')
    plt.text(350, plt.ylim()[1] * 0.85, "Oracle Chose:", ha='left', color='black')
    plt.text(430, plt.ylim()[1] * 0.75, "MPS", ha='left', color='black')
    leg = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.savefig('./plots/system-tput-e2e.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 2)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p2 = sns.lineplot(x=job_mix_id, y=mps_p99_lat, label='MPS', alpha=0.8, zorder=1, linewidth=1.4)
    sns.lineplot(x=job_mix_id, y=mig_p99_lat, label='MISOServe++', alpha=0.8, zorder=2, linewidth=1.4)
    sns.lineplot(x=job_mix_id, y=tiebreaker_p99_lat, label='TieBreaker', alpha=1, zorder=3, linewidth=0.6)
    p2.grid(axis = "x")
    # sns.lineplot(x=job_mix_id, y=oracle_p99_lat, label='Oracle', alpha=1, zorder=3)
    plt.xlabel('Job Mix')
    plt.ylabel('P99 Latency (ms)')
    plt.axvline(x=334, color='black', linestyle='--')
    plt.text(320, plt.ylim()[1] * 0.85, "Oracle Chose:", ha='right', color='black')
    plt.text(240, plt.ylim()[1] * 0.75, "MIG", ha='right', color='black')
    plt.text(350, plt.ylim()[1] * 0.85, "Oracle Chose:", ha='left', color='black')
    plt.text(430, plt.ylim()[1] * 0.75, "MPS", ha='left', color='black')
    leg = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.savefig('./plots/system-total-p99-e2e.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 2)) 
    p3 = sns.lineplot(x=job_mix_id, y=mps_p90_lat, label='MPS', alpha=0.8, zorder=1, linewidth=1.4)
    sns.lineplot(x=job_mix_id, y=mig_p90_lat, label='MISOServe++', alpha=0.8, zorder=2, linewidth=1.4)
    sns.lineplot(x=job_mix_id, y=tiebreaker_p90_lat, label='TieBreaker', alpha=1, zorder=4, linewidth=0.6)
    p3.grid(axis = "x")
    # sns.lineplot(x=job_mix_id, y=oracle_p90_lat, label='Oracle', alpha=1, zorder=3)
    plt.xlabel('Job Mix')
    plt.ylabel('P90 Latency (ms)')
    plt.axvline(x=334, color='black', linestyle='--')
    plt.text(320, plt.ylim()[1] * 0.85, "Oracle Chose:", ha='right', color='black')
    plt.text(240, plt.ylim()[1] * 0.75, "MIG", ha='right', color='black')
    plt.text(350, plt.ylim()[1] * 0.85, "Oracle Chose:", ha='left', color='black')
    plt.text(430, plt.ylim()[1] * 0.75, "MPS", ha='left', color='black')
    lge = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.savefig('./plots/system-total-p90-e2e.png', bbox_inches='tight', dpi=300)
    plt.close()

def summary_statistics_plot(df):
    tput_percent_increases = []
    tput_x_increases = []
    p99_percent_decreases = []
    p99_x_decreases = []
    p90_percent_decreases = []
    p90_x_decreases = []
    throughput_cases = {'case_1': [], 'case_2': [], 'case_3': [], 'case_4': [], 'case_5': []}
    no_mispredictions = 0
    for job_mix in df['job_mix'].unique():
        sub_df = df[df['job_mix'] == job_mix]
        
        # Get tput data per mechanism
        tiebreaker_tput = sub_df[sub_df['system'] == 'tiebreaker']['total_tput'].tolist()[0]
        mig_tput = sub_df[sub_df['system'] == 'mig']['total_tput'].tolist()[0]
        mps_tput = sub_df[sub_df['system'] == 'mps-uncap']['total_tput'].tolist()[0]
        tput_percent_increase_mig = round((tiebreaker_tput - mig_tput) / mig_tput * 100, 2)
        tput_percent_increase_mps = round((tiebreaker_tput - mps_tput) / mps_tput * 100, 2)
        tput_x_increase_mig = round(tiebreaker_tput / mig_tput, 2)
        tput_x_increase_mps = round(tiebreaker_tput / mps_tput, 2)

        # Get p90 latency data per mechanism
        tiebreaker_p90_latency = sub_df[sub_df['system'] == 'tiebreaker']['max_total_p90_latency'].tolist()[0]
        mig_p90_latency = sub_df[sub_df['system'] == 'mig']['max_total_p90_latency'].tolist()[0]
        mps_p90_latency = sub_df[sub_df['system'] == 'mps-uncap']['max_total_p90_latency'].tolist()[0]
        p90_latency_percent_decrease_mig = round((mig_p90_latency - tiebreaker_p90_latency) / mig_p90_latency * 100, 2)
        p90_latency_percent_decrease_mps = round((mps_p90_latency - tiebreaker_p90_latency) / mps_p90_latency * 100, 2)
        p90_latency_x_decrease_mig = round(mig_p90_latency / tiebreaker_p90_latency, 2)
        p90_latency_x_decrease_mps = round(mps_p90_latency / tiebreaker_p90_latency, 2)

        # Get p99 latency data per mechanism
        tiebreaker_p99_latency = sub_df[sub_df['system'] == 'tiebreaker']['max_total_p99_latency'].tolist()[0]
        mig_p99_latency = sub_df[sub_df['system'] == 'mig']['max_total_p99_latency'].tolist()[0]
        mps_p99_latency = sub_df[sub_df['system'] == 'mps-uncap']['max_total_p99_latency'].tolist()[0]
        p99_latency_percent_decrease_mig = round((mig_p99_latency - tiebreaker_p99_latency) / mig_p99_latency * 100, 2)
        p99_latency_percent_decrease_mps = round((mps_p99_latency - tiebreaker_p99_latency) / mps_p99_latency * 100, 2)
        p99_latency_x_decrease_mig = round(tiebreaker_p99_latency / mig_p99_latency, 2)
        p99_latency_x_decrease_mps = round(tiebreaker_p99_latency / mps_p99_latency, 2)

        if tput_x_increase_mig == 1:
            tput_percent_increases.append(tput_percent_increase_mps)
            tput_x_increases.append(tput_x_increase_mps)
            p90_percent_decreases.append(p90_latency_percent_decrease_mps)
            if p90_latency_percent_decrease_mps == -1164.67:
                print(sub_df[['job_mix', 'mode', 'total_p100_m1', 'total_p100_m2', 'total_p100_m3', 'slo_m1', 'slo_m2', 'slo_m3', 'slo_violation_no', 'system', 'total_tput', 'max_total_p90_latency', 'max_total_p99_latency']].to_string())
            p90_x_decreases.append(p90_latency_x_decrease_mps)
            p99_percent_decreases.append(p99_latency_percent_decrease_mps)
            p99_x_decreases.append(p99_latency_x_decrease_mps)
        else:
            tput_percent_increases.append(tput_percent_increase_mig)
            tput_x_increases.append(tput_x_increase_mig)
            p90_percent_decreases.append(p90_latency_percent_decrease_mig)
            if p90_latency_percent_decrease_mig == -1164.67:
                print(sub_df[['job_mix', 'mode', 'total_p100_m1', 'total_p100_m2', 'total_p100_m3', 'slo_m1', 'slo_m2', 'slo_m3', 'slo_violation_no', 'system', 'total_tput', 'max_total_p90_latency', 'max_total_p99_latency']].to_string())
            p90_x_decreases.append(p90_latency_x_decrease_mig)
            p99_percent_decreases.append(p99_latency_percent_decrease_mig)
            p99_x_decreases.append(p99_latency_x_decrease_mig)
        
        # Accurate prediction
        if sub_df[sub_df['system'] == 'tiebreaker']['mode'].tolist()[0] == sub_df[sub_df['system'] == 'oracle']['mode'].tolist()[0]:
            correct_mode = sub_df[sub_df['system'] == 'tiebreaker']['mode'].tolist()[0]
            other_mode = 'mps-uncap'
            other_tput = mps_tput
            if correct_mode == 'mps-uncap':
                other_mode = 'mig'
                other_tput = mig_tput
            
            # TieBreaker decreased throughput
            if tiebreaker_tput < other_tput:
                tiebreaker_slo_no = sub_df[sub_df['system'] == 'tiebreaker']['slo_violation_no'].tolist()[0]
                other_slo_no = sub_df[sub_df['system'] == other_mode]['slo_violation_no'].tolist()[0]
                
                if tiebreaker_slo_no == 0:
                    # Case 1: TieBreaker no SLO violation, other mechanism SLO violation
                    if other_slo_no > 0:
                        throughput_cases['case_1'].append(other_tput / tiebreaker_tput)
                    # Case 2: TieBreaker no SLO violation, other mechanism no SLO violation
                    else:
                        throughput_cases['case_2'].append(other_tput / tiebreaker_tput)
                else:
                    # Case 3: TieBreaker SLO violation, other mechanism no SLO violation
                    if other_slo_no == 0:
                        throughput_cases['case_3'].append(other_tput / tiebreaker_tput)
                    else:
                        print('Should not end up here')
        else:
            no_mispredictions += 1
            correct_mode = sub_df[sub_df['system'] == 'oracle']['mode'].tolist()[0]
            correct_tput = mig_tput
            if correct_mode == 'mps-uncap':
                correct_tput = mps_tput
            # Case 4: Inaccurate prediction and less throughput
            if tiebreaker_tput < correct_tput:
                throughput_cases['case_4'].append(correct_tput / tiebreaker_tput)
            # Case 5: Inaccurate prediction but not lower throughput
            else:
                throughput_cases['case_5'].append(correct_tput / tiebreaker_tput)


    correct_tput_decreases = len(throughput_cases['case_1'])
    incorrect_tput_decreases = len(throughput_cases['case_4'])


    print(f'Throughput % increase range: {min(tput_percent_increases)} - {max(tput_percent_increases)}%')
    print(f'Throughput average % increase: {mean(tput_percent_increases)}%')
    print(f'Throughput median increase: {median(tput_percent_increases)}%')
    print(f'Throughput increase range: {min(tput_x_increases)} - {max(tput_x_increases)}x')
    print(f'Throughput average increase: {mean(tput_x_increases)}x')
    print(f'Throughput median increase: {median(tput_x_increases)}')
    print()
    print(f'P90 latency % decrease range: {min(p90_percent_decreases)} - {max(p90_percent_decreases)}%')
    print(f'P90 latency average % decrease: {mean(p90_percent_decreases)}%')
    print(f'P90 latency median % decrease: {median(p90_percent_decreases)}%')
    print(f'P90 latecy decrease range: {min(p90_x_decreases)} - {max(p90_x_decreases)}x')
    print(f'P90 latency average decrease: {mean(p90_x_decreases)}x')
    print(f'P90 latency median decrease: {median(p90_x_decreases)}')
    print()
    print(f'P99 latency % decrease range: {min(p99_percent_decreases)} - {max(p99_percent_decreases)}%')
    print(f'P99 latency average % decrease: {mean(p99_percent_decreases)}%')
    print(f'P99 latency median % decrease: {median(p99_percent_decreases)}%')
    print(f'P99 latecy decrease range: {min(p99_x_decreases)} - {max(p99_x_decreases)}x')
    print(f'P99 latency average decrease: {mean(p99_x_decreases)}x')
    print(f'P99 latency median decrease: {median(p99_x_decreases)}')
    print()
    # print(f'Number of misprediction: {no_mispredictions}')
    # print(f'Number of correct throughput decreases: {correct_tput_decreases}')
    # print(f'Number of incorrect throughput decreases: {incorrect_tput_decreases}')

    # Create histogram plots per metric
    plt.figure(figsize=(6, 2)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p1 = sns.histplot(data=tput_x_increases)
    p1.grid(axis = "x")
    plt.axvline(x=0.9, color='black', linestyle='--')

    plt.xlabel('Throughput Increase ([num]x)')
    plt.ylabel('Count')
    plt.savefig('./plots/tput-improvement-histogram.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 2)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p2 = sns.histplot(data=p90_x_decreases)
    p2.grid(axis = "x")
    plt.axvline(x=0.9, color='black', linestyle='--')

    plt.xlabel('P90 Latency Decrease ([num]x)')
    plt.ylabel('Count')
    plt.savefig('./plots/p90-lat-improvement-histogram.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 2)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p3 = sns.histplot(data=p99_x_decreases)
    p3.grid(axis = "x")
    plt.axvline(x=0.9, color='black', linestyle='--')

    plt.xlabel('P99 Latency Decrease ([num]x)')
    plt.ylabel('Count')
    plt.savefig('./plots/p99-lat-improvement-histogram.png', bbox_inches='tight', dpi=300)
    plt.close()

def e2e_plot(df: pd.DataFrame):

    # Throughput plot
    df_tput = df
    df_oracle = df_tput[df_tput['system'] == 'oracle'].sort_values(by='total_tput')
    df_oracle['job_no'] = range(1, len(df_oracle) + 1)
    job_no_dict = dict(zip(df_oracle['job_mix'], df_oracle['job_no']))
    df_tput['job_no'] = df_tput['job_mix'].map(job_no_dict)
    df_tput = df_tput.sort_values(by=['job_no'])

    job_no = df_tput[df_tput['system'] == 'oracle']['job_no'].tolist()
    oracle_tput = df_tput[df_tput['system'] == 'oracle']['total_tput'].tolist()
    tiebreaker_tput = df_tput[df_tput['system'] == 'tiebreaker']['total_tput'].tolist()
    mig_tput = df_tput[df_tput['system'] == 'mig']['total_tput'].tolist()
    mps_tput = df_tput[df_tput['system'] == 'mps-uncap']['total_tput'].tolist()
    
    # tiebreaker_p90_lat = df[df['system'] == 'tiebreaker']['max_total_p90_latency'].tolist()
    # mig_p90_lat = df[df['system'] == 'mig']['max_total_p90_latency'].tolist()
    # mps_p90_lat = df[df['system'] == 'mps-uncap']['max_total_p90_latency'].tolist()
    
    plt.figure(figsize=(6, 2)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p1 = sns.lineplot(x=job_no, y=mps_tput, label='MPS', alpha=0.8, zorder=1)
    sns.lineplot(x=job_no, y=mig_tput, label='MISOServe++', alpha=0.8, zorder=1)
    sns.lineplot(x=job_no, y=tiebreaker_tput, label='TieBreaker', alpha=1, zorder=3)
    sns.lineplot(x=job_no, y=oracle_tput, label='Oracle', alpha=1, zorder=4, linestyle='dotted')
    p1.grid(axis = "x")
    plt.xlabel('Job Number')
    plt.ylabel('Total GPU Throughput\n(images/second)')
    leg = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.savefig('./plots/system-tput-e2e-sorted.png', bbox_inches='tight', dpi=300)
    plt.close()

    # SLO Violation Plot
    tiebreaker_slo_violations = 0
    mps_slo_violations = 0
    mig_slo_violations = 0
    oracle_slo_violations = 0
    for job_mix in df['job_mix'].unique():
        sub_df = df[df['job_mix'] == job_mix]
        if sub_df[sub_df['system'] == 'tiebreaker']['slo_violation_no'].tolist()[0] > 0:
            tiebreaker_slo_violations += 1
        if sub_df[sub_df['system'] == 'mps-uncap']['slo_violation_no'].tolist()[0] > 0:
            mps_slo_violations += 1
        if sub_df[sub_df['system'] == 'mig']['slo_violation_no'].tolist()[0] > 0:
            mig_slo_violations += 1
        if sub_df[sub_df['system'] == 'oracle']['slo_violation_no'].tolist()[0] > 0:
            oracle_slo_violations += 1
    # tiebreaker_slo_violation_ratio = tiebreaker_slo_violations / len(df['job_mix'].unique().tolist()) * 100
    # mps_slo_violation_ratio = mps_slo_violations / len(df['job_mix'].unique().tolist()) * 100
    # mig_slo_violation_ratio = mig_slo_violations / len(df['job_mix'].unique().tolist()) * 100
    # oracle_slo_violation_ratio = oracle_slo_violations / len(df['job_mix'].unique().tolist()) * 100
    list_df = [['TieBreaker', tiebreaker_slo_violations], ['MPS', mps_slo_violations], ['MISOServe++', mig_slo_violations]]
    df_slo_ratio = pd.DataFrame(list_df, columns=['system', 'slo_violations'])
    print(df_slo_ratio)
    plt.figure(figsize=(4, 1.5)) 
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    p1 = sns.barplot(data=df_slo_ratio, x='system', y='slo_violations', width=0.4)
    # p1.set_ylim(0, 50)
    p1.grid(axis="x")
    plt.xlabel('')
    plt.ylabel('# of Job Mixes\nw/ SLO Violations')
    plt.savefig('./plots/system-slo-violation-comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # # P99 Latency plot
    # df_p99_lat = df
    # df_oracle = df_p99_lat[df_p99_lat['system'] == 'oracle'].sort_values(by='max_total_p99_latency')
    # df_oracle['job_no'] = range(1, len(df_oracle) + 1)
    # job_no_dict = dict(zip(df_oracle['job_mix'], df_oracle['job_no']))
    # df_p99_lat['job_no'] = df_p99_lat['job_mix'].map(job_no_dict)
    # df_p99_lat = df_p99_lat.sort_values(by=['job_no'])

    # job_no = df_p99_lat[df_p99_lat['system'] == 'oracle']['job_no'].tolist()
    # oracle_p99_lat = df_p99_lat[df_p99_lat['system'] == 'oracle']['max_total_p99_latency'].tolist()
    # tiebreaker_p99_lat = df_p99_lat[df_p99_lat['system'] == 'tiebreaker']['max_total_p99_latency'].tolist()
    # mig_p99_lat = df_p99_lat[df_p99_lat['system'] == 'mig']['max_total_p99_latency'].tolist()
    # mps_p99_lat = df_p99_lat[df_p99_lat['system'] == 'mps-uncap']['max_total_p99_latency'].tolist()   

    # plt.figure(figsize=(6, 2)) 
    # sns.set_style("whitegrid", {'grid.linestyle': ':'})
    # p2 = sns.lineplot(x=job_no, y=mps_p99_lat, label='MPS', alpha=0.8, zorder=1)
    # sns.lineplot(x=job_no, y=mig_p99_lat, label='MISOServe++', alpha=0.8, zorder=2)
    # sns.lineplot(x=job_no, y=tiebreaker_p99_lat, label='TieBreaker', alpha=1, zorder=3)
    # sns.lineplot(x=job_no, y=oracle_p99_lat, label='Oracle', alpha=1, zorder=3, linestyle='dotted')
    # p2.grid(axis = "x")
    # plt.xlabel('Job Number')
    # plt.ylabel('P99 Latency (ms)')
    # leg = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(2.0)
    # plt.savefig('./plots/system-total-p99-e2e-sorted.png', bbox_inches='tight', dpi=300)
    # plt.close()

    # # P90 Latency Plot
    # df_p90_lat = df
    # df_oracle = df_p90_lat[df_p90_lat['system'] == 'oracle'].sort_values(by='max_total_p90_latency')
    # df_oracle['job_no'] = range(1, len(df_oracle) + 1)
    # job_no_dict = dict(zip(df_oracle['job_mix'], df_oracle['job_no']))
    # df_p90_lat['job_no'] = df_p90_lat['job_mix'].map(job_no_dict)
    # df_p90_lat = df_p90_lat.sort_values(by=['job_no'])

    # job_no = df_p90_lat[df_p90_lat['system'] == 'oracle']['job_no'].tolist()
    # oracle_p90_lat = df_p90_lat[df_p90_lat['system'] == 'oracle']['max_total_p90_latency'].tolist()
    # tiebreaker_p90_lat = df_p90_lat[df_p90_lat['system'] == 'tiebreaker']['max_total_p90_latency'].tolist()
    # mig_p90_lat = df_p90_lat[df_p90_lat['system'] == 'mig']['max_total_p90_latency'].tolist()
    # mps_p90_lat = df_p90_lat[df_p90_lat['system'] == 'mps-uncap']['max_total_p90_latency'].tolist()   

    # plt.figure(figsize=(6, 2)) 
    # sns.set_style("whitegrid", {'grid.linestyle': ':'})
    # p2 = sns.lineplot(x=job_no, y=mps_p90_lat, label='MPS', alpha=0.8, zorder=1)
    # sns.lineplot(x=job_no, y=mig_p90_lat, label='MISOServe++', alpha=0.8, zorder=2)
    # sns.lineplot(x=job_no, y=tiebreaker_p90_lat, label='TieBreaker', alpha=1, zorder=3)
    # sns.lineplot(x=job_no, y=oracle_p90_lat, label='Oracle', alpha=1, zorder=3, linestyle='dotted')
    # p2.grid(axis = "x")
    # plt.xlabel('Job Number')
    # plt.ylabel('P90 Latency (ms)')
    # leg = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center', frameon=False)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(2.0)
    # plt.savefig('./plots/system-total-p90-e2e-sorted.png', bbox_inches='tight', dpi=300)
    # plt.close()


if __name__=='__main__':
    pattern_type = 'closed'
    size = 3
    df = preprocess_data(pattern=pattern_type, size=size)
    e2e_plot(df=df)
    # summary_statistics_plot(df=df)
