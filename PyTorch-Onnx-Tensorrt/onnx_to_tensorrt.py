#from __future__ import print_function

import os
import sys
import time

import tensorrt as trt
from PIL import Image
import numpy as np
import torch
import utils.calibrator as calibrator
import argparse

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


def build_engine(onnx_file_path, engine_file_path, verbose=False):
    """Takes an ONNX file and creates a TensorRT engine."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True
        #builder.strict_type_constraints = True

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        if trt.__version__[0] >= '7':
            # The actual yolov3.onnx is generated with batch size 64.
            # Reshape input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape
        print('Completed parsing of ONNX file')

        print('Building an engine; this may take a while...')
        engine = builder.build_cuda_engine(network)
        print('Completed creating engine')
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        return engine


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose output (for debugging)')
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                 'yolov3-tiny-288', 'yolov3-tiny-416'])
    args = parser.parse_args()

    onnx_file_path = '%s.onnx' % args.model
    engine_file_path = '%s.trt' % args.model
    _ = build_engine(onnx_file_path, engine_file_path, args.verbose)


if __name__ == '__main__':
    main()
