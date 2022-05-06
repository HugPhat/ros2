import tensorrt as trt
import numpy as np


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def GiB(val):
    return val * 1 << 30


def ONNX_to_TRT(onnx_model_path=None, trt_engine_path=None, fp16_mode=False, size=8):

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    # increase if can not finsih in few minutes
    config.max_workspace_size = GiB(size)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_model_path, 'rb') as model:
        assert parser.parse(model.read())
        serialized_engine = builder.build_engine(network, config)

    with open(trt_engine_path, 'wb') as f:
        f.write(serialized_engine.serialize())
    

    print('TensorRT file in ' + trt_engine_path)
    print('============ONNX->TensorRT SUCCESS============')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='path to onnx', type=str)
    parser.add_argument('--half', help='FP16',
                        action="store_true", default=False)
    parser.add_argument('--size', help='assign memory size to export engine', type=int, default=8)
    #parser.add_argument()
    args = parser.parse_args()

    onnx_path = args.d
    fp16_trt = args.half

    ONNX_to_TRT(onnx_model_path=onnx_path, trt_engine_path=onnx_path.replace(
        '.onnx', '.trt'), fp16_mode=fp16_trt, size=args.size)
