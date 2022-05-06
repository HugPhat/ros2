import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import copy
import time

#
from trt_yolov5.yolov5_base import yolov5_base
##


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TrtModel():
    '''
    TensorRT infer
    '''

    def __init__(self, trt_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            #print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def __call__(self, img_np_nchw):
        '''
        TensorRT
        :param img_np_nchw: image
        '''
        self.ctx.push()

        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs[0], img_np_nchw.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size,
                              bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()
        #print(host_outputs[0].shape)
        #print(len(host_outputs))
        #print(np.concatenate(host_outputs, axis=0).shape)
        return host_outputs

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def __exit__(self, exc_type, exc_value, traceback):
        self.ctx.pop()


class trt_yolov5(yolov5_base):

    def __init__(self, model_path, tensor_height: int = 640,
                 tensor_width: int = 640) -> None:
        self.model = self.building_model(model_path)
        self.classes = yolov5_base.coco_classes

        self.H = tensor_height
        self.W = tensor_width
        self.img_size = (self.H, self.W)

    def building_model(self, model_path):
        return TrtModel(model_path)

    def detect(self, image):
        tensor = cv2.resize(image, (self.W, self.H),
                            interpolation= cv2.INTER_NEAREST)
        tensor = tensor[:, :, ::-1].transpose(2, 0, 1)
        tensor = tensor.astype(np.float32)  # uint8 to fp16/32
        tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        tensor = np.expand_dims(tensor, axis=0)
        pred = self.model(tensor)[0]
        pred = pred.reshape([1, -1, 85])
        return pred

    def predict(self, image: np.ndarray,
                conf=0.2,
                iou=0.4,
                accepted_classes=[]
                ) -> list:
        """ 
        Returns:
            list: ids, confs, boxes
        """
        inputImage = yolov5_base.format_yolov5(image)
        preds = self.detect(inputImage)
        class_ids, confidences, boxes = yolov5_base.wrap_detection(
                                inputImage, preds[0], self.H, self.W, conf, iou, 
                                accepted_classes=accepted_classes)
        
        return class_ids, confidences, boxes
