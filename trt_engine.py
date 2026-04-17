import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        print(f"Loading TensorRT Engine from {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    'host': host_mem, 'device': device_mem, 
                    'shape': self.engine.get_binding_shape(binding),
                    'dtype': dtype, 'name': binding
                })
            else:
                self.outputs.append({
                    'host': host_mem, 'device': device_mem, 
                    'shape': self.engine.get_binding_shape(binding),
                    'dtype': dtype, 'name': binding
                })
        print(f"TensorRT Engine Loaded. Inputs: {self.inputs[0]['shape']}")

    def infer(self, image_np):
        """
        Executes inference on a preprocessed numpy array.
        Assumes image_np shape matches self.inputs[0]['shape'].
        """
        np.copyto(self.inputs[0]['host'], image_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
