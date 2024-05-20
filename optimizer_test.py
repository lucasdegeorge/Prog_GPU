import onnx
import onnxoptimizer
import onnxruntime as ort
import numpy as np
import time

model_path = 'dump_llama.onnx'
model = onnx.load(model_path)

passes = [
    'eliminate_identity',
    'eliminate_nop_dropout',
    'eliminate_nop_transpose',
    'fuse_consecutive_transposes',
    'fuse_transpose_into_gemm',
    'fuse_matmul_add_bias_into_gemm',
    'eliminate_deadend',
    'fuse_pad_into_conv'
]

optimized_model = onnxoptimizer.optimize(model, passes)
optimized_model_path = 'optimized_llama.onnx'
onnx.save(optimized_model, optimized_model_path)

input_data = np.random.rand(2, 1024).astype(np.int64)
mask = np.random.randint(0, 2, (2, 1024)).astype(np.float32)

ortvalue = ort.OrtValue.ortvalue_from_numpy(input_data)
ortvalue_mask = ort.OrtValue.ortvalue_from_numpy(mask)

raw_session = ort.InferenceSession(
    'dump_llama.onnx',
)
optimized_session = ort.InferenceSession(
    'optimized_llama.onnx',
)
start = time.time()
raw_session_results = raw_session.run(["342"], {"input": ortvalue, "mask": ortvalue_mask})
ellapsed = time.time() - start

start_optimized = time.time()
optimized_session_results = optimized_session.run(["342"], {"input": ortvalue, "mask": ortvalue_mask})
ellapsed_optimized = time.time() - start_optimized

print(f"Raw session ellapsed: {ellapsed}")
print(f"Optimized session ellapsed: {ellapsed_optimized}")

np.testing.assert_allclose(raw_session_results, optimized_session_results, rtol=1e-5, atol=1e-7)
print("Results are equal!")