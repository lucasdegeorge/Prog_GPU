import contextlib
import io
import os
import random
import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
import onnx
import onnxoptimizer
import onnxruntime as ort

from onnx_utils import *


def measure_inference_time(session, input_data, mask, num_runs=10):
    times = []
    for _ in range(num_runs):
        start = time.time()
        session.run(None, {"input": input_data, "mask": mask})
        times.append(time.time() - start)
    return np.mean(times)


def plot_times(raw_times, optimized_times, model_sizes, pass_name):
    plt.figure(figsize=(10, 5))
    plt.plot(model_sizes, raw_times, label='Raw Model', marker='o', color='blue')
    plt.plot(model_sizes, optimized_times, label='Optimized Model', marker='o', color='red')
    plt.xlabel('Model Size (Hidden Layers)')
    plt.ylabel('Average Inference Time (s)')
    plt.title(f'Evolution of Elapsed Time with {pass_name} Pass')
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    passes = [
        'eliminate_identity',
        'fuse_transpose_into_gemm',
        'fuse_matmul_add_bias_into_gemm',
    ]

    # comment for cpu execution
    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]

    model_sizes = [512, 1024, 2048]
    num_hidden_layers = 2
    input_dims = [(2, 1024)]
    vocab_size = 32000
    intermediate_size = 11008
    max_position_embeddings = 2048
    num_attention_heads = [4, 8, 16]

    raw_times_pass = {p: [] for p in passes}
    optimized_times_pass = {p: [] for p in passes}

    for size, nb_head in zip(model_sizes, num_attention_heads):
        print("-------- Model Size: ", size, "-- Nb head: ", nb_head, " --------")
        model, example_args_collection = get_llama_model(input_dims, hidden_size=size,
                                                            num_hidden_layers=num_hidden_layers,
                                                            vocab_size=vocab_size,
                                                            intermediate_size=intermediate_size,
                                                            max_position_embeddings=max_position_embeddings,
                                                            num_attention_heads=nb_head)
        raw_filename = f"models/llama_raw_{size}.onnx"
        export_model(model, example_args_collection[0], raw_filename)
        # onnx.save(model, raw_filename)

        raw_session = ort.InferenceSession(raw_filename, providers=providers)
        input_data = np.random.randint(0, vocab_size, (2, 1024)).astype(np.int64)
        mask = np.tril(np.ones((2, 1024), dtype=np.float32))

        input_data_tensor = torch.tensor(input_data)
        mask_tensor = torch.tensor(mask)
        input_data_tensor_cuda = input_data_tensor.cuda()
        mask_tensor_cuda = mask_tensor.cuda()

        raw_time = measure_inference_time(raw_session, input_data, mask)
        
        for p in passes:
            print("pass: ", p)
            model = onnx.load(raw_filename)
            optimized_model = onnxoptimizer.optimize(model, [p])
            optimized_filename = f"models/llama_optimized_{size}_{p}.onnx"
            onnx.save(optimized_model, optimized_filename)
            optimized_session = ort.InferenceSession(optimized_filename, providers=providers)
            
            optimized_time = measure_inference_time(optimized_session, input_data, mask)
            
            raw_times_pass[p].append(raw_time)
            optimized_times_pass[p].append(optimized_time)
        
        # os.remove(raw_filename)

    for p in passes:
        plot_times(raw_times_pass[p], optimized_times_pass[p], model_sizes, p)

    return raw_times_pass, optimized_times_pass


if __name__ == '__main__':
    raw_times_pass, optimized_times_pass = main()
