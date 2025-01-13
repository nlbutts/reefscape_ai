import onnxruntime as ort
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Onnx perf')
parser.add_argument('-r', '--runs', help='Runs', type=int, required=True)
parser.add_argument('-m', '--model', help='Model', type=str, required=True)
args = parser.parse_args()

print(ort.get_all_providers())

# Enable profiling (this creates a log file)
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4  # Adjust this based on your CPU cores
options.inter_op_num_threads = 4
options.enable_cpu_mem_arena = True

session = ort.InferenceSession(args.model, options, providers=["ROCMExecutionProvider"])
#session = ort.InferenceSession(args.model)
#session = ort.InferenceSession(args.model, providers=["MIGraphXExecutionProvider"])

# Run inference
input = session.get_inputs()[0]
output = session.get_outputs()[0]
input_name = input.name
output_name = output.name

in_shape = input.shape
input_data = np.random.rand(in_shape[0], in_shape[1], in_shape[2], in_shape[3])
input_data = input_data.astype('float32')

print('Warm up')
for i in range(10):
    results = session.run([output_name], {input_name: input_data})

print('Test run')
runs = args.runs
start_time = time.time()
for i in range(runs):
    results = session.run([output_name], {input_name: input_data})
end_time = time.time()

diff = end_time - start_time
its = diff / runs
print(f'Ran: {runs} time per inference: {its}')