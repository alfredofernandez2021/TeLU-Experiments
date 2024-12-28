import torch
import torch.nn.functional as F
import numpy
import time


# def telu(input):
#     return input * torch.tanh(torch.exp(input))

# def gelu(input):
#     #return torch.div(input,2) * (1 + torch.erf(torch.div(input, numpy.math.NPY_SQRT2f)))
#     return input * (1 + torch.erf(input))

# def smish(input):
#     return input * torch.tanh(torch.log( 1 + (1 / (1 + torch.exp(-1 * input))) ))

# def logish(input):
#     return input * torch.log( 1 + (1 / (1 + torch.exp(-1 * input))) )

# def relu(input):
#     return torch.where(input < 0, 0, input)

# def mish(input):
#     return input * torch.tanh( torch.log( 1 + torch.exp(input)) )

# def swish(input):
#     return input * (1 / (1 + torch.exp(-1 * input)))

# def softplus(input):
#     return torch.log(1 + torch.exp(input))

# def elu(input):
#     return torch.where(input > 0, input, torch.exp(input))

def telu(input):
    return input * torch.tanh(torch.exp(input))

def gelu(input):
    #root2 = torch.sqrt(torch.ones_like(input)*2)
    #return input * (1 + torch.erf(input/root2))/2
    return input * (1 + torch.erf(input/1.4142))/2

def smish(input):
    exp = torch.exp(input)
    return input * torch.tanh(torch.log( 1 + (exp / (1 + exp)) ))
    #return input * torch.tanh(torch.log( torch.sigmoid(input) ))

def logish(input):
    exp = torch.exp(input)
    return input * torch.log( 1 + (exp / (1 + exp)))
    #return input * torch.log( torch.sigmoid(input))

def relu(input):
    return torch.max(torch.zeros_like(input, dtype=torch.float32), input)
    #return torch.max(0.0, input)

def mish(input):
    return input * torch.tanh( torch.log( 1 + torch.exp(input)) )

def swish(input):
    exp = torch.exp(input)
    return input * (exp / (1 + exp))
    #return input * torch.sigmoid(input)

def softplus(input):
    return torch.log(1 + torch.exp(input))

def elu(input):
    return torch.where(input > 0, input, torch.exp(input)-1)


def measure_fwd_bwd_time(function, x, input_size):
    #Time = time.time()
    fwd_time = 0
    bwd_time = 0

    for _ in range(1000000):
        start_time = time.time()
        torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
        y = function(x)
        torch.cuda.synchronize()
        fwd_time += time.time() - start_time

        # Create a gradient tensor for backward computation
        grad = torch.ones_like(y)
        start_time = time.time()
        torch.cuda.synchronize()
        y.backward(grad)
        torch.cuda.synchronize()
        bwd_time += time.time() - start_time

    return fwd_time, bwd_time


def benchmark_functions(device, dtype):
    input_sizes = [1000000]  # Example input sizes
    functions = {
        'TeLU': lambda x: telu(x),
        'GeLU': lambda x: gelu(x),
        'Smish': lambda x: smish(x),
        'Logish': lambda x: logish(x),
        'ReLU': lambda x: relu(x),
        'Mish': lambda x: mish(x),
        'Swish': lambda x: swish(x),
        #'Softplus': lambda x: softplus(x),
        'ELU': lambda x: elu(x),

    }

    # z = torch.randn(10, device=device, dtype=torch.float32, requires_grad=True)
    # g = relu(z)
    # print(z)
    # print(g)
    
    print(f"Benchmarking on device: {device}, dtype: {dtype}")
    for input_size in input_sizes:
        x = torch.randn(input_size, device=device, dtype=dtype, requires_grad=True)
        print(f"\nInput size: {input_size}")
        for func_name, function in functions.items():
            fwd_time, bwd_time = measure_fwd_bwd_time(function, x, input_size)
            print(f"{func_name}: Forward Time: {fwd_time:.6f}s, Backward Time: {bwd_time:.6f}s")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Benchmark for float32 and float32
#for dtype in [torch.float32, torch.float32]:
for dtype in [torch.float32]:
    print(f"\nData type {dtype}")
    benchmark_functions(device, dtype)
