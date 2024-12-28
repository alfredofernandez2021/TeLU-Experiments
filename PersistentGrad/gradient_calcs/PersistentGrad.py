import torch
import torch.nn.functional as F
import time


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


def measure_gradient(func_name, function, input_size):

    x = torch.arange(start=-100.0, end=1.0, step=1.0, device=device, dtype=dtype, requires_grad=True)
    y = function(x)

    grad = torch.ones_like(y)
    y.backward(grad)

    largest_i = -1
    for i in range(len(x)):
        if torch.is_nonzero(x.grad[i]):
            pass
        else:
            largest_i = i

    print()
    print(func_name)
    print(largest_i)
    # print(x.grad[largest_i])
    # print(x[largest_i])
    # print(x.grad[largest_i+1])
    # print(x[largest_i+1])
    #print(x.grad[0], x.grad[90])
    print(x.grad)
    print()
    #print(function)

def benchmark_functions(device, dtype):
    functions = {
        'TeLU': lambda x: telu(x),
        'Mish': lambda x: mish(x),
        'Swish': lambda x: swish(x),
        'Logish': lambda x: logish(x),
        'Smish': lambda x: smish(x),
        'Softplus': lambda x: softplus(x),
        'GeLU': lambda x: gelu(x),
        'ELU': lambda x: elu(x),
        'ReLU': lambda x: relu(x),

    }
    
    print(f"Benchmarking on device: {device}, dtype: {dtype}")
    for input_size in [1]:
        print(f"\nInput size: {input_size}")
        for func_name, function in functions.items():
            measure_gradient(func_name, function, input_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for dtype in [torch.float32]:
    print(f"\nData type {dtype}")
    benchmark_functions(device, dtype)