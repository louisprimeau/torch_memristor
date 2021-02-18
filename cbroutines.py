import torch
import crossbar

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, x, W, b):
        ctx.save_for_backward(x, W, b)
        return ticket.vmm(x) + b
        
    @staticmethod
    def backward(ctx, dx):
        x, W, b = ctx.saved_tensors
        return (None, 
                torch.transpose(torch.transpose(dx, 0, 1).matmul(W), 0, 1), 
                dx.matmul(torch.transpose(x,0,1)), 
                torch.eye(b.numel())
               )

class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, cb):
        super(Linear, self).__init__()
        self.W = torch.nn.parameter.Parameter(torch.rand(output_size, input_size))
        self.b = torch.nn.parameter.Parameter(torch.rand(output_size, 1))
        self.cb = cb
        self.ticket = cb.register_linear(torch.transpose(self.W,0,1))
        self.f = linear()
        self.cbon = False
        
    def forward(self, x):
        return self.f.apply(self.ticket, x, self.W, self.b) if self.cbon else self.W.matmul(x) + self.b

    def remap(self):
        self.ticket = self.cb.register_linear(torch.transpose(self.W,0,1))
    
    def use_cb(self, state):
        self.cbon = state
        
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, cb):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.linear_in = Linear(input_size, hidden_layer_size, cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = torch.nn.Tanh()
        
    def forward(self, x):
        
        h_i = torch.zeros(self.hidden_layer_size, 1)
        for x_i in x:
            h_i = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))
        return h_i 
    
    def remap(self):
        self.linear_in.remap()
        self.linear_hidden.remap()
        
class eulerforward(torch.nn.Module):
    def __init__(self, hidden_layer_size, N, cb):
        super(eulerforward, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.N = N
        self.nonlinear = torch.nn.Tanh()
    def forward(self, x0, t0, t1):
        x, h = x0, (t1 - t0) / self.N
        for i in range(self.N):
            x = x + h * self.linear(x)
        return x
    def remap(self):
        self.linear.remap()
       
    def use_cb(self, state):
        self.linear.use_cb(state)
        
class NODERNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, cb):
        super(NODERNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.cb = cb
        self.linear_in = Linear(input_size, hidden_layer_size, cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.solve = eulerforward(hidden_layer_size, 2, cb)
        self.nonlinear = torch.nn.Tanh()
        
    def forward(self, x, t):
        h_i = torch.zeros(self.hidden_layer_size, 1)
        for i, x_i in enumerate(x):
            h_i = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(self.solve(h_i, t[i-1] if i>0 else t[i], t[i])))
        return h_i 
    
    def remap(self):
        self.linear_in.remap()
        self.linear_hidden.remap()
        self.solve.remap()
    
    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)
