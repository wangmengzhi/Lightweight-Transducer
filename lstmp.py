import math
import torch
from torch.nn import init
import torch.jit as jit
from torch.nn import Parameter
# from torch.jit import Tensor  # there is an error
from torch import Tensor
from typing import List, Tuple

class LSTMP(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, projection_size,dropout=0):
        super(LSTMP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.weight_x = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_r = Parameter(torch.randn(4 * hidden_size, projection_size))
        self.weight_p = Parameter(torch.randn(projection_size, hidden_size))
        self.weight_bias = Parameter(torch.randn(4 * hidden_size))
        self.init_weights()
        self.dropout=torch.nn.Dropout(dropout,inplace=False)
    
    @jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor
        # state cannot be None
        hx = x.new_zeros(x.size(1), self.projection_size)
        cx = x.new_zeros(x.size(1), self.hidden_size)
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, hx,cx = self.step(inputs[i], hx,cx)
            outputs += [out]
        return torch.stack(outputs)
    
    @jit.script_method
    def step(self, input, hx, cx):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        # input: batch_size * input_size
        # state: hx -> batch_size * projection_size 
        #        cx -> batch_size * hidden_size 
        # state cannot be None
        #if state is not None:
            #hx, cx = state
        #else:
            #hx = input.new_zeros(input.size(0), self.projection_size, requires_grad=False)
            #cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        #hx, cx = state
        gates = torch.mm(input, self.weight_x.t()) + torch.mm(hx, self.weight_r.t()) + self.weight_bias
        ingate, forgetgate, outgate, cellgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        out = torch.mm(self.dropout(hy), self.weight_p.t())
        hy = torch.mm(hy, self.weight_p.t())

        return out, hy, cy
    
    def init_weights(self):
        #stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv=0.1
        init.uniform_(self.weight_x, -stdv, stdv)
        init.uniform_(self.weight_r, -stdv, stdv)
        init.uniform_(self.weight_p, -stdv, stdv)
        init.zeros_(self.weight_bias)

class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih, -stdv, stdv)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.bias_ih)
        init.uniform_(self.bias_hh)

class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
    # def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        # self.cell = LSTMCell(input_size, hidden_size)
        # print('initial params of weight_ih: ')
        # print(self.cell.weight_ih)
        # print('initial params of weight_hh: ')
        # print(self.cell.weight_hh)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

def test():
    input_size = 320
    hidden_size = 768
    projection_size=256
    rnn = LSTMP(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
    x = torch.rand((50, 4, 320))
    hx = x.new_zeros(x.size(1), projection_size, requires_grad=False)
    cx = x.new_zeros(x.size(1), hidden_size, requires_grad=False)
    state = [hx, cx]
    y, h = rnn(x, state)
