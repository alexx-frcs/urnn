import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Diagonal unitary matrix
class DiagonalMatrix():
    def __init__(self, name, num_units):
        init_w = torch.FloatTensor(num_units).uniform_(-np.pi, np.pi)
        self.w = nn.Parameter(init_w)
        self.vec = torch.cos(self.w) + 1j * torch.sin(self.w)

    # [batch_sz, num_units]
    def mul(self, z):
        # [num_units] * [batch_sz, num_units] -> [batch_sz, num_units]
        return self.vec * z

# Reflection unitary matrix
class ReflectionMatrix():
    def __init__(self, name, num_units):
        self.num_units = num_units

        self.re = nn.Parameter(torch.FloatTensor(num_units).uniform_(-1, 1))
        self.im = nn.Parameter(torch.FloatTensor(num_units).uniform_(-1, 1))
        self.v = self.re + 1j * self.im  # [num_units]
        # self.v = normalize(self.v)
        self.vstar = torch.conj(self.v)  # [num_units]

    # [batch_sz, num_units]
    def mul(self, z):
        v = self.v.view(self.num_units, 1)  # [num_units, 1]
        vstar = torch.conj(v)  # [num_units, 1]
        vstar_z = torch.matmul(z, vstar)  # [batch_size, 1]
        sq_norm = torch.sum(torch.abs(self.v) ** 2)  # [1]
        factor = 2 / (sq_norm + 1e-6)
        return z - factor * vstar_z * v.t()  # [batch_size, num_units]

# Permutation unitary matrix
class PermutationMatrix:
    def __init__(self, name, num_units):
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        self.P = torch.tensor(perm, dtype=torch.long)

    # [batch_sz, num_units], permute columns
    def mul(self, z):
        return z[:, self.P]

# FFTs
# z: complex[batch_sz, num_units]

def FFT(z):
    return torch.fft.fft(z, dim=1)

def IFFT(z):
    return torch.fft.ifft(z, dim=1)
    
def normalize(z):
    norm = torch.sqrt(torch.sum(torch.abs(z)**2))
    factor = norm + 1e-6
    return z / factor

# z: complex[batch_sz, num_units]
# bias: real[num_units]
def modReLU(z, bias):  # relu(|z|+b) * (z / |z|)
    norm = torch.abs(z)
    scale = F.relu(norm + bias) / (norm + 1e-6)
    scaled = z * scale
    return scaled

###################################################################################################

# 4k / 7k trainable params
class URNNCell(nn.Module):
    """The most basic URNN cell.
    Args:
        num_units (int): The number of units in the LSTM cell, hidden layer size.
        num_in: Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in):
        super(URNNCell, self).__init__()
        # save class variables
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units * 2
        self._output_size = num_units * 2

        # set up input -> hidden connection
        self.w_ih = nn.Parameter(torch.empty(2 * num_units, num_in))
        nn.init.xavier_uniform_(self.w_ih)
        self.b_h = nn.Parameter(torch.zeros(num_units))

        # elementary unitary matrices to get the big one
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)
        self.P = PermutationMatrix("P", num_units)

    # needed properties
    @property
    def input_size(self):
        return self._num_in  # real

    @property
    def state_size(self):
        return self._state_size  # real

    @property
    def output_size(self):
        return self._output_size  # real

    def forward(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x 2*num_units): Previous cell state REAL
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units*2): New state of the cell.
        """
        # prepare input linear combination
        inputs_mul = inputs @ self.w_ih.t()  # [batch_sz, 2*num_units]
        inputs_mul_c = inputs_mul[:, :self._num_units] + 1j * inputs_mul[:, self._num_units:]  # [batch_sz, num_units]
        
        # print('state shape', state.shape)
        # print('inputs shape', inputs.shape)
        # prepare state linear combination (always complex!)
        state_c = state[:, :self._num_units] + 1j * state[:, self._num_units:]  # [batch_sz, num_units]

        state_mul = self.D1.mul(state_c)
        state_mul = FFT(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = self.P.mul(state_mul)
        state_mul = self.D2.mul(state_mul)
        state_mul = IFFT(state_mul)
        state_mul = self.R2.mul(state_mul)
        state_mul = self.D3.mul(state_mul) 
        # [batch_sz, num_units]
        
        # calculate preactivation
        preact = inputs_mul_c + state_mul  # [batch_sz, num_units]

        new_state_c = modReLU(preact, self.b_h)  # [batch_sz, num_units], complex
        new_state = torch.cat([new_state_c.real, new_state_c.imag], dim=1)  # [batch_sz, 2*num_units], real
        # outside network (last dense layer) is ready for 2*num_units -> num_out
        output = new_state
        return output, new_state
