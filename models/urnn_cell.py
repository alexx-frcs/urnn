import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Diagonal unitary matrix
class DiagonalMatrix():
    def __init__(self, num_units):
        init_w = torch.rand(num_units) * 2 * np.pi - np.pi  # Uniform distribution between -π and π
        self.w = nn.Parameter(init_w)
        self.vec = torch.cos(self.w) + 1j * torch.sin(self.w)  # e^{i w}

    # z: [batch_size, num_units], complex tensor
    def mul(self, z):
        # Element-wise multiplication
        return self.vec * z  # [batch_size, num_units]

# Reflection unitary matrix
class ReflectionMatrix():
    def __init__(self, num_units):
        self.num_units = num_units
        re = torch.rand(num_units) * 2 - 1  # Uniform distribution between -1 and 1
        im = torch.rand(num_units) * 2 - 1  # Uniform distribution between -1 and 1
        self.v = nn.Parameter(re + 1j * im)  # [num_units]
        # self.v = normalize(self.v)  # Normalization can be applied if needed

    # z: [batch_size, num_units], complex tensor
    def mul(self, z):
        v = self.v  # [num_units]
        vstar = torch.conj(v)  # [num_units]
        vstar = vstar.view(-1, 1)  # [num_units, 1]
        vstar_z = z @ vstar  # [batch_size, 1]
        sq_norm = torch.sum(torch.abs(v) ** 2)  # [1]
        factor = 2 / sq_norm
        return z - factor * vstar_z * v.unsqueeze(0)  # [batch_size, num_units]

# Permutation unitary matrix
class PermutationMatrix():
    def __init__(self, num_units):
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        self.P = torch.tensor(perm, dtype=torch.long)

    # z: [batch_size, num_units], complex tensor
    def mul(self, z):
        return z[:, self.P]  # Permute columns

# FFTs
def FFT(z):
    return torch.fft.fft(z, dim=-1)

def IFFT(z):
    return torch.fft.ifft(z, dim=-1)

def normalize(z):
    norm = torch.sqrt(torch.sum(torch.abs(z) ** 2))
    factor = norm + 1e-6
    return z / factor

# z: complex tensor [batch_size, num_units]
# bias: real tensor [num_units]
def modReLU(z, bias):  # ReLU(|z| + b) * (z / |z|)
    norm = torch.abs(z)  # [batch_size, num_units]
    bias = bias.view(1, -1)  # [1, num_units] for broadcasting
    scale = F.relu(norm + bias) / (norm + 1e-6)  # Avoid division by zero
    return z * scale  # [batch_size, num_units]

###############################################################################################

# URNNCell
class URNNCell(nn.Module):
    """The most basic URNN cell.
    Args:
        num_units (int): The number of units in the URNN cell, hidden layer size.
        num_in (int): Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in):
        super(URNNCell, self).__init__()
        self.num_units = num_units
        self.num_in = num_in

        # Input to hidden weights
        self.w_ih = nn.Parameter(torch.empty(2 * num_units, num_in))
        nn.init.xavier_uniform_(self.w_ih)
        self.b_h = nn.Parameter(torch.zeros(num_units))  # Real bias term

        # Elementary unitary matrices
        self.D1 = DiagonalMatrix(num_units)
        self.R1 = ReflectionMatrix(num_units)
        self.D2 = DiagonalMatrix(num_units)
        self.R2 = ReflectionMatrix(num_units)
        self.D3 = DiagonalMatrix(num_units)
        self.P = PermutationMatrix(num_units)

    def forward(self, inputs, state):
        """
        Args:
            inputs (Tensor): [batch_size, num_in], real tensor
            state (Tensor): [batch_size, 2 * num_units], real tensor (concatenated real and imag parts)
        Returns:
            output (Tensor): [batch_size, 2 * num_units], real tensor
            new_state (Tensor): [batch_size, 2 * num_units], real tensor
        """
        # Prepare input linear combination
        inputs_mul = inputs @ self.w_ih.t()  # [batch_size, 2 * num_units]
        inputs_mul_c = torch.complex(inputs_mul[:, :self.num_units], inputs_mul[:, self.num_units:])  # [batch_size, num_units]

        # Prepare state
        state_c = torch.complex(state[:, :self.num_units], state[:, self.num_units:])  # [batch_size, num_units]

        # Apply unitary transformations
        state_mul = self.D1.mul(state_c)
        state_mul = FFT(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = self.P.mul(state_mul)
        state_mul = self.D2.mul(state_mul)
        state_mul = IFFT(state_mul)
        state_mul = self.R2.mul(state_mul)
        state_mul = self.D3.mul(state_mul)  # [batch_size, num_units]

        # Calculate preactivation
        preact = inputs_mul_c + state_mul  # [batch_size, num_units]

        # Apply modReLU activation
        new_state_c = modReLU(preact, self.b_h)  # [batch_size, num_units], complex tensor

        # Concatenate real and imaginary parts
        new_state = torch.cat([new_state_c.real, new_state_c.imag], dim=1)  # [batch_size, 2 * num_units]

        # Output is the same as new_state
        output = new_state  # [batch_size, 2 * num_units]

        return output, new_state
