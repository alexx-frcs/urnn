import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearUnit(nn.Module):

    def __init__(self, num_units, num_in, seq_len):
        super(LinearUnit, self).__init__()
        # save class variables
        self.num_in = num_in
        self.S = num_units
        self.hidden_size = num_units
        self.K = seq_len



        #initialization of two filters, one for each half of the sequence
        alpha = 1
        alpha = torch.tensor(alpha, dtype=torch.float32) if not isinstance(alpha, torch.Tensor) else alpha
        self.K = torch.tensor(self.K, dtype=torch.float32) if not isinstance(self.K, torch.Tensor) else self.K

        # Calculate as_1 and as_2 (complex tensors)
        radius1 = torch.exp(-alpha / self.K)
        start1 = -torch.pi * self.S / (2 * self.K)
        end1 = torch.pi * self.S / (2 * self.K)
        phases1 = torch.linspace(start=start1, end=end1, steps=self.S)
        as_1 = radius1 * torch.exp(1j * phases1)

        radius2 = torch.exp(-alpha / (self.K / 2))
        start2 = -torch.pi * self.S / self.K
        end2 = torch.pi * self.S / self.K
        phases2 = torch.linspace(start=start2, end=end2, steps=self.S)
        as_2 = radius2 * torch.exp(1j * phases2)

        # Combine as_1 and as_2 to form the diagonal of A
        as_combined = torch.cat((as_1, as_2), dim=0)
        as_combined_real = as_combined.real
        as_combined_imag = as_combined.imag
        as_combined = torch.cat((as_combined_real, as_combined_imag), dim=0)
        A = torch.diag(as_combined)

        # Calculate bs1 and bs2 (ensure they are complex tensors)
        z1_indices = torch.arange(-self.S / 2, self.S / 2)
        z1 = (-1.0) ** z1_indices
        bs1_real = z1 * (torch.exp(2 * alpha) - torch.exp(-2 * alpha)) * torch.exp(-alpha) / (2 * self.K)
        bs1 = bs1_real.type(torch.complex64)  # Convert to complex tensor with zero imaginary part

        z2_indices = torch.arange(-self.S / 2, self.S / 2)
        z2 = (-1.0) ** z2_indices
        bs2_real = z2 * (torch.exp(2 * alpha) - torch.exp(-2 * alpha)) * torch.exp(-alpha) / self.K
        bs2 = bs2_real.type(torch.complex64)  # Convert to complex tensor
        as_combined_imag = as_combined.imag
        as_combined = torch.cat((as_combined_real, as_combined_imag), dim=0)
        A = torch.diag(as_combined)

        bs_combined = torch.cat((bs1, bs2), dim=0).unsqueeze(1)

        B = torch.cat((bs_combined, bs_combined), dim=1)

        # Now B is a complex tensor, and you can access B.real and B.imag
        A_real = A.real
        A_imag = A.imag
        B_real = B.real
        B_imag = B.imag

        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)
        self.B_real = nn.Parameter(B_real)
        self.B_imag = nn.Parameter(B_imag)

    
    @property
    def input_size(self):
        return self.num_in
    
    @property 
    def state_size(self):
        return self.S
    
    @property
    def output_size(self):
        return self.output_size


    def forward(self, inputs, state):

        """The most basic LRU cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """

        inputs = inputs.type(torch.complex64)
        state = state.type(torch.complex64)

        # print('dtype B', self.B_real.dtype)
        # print('dtype inputs', inputs.dtype)
        # print('dtype B', self.B_imag.dtype)

        inputs_mul = (inputs @ self.B_real.t()).type(torch.complex64) + (1j * inputs @ self.B_imag.t()).type(torch.complex64) # [batch_sz, num_units]

        state_c = state[:, :self._num_units] + 1j * state[:, self._num_units:]  # [batch_sz, num_units]

        state_mul = self.A_real.mul(state_c) + 1j * self.A_imag.mul(state_c) # [batch_sz, num_units]

        new_state_c = inputs_mul + state_mul # [batch_sz, num_units], complex
        new_state = torch.cat([new_state_c.real, new_state_c.imag], dim=1)  # [batch_sz, 2*num_units], real

        # outside network (last dense layer) is ready for 2*num_units -> num_out
        output = new_state
        return output, new_state