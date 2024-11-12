import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearUnit(nn.Module):

    def __init__(self, num_units, num_in, seq_len):
        super(LinearUnit, self).__init__()
        # save class variables
        self.num_in = num_in
        self.S = num_units//2
        self.num_units = num_units
        self.hidden_size = num_units
        self.K = seq_len



        #initialization of two filters, one for each half of the sequence
        alpha = 1
        alpha = torch.tensor(alpha, dtype=torch.float32) if not isinstance(alpha, torch.Tensor) else alpha
        self.K = torch.tensor(self.K, dtype=torch.float32) if not isinstance(self.K, torch.Tensor) else self.K

        # Calculate as_1 and as_2 (complex tensors)
        radius = torch.exp(-alpha / self.K)
        start = -torch.pi * self.S / (2 * self.K)
        end = torch.pi * self.S / (2 * self.K)
        phases = torch.linspace(start=start, end=end, steps=self.S)
        # phases = torch.linspace(start=-torch.pi/10, end=torch.pi/10, steps=self.S)
        as_1 = radius * torch.exp(1j * phases)  # as_1 est un tenseur complexe de taille (S,)

        # Génération de bs1
        z1 = (-1.0) ** torch.arange(-self.S / 2, self.S / 2)
        bs1 = z1 * (torch.exp(2 * alpha) - torch.exp(-2 * alpha)) * torch.exp(-alpha) / (2 * self.K)  # bs1 est réel de taille (S,)

        # Génération de as_2
        radius = torch.exp(-alpha / (self.K / 2))
        start = -torch.pi * self.S / self.K
        end = torch.pi * self.S / self.K
        # phases = torch.linspace(start=-torch.pi/10, end=torch.pi/10, steps=self.S)
        phases = torch.linspace(start=start, end=end, steps=self.S)
        as_2 = radius * torch.exp(1j * phases)  # as_2 est un tenseur complexe de taille (S,)

        z2 = (-1.0) ** torch.arange(-self.S / 2, self.S / 2)
        bs2 = z2 * (torch.exp(2 * alpha) - torch.exp(-2 * alpha)) * torch.exp(-alpha) / self.K  # bs2 est réel de taille (S,)

        as_ = torch.cat((as_1, as_2), dim=0)  # Taille (2S,)

        bs = torch.cat((bs1, bs2), dim=0)  # Taille (2S,)
        print('as shape', as_.shape)
        print('bs shape', bs.shape)


        as_real = as_.real  # Taille (2S,)
        as_imag = as_.imag  # Taille (2S,)

        self.as_real = nn.Parameter(as_real)
        self.as_imag = nn.Parameter(as_imag)

        bs_real = bs.real  # Taille (2S,)
        bs_imag = torch.zeros_like(bs_real)  # Taille (2S,)
        self.bs_real = nn.Parameter(bs_real)
        self.bs_imag = nn.Parameter(bs_imag)

    
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

        # print('dtype B', self.B_real.dtype)
        # print('dtype inputs', inputs.dtype)
        # print('dtype B', self.B_imag.dtype)
        print('state shape', state.shape)
        print('inputs shape', inputs.shape)

        A_diagonal = torch.cat((self.as_real[:self.S], self.as_imag[:self.S]), dim=0)  # Taille (2S,)
        A = torch.diag(A_diagonal)  # Taille (2S, 2S)
        print('A shape', A.shape)

        bs_column = torch.cat((self.bs_real[:self.S], self.bs_imag[:self.S]), dim=0).unsqueeze(1)  # Taille (2S, 1)
        B = torch.cat((bs_column, bs_column), dim=1)  # Taille (2S, 2)
        print('B shape', B.shape)
        inputs_mul = inputs @ B.t() 
        inputs_mul_c = inputs_mul[:, :self.S] + 1j * inputs_mul[:, self.S:]  # [batch_sz, num_units]
        state_mul = state @ A.t()  # [batch_sz, num_units]
        state_mul_c = state_mul[:, :self.S] + 1j * state_mul[:, self.S:]  # [batch_sz, num_units]

        new_state_c = inputs_mul_c + state_mul_c # [batch_sz, num_units], complex
        new_state = torch.cat([new_state_c.real, new_state_c.imag], dim=1)  # [batch_sz, 2*num_units], real

        # outside network (last dense layer) is ready for 2*num_units -> num_out
        output = new_state
        return output, new_state