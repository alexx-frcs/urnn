import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .urnn_cell import URNNCell

def serialize_to_file(loss):
    file = open('losses.txt', 'w')
    for l in loss:
        file.write("{0}\n".format(l))
    file.close()

class TFRNN(nn.Module):
    def __init__(
        self,
        name,
        rnn_cell,
        num_in,
        num_hidden,
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function):

        super(TFRNN, self).__init__()

        # self
        self.name = name
        self.loss_list = []
        self.init_state_C = np.sqrt(3 / (2 * num_hidden))
        self.log_dir = './logs/'

        # init cell
        if rnn_cell == URNNCell:
            self.cell = rnn_cell(num_units=num_hidden, num_in=num_in)
        else:
            self.cell = rnn_cell(input_size=num_in, hidden_size=num_hidden, nonlinearity='tanh' if activation_hidden == nn.Tanh() else 'relu')

        # extract output size
        self.output_size = num_hidden  # In PyTorch, hidden_size is used

        # set up h->o parameters
        self.w_ho = nn.Parameter(torch.empty(num_out, self.output_size))
        nn.init.xavier_uniform_(self.w_ho)
        self.b_o = nn.Parameter(torch.zeros(num_out, 1))

        # Activation functions
        self.activation_out = activation_out if activation_out is not None else (lambda x: x)
        self.single_output = single_output

        # Loss function
        self.loss_function = loss_function

        # Optimizer
        self.optimizer = optimizer(self.parameters())

        # Number of trainable params
        t_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Network __init__ over. Number of trainable params=', t_params)

    def forward(self, input_x):
        # input_x: [batch_size, max_time, num_in]
        batch_size, max_time, _ = input_x.size()
        device = input_x.device

        # Initialize hidden state
        if isinstance(self.cell, URNNCell):
            state = torch.empty(batch_size, self.cell.state_size).uniform_(-self.init_state_C, self.init_state_C).to(device)
        else:
            state = torch.zeros(batch_size, self.cell.hidden_size).to(device)

        outputs_h = []
        for t in range(max_time):
            input_t = input_x[:, t, :]  # [batch_size, num_in]
            if isinstance(self.cell, URNNCell):
                output, state = self.cell(input_t, state)
            else:
                output, state = self.cell(input_t, state)
            outputs_h.append(output.unsqueeze(1))  # [batch_size, 1, output_size]
        outputs_h = torch.cat(outputs_h, dim=1)  # [batch_size, max_time, output_size]

        # produce final outputs from hidden layer outputs
        if self.single_output:
            outputs_h = outputs_h[:, -1, :]  # [batch_size, output_size]
            preact = torch.matmul(outputs_h, self.w_ho.t()) + self.b_o.t()
            outputs_o = self.activation_out(preact)  # [batch_size, num_out]
        else:
            # outputs_h: [batch_size, max_time, output_size]
            preact = torch.einsum('ijk,kl->ijl', outputs_h, self.w_ho.t()) + self.b_o.t()
            outputs_o = self.activation_out(preact)  # [batch_size, max_time, num_out]

        return outputs_o

    def train(self, dataset, batch_size, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # fetch validation and test sets
        num_batches = dataset.get_batch_count(batch_size)
        X_val, Y_val = dataset.get_validation_data()
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)

        # init loss list
        self.loss_list = []
        print("Starting training for", self.name)
        print("NumEpochs:", '{0:3d}'.format(epochs),
              "|BatchSize:", '{0:3d}'.format(batch_size),
              "|NumBatches:", '{0:5d}'.format(num_batches), '\n')

        # train for several epochs
        for epoch_idx in range(epochs):
            print("Epoch Starting:", epoch_idx, '\n')
            # train on several minibatches
            for batch_idx in range(num_batches):
                # get one batch of data
                # X_batch: [batch_size x time x num_in]
                # Y_batch: [batch_size x time x num_target] or [batch_size x num_target] (single_output?)
                X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)
                X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
                Y_batch = torch.tensor(Y_batch, dtype=torch.float32).to(device)

                # evaluate
                batch_loss = self.evaluate(X_batch, Y_batch, training=True)

                # save the loss for later
                self.loss_list.append(batch_loss)

                # plot
                if batch_idx % 10 == 0:
                    total_examples = batch_size * num_batches * epoch_idx + batch_size * batch_idx + batch_size

                    # print stats
                    serialize_to_file(self.loss_list)
                    print("Epoch:", '{0:3d}'.format(epoch_idx),
                          "|Batch:", '{0:3d}'.format(batch_idx),
                          "|TotalExamples:", '{0:5d}'.format(total_examples),  # total training examples
                          "|BatchLoss:", '{0:8.4f}'.format(batch_loss))

            # validate after each epoch
            validation_loss = self.evaluate(X_val, Y_val)
            mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
            print("Epoch Over:", '{0:3d}'.format(epoch_idx),
                  "|MeanEpochLoss:", '{0:8.4f}'.format(mean_epoch_loss),
                  "|ValidationSetLoss:", '{0:8.4f}'.format(validation_loss), '\n')

    def test(self, dataset, batch_size, epochs=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # fetch test set
        X_test, Y_test = dataset.get_test_data()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

        test_loss = self.evaluate(X_test, Y_test)
        print("Test set loss:", test_loss)

    def evaluate(self, X, Y, training=False):
        # X: [batch_size, time_step, num_in]
        # Y: [batch_size, num_target] or [batch_size, time_step, num_target]
        self.optimizer.zero_grad()

        # forward pass
        outputs_o = self.forward(X)

        # compute loss
        if isinstance(self.loss_function, type(nn.MSELoss())):
            loss_fn = nn.MSELoss()
            loss = loss_fn(outputs_o.squeeze(), Y.squeeze())
        elif isinstance(self.loss_function, type(nn.CrossEntropyLoss())):
            loss_fn = nn.CrossEntropyLoss()
            # For CrossEntropyLoss, outputs_o should be of shape [batch_size, num_classes]
            # and Y should be of shape [batch_size], containing class indices
            if self.single_output:
                outputs_o = outputs_o.view(-1, outputs_o.size(-1))
                Y = Y.view(-1).long()
            else:
                outputs_o = outputs_o.view(-1, outputs_o.size(-1))
                Y = Y.view(-1).long()
            loss = loss_fn(outputs_o, Y)
        else:
            raise Exception('New loss function')

        if training:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    # loss list getter
    def get_loss_list(self):
        return self.loss_list
