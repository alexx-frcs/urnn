import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
from models.tf_rnn import TFRNN
from models.urnn_cell import URNNCell
from models.lru_cell import LinearUnit

'''
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
'''

loss_path = 'results/'

glob_learning_rate = 0.001
glob_decay = 0.9

def baseline_ap():
    return 0.167

def serialize_loss(loss, name):
    with open(loss_path + name, 'w') as file:
        for l in loss:
            file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        # init adding problem
        self.ap_batch_size = 50
        self.ap_epochs = 10

        self.ap_timesteps = [100, 200, 400, 750]
        self.ap_samples = [30000, 50000, 40000, 100000]
        self.ap_data = [AddingProblemDataset(sample, timesteps) for timesteps, sample in zip(self.ap_timesteps, self.ap_samples)]
        self.dummy_ap_data = AddingProblemDataset(100, 50)  # samples, timestamps
        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=', sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        serialize_loss(net.get_loss_list(), net.name + sample_len)
        print('Training network ', net.name, ' done.')

    def train_urnn_for_timestep_idx(self, idx):
        print('Initializing and training URNNs for one timestep...')

        # AP
        self.ap_urnn = TFRNN(
            name="ap_urnn",
            num_in=2,
            num_hidden=512,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=URNNCell,
            activation_hidden=None,  # modReLU
            activation_out=lambda x: x,
            optimizer=lambda params: optim.RMSprop(params, lr=glob_learning_rate, alpha=glob_decay),
            loss_function=nn.MSELoss()
        )
        self.train_network(self.ap_urnn, self.ap_data[idx],
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training URNNs for one timestep done.')

    def train_lru_for_timestep_idx(self, idx):
        print('Initializing and training URNNs for one timestep...')

        # AP
        self.ap_lru = TFRNN(
            name="lru_urnn",
            num_in=2,
            num_hidden=64,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=LinearUnit,
            activation_hidden=None,  # modReLU
            activation_out=nn.ReLU(),
            optimizer=lambda params: optim.RMSprop(params, lr=glob_learning_rate, alpha=glob_decay),
            loss_function=nn.MSELoss()
        )
        print('as init', self.ap_lru.cell.as_real + 1j * self.ap_lru.cell.as_imag)
        self.train_network(self.ap_lru, self.ap_data[idx],
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training LRU for one timestep done.')

    def train_rnn_lstm_for_timestep_idx(self, idx):
        print('Initializing and training RNN&LSTM for one timestep...')

        # AP

        self.ap_simple_rnn = TFRNN(
            name="ap_simple_rnn",
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=nn.RNNCell,
            activation_hidden=nn.Tanh(),
            activation_out=lambda x: x,
            optimizer=lambda params: optim.RMSprop(params, lr=glob_learning_rate, alpha=glob_decay),
            loss_function=nn.MSELoss()
        )
        self.train_network(self.ap_simple_rnn, self.ap_data[idx],
                           self.ap_batch_size, self.ap_epochs)

        self.ap_lstm = TFRNN(
            name="ap_lstm",
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=nn.LSTMCell,
            activation_hidden=nn.Tanh(),
            activation_out=lambda x: x,
            optimizer=lambda params: optim.RMSprop(params, lr=glob_learning_rate, alpha=glob_decay),
            loss_function=nn.MSELoss()
        )
        self.train_network(self.ap_lstm, self.ap_data[idx],
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training networks for one timestep done.')

    def train_networks(self):
        print('Starting training...')

        timesteps_idx = 4
        for i in range(timesteps_idx):
            self.train_lru_for_timestep_idx(i)
        for i in range(timesteps_idx):
            self.train_urnn_for_timestep_idx(i)
        for i in range(timesteps_idx):
            self.train_rnn_lstm_for_timestep_idx(i)

        print('Done and done.')

if __name__ == "__main__":
    main = Main()
    main.init_data()
    main.train_networks()
