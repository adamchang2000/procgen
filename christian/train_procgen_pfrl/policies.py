import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs):

        super(ImpalaCNN, self).__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def forward(self, obs):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))

class MultiBatchImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl. MultiBatch variant"""

    def __init__(self, obs_space, num_outputs):

        super(MultiBatchImpalaCNN, self).__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def forward(self, obss):
        # assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'

        logits = []
        values = []
        for i in range(len(obss)):
            obs = obss[i]
            x = obs / 255.0  # scale to 0-1
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
            for conv_seq in self.conv_seqs:
                x = conv_seq(x)
            x = torch.flatten(x, start_dim=1)
            x = torch.relu(x)
            x = self.hidden_fc(x)
            x = torch.relu(x)
            logits_i = self.logits_fc(x)
            logits.append(logits_i)
            values.append(self.value_fc(x))
        
        values = torch.stack(values)
        logits = torch.stack(logits)
        dist = torch.distributions.Categorical(logits=logits)
        # value = self.value_fc(x)
        return dist, values

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))

from vit_pytorch.vit import ViT
        
class VIT_Wrapper(nn.Module):
    def __init__(self, obs_space, num_outputs, 
                 patch_size=4, dim=128, depth=3, heads=4,
                 dropout=0.1, pool='cls'):

        super(VIT_Wrapper, self).__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)
        
        # Fruitbot: 64x64
        # Ablation:
        # - Patch-Size = {4, 8, 16}
        # - Dim = {256, 512, 1024, 2048} Constant
        # - depth = {1, 2, 3, 4, 5, 6}
        # - heads = {4, 8, 16, 32} Constant
        # - dropout = {0.1, 0.2, 0.3, 0.4} Constant
        # - pool = {'cls', 'mean'} Constant
        
        self.vit = ViT(
            image_size = h,
            patch_size = patch_size,
            num_classes = num_outputs,
            dim = dim,
            heads = heads,
            depth = depth,
            mlp_dim = 2 * dim,
            dropout = dropout,
            emb_dropout = dropout,
            pool = pool
        )
        
        self.value_fc = nn.Linear(
            in_features=dim, out_features=1)
        
    def forward(self, obs):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        x, logits = self.vit(x)
        
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(torch.relu(x))

        # print("ppo_value:", value.shape)

        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))
        