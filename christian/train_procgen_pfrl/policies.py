import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os

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

class TMPNet(nn.Module):

    def __init__(self, obs_space, num_outputs, target_width=7,
                 pooling=2, out_features = 256, conv_out_features=32,
                 d=torch.device('cuda')):
        super(TMPNet, self).__init__()
        h, w, c = obs_space.shape
        shape = (c, h, w)

        # Loading templates
        templates = {}
        image_list = os.listdir('images')
        image_list = tqdm([i for i in image_list if '.png' in i])
        for i, f in enumerate(image_list):
            if '.png' not in f:
                continue

            img = cv2.imread(os.path.join("images", f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            templates[f] = {}

            if 'robot' in f:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            fact = img.shape[1] / target_width
            templates[f]['img'] = cv2.resize(
                img, 
                (int(img.shape[1] / fact), int(img.shape[0] / fact)), 
                interpolation=cv2.INTER_CUBIC).astype(np.float32)

            # Pad for Same Shape Filter
            tmp_mat = np.zeros((7, 7, 3))
            h, w, c = templates[f]['img'].shape

            if h > target_width:
                psh, peh = 0, target_width
                ish = int(np.floor((h - target_width) * 0.5))
                ieh = ish + 7
            else:
                psh = int(np.ceil((target_width - h) * 0.5))
                peh = psh + h
                ish, ieh = 0, h
            tmp_mat[psh:peh, :, :] += templates[f]['img'][ish:ieh, :, :]
            templates[f]['img'] = tmp_mat

            #normalize by filter size and values
            templates[f]['img'] = templates[f]['img'] / (
                np.mean(templates[f]['img']) *\
                templates[f]['img'].shape[0] *\
                templates[f]['img'].shape[1])

            #zero-mean the filter
            templates[f]['img'] -= np.mean(templates[f]['img']) 

        for k, v in templates.items():
            v["fruit"] = "fruit" in k # fruit increases score
            v["food"] = "food" in k # food decreases score

        # Setting Instance Variables
        self.templates = templates
        self.target_width = target_width
        self.pooling = pooling
        self.out_feats = out_features
        self.out_conv_feats = conv_out_features

        # Constructing TempConv
        self.filter_names = []
        custom_weight = []
        for k, v in templates.items():
            self.filter_names.append(k)
            custom_weight.append(v['img'].transpose(2, 0, 1))
        custom_weight = np.array(custom_weight)
        custom_weight = torch.from_numpy(custom_weight)
        self.temp_conv = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=20, 
            kernel_size=target_width,
            padding=int((target_width - 1) * 0.5),
            bias=None
        ).requires_grad_(False)
        self.temp_conv.weight.data = custom_weight.float()

        # Mini-Conv for Additional Processing
        self.conv = nn.Conv2d(in_channels=len(templates),
                              out_channels=conv_out_features,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)

        # Logit and Value FCs
        out_img_size = 64 / self.pooling
        out_max_size = int(np.floor(0.5 * (out_img_size - 1)) + 1)
        in_features = conv_out_features * out_max_size ** 2
        self.hidden_fc = nn.Linear(in_features=in_features,
                                   out_features=out_features)
        self.logits_fc = nn.Linear(in_features=out_features, 
                                   out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=out_features, 
                                  out_features=1)

        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def template_match_torch(self, i, d=torch.device('cuda')):
        with torch.no_grad():
            dst = self.temp_conv(i)
        filters = F.max_pool2d(dst, (self.pooling, self.pooling))
        return filters

    def normalize(self, x, torch_wise=True):
        mean_func = torch.mean if torch_wise else np.mean
        x = x / (mean_func(x) * x.shape[0] * x.shape[1])
        return x - mean_func(x)

    def forward(self, obs, d=torch.device('cuda')):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = self.normalize(obs.float()) # obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        with torch.no_grad():
            filters = self.template_match_torch(x, d)

        # Learn on the pre-computed filters
        x = self.conv(filters)
        x = torch.relu(x)
        x = self.max_pool2d(x)

        # Convert to vectors
        x = torch.flatten(x, start_dim=1)
        x = self.hidden_fc(x)
        x = torch.relu(x)

        # Get Distribution and Estimated Value
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)

        return dist, value
    
    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))

class ImpalaCNN_TMP(nn.Module):
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

    def template_match(obs, templates):
        """template match against RGB window, observation of shape (nxmx3)"""

        rgb = obs
        bgr = obs
        # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) #convert rgb to bgr
        # print(bgr.dtype)

        names = []
        filter_out = np.zeros((
            len(list(templates.keys())), bgr.shape[0], bgr.shape[1]))

        idx = 0
        for k, v in templates.items():
            names.append(k)
            kernel = v['img']
            dst = np.zeros((bgr.shape[0], bgr.shape[1]))
            for i in range(3):
                a = cv2.filter2D(bgr[:,:,i], -1, kernel[:,:,i])
                dst += a
            filter_out[idx] = dst
            idx += 1

        filter_out_orig = filter_out
        filter_out = filter_out.transpose((1, 2, 0))
        filter_out_max = np.max(filter_out, axis=-1)
        filter_out_amax = np.argmax(filter_out, axis=-1)
            
        return filter_out_orig, filter_out_amax, names

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
        