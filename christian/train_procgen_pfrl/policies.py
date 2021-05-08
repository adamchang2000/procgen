import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os

class ResidualBlock(nn.Module):

    def __init__(self, channels, k_size=3):
        # Padding originally 1

        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=k_size,
                               padding=int((k_size - 1) * 0.5))
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=k_size,
                               padding=int((k_size - 1) * 0.5))

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels, k_size=3):
        # Padding originally 1

        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=k_size,
                              padding=int((k_size - 1) * 0.5))
        self.max_pool2d = nn.MaxPool2d(kernel_size=k_size,
                                       stride=2,
                                       padding=int((k_size - 1) * 0.5))
        self.res_block0 = ResidualBlock(self._out_channels, k_size)
        self.res_block1 = ResidualBlock(self._out_channels, k_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class SmallConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels, k_size=3):
        # Padding originally 1
        super(SmallConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=k_size,
                              padding=int((k_size - 1) * 0.5))
        self.max_pool2d = nn.MaxPool2d(kernel_size=k_size,
                                       stride=2,
                                       padding=int((k_size - 1) * 0.5))

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        return torch.relu(x)

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class SmallCNN(nn.Module):
    def __init__(self, num_outputs, mid_feats=256, T=0.5, alpha=0.5,
                 cl_func=lambda x, y: 0):
        super(SmallCNN, self).__init__()
        self.conv_seqs = nn.ModuleList()
        self.num_outputs = num_outputs
        self.last_shape = None
        self.mid_feats = mid_feats
        self.T = T
        self.alpha = alpha
        self.cl_func = cl_func
    
    def append(self, module, output_shape):
        self.conv_seqs.append(module)
        self.last_shape = output_shape

    def compile(self):
        assert self.last_shape != None, "Haven't appended anything to modlist"

        num_outputs = self.num_outputs
        shape = self.last_shape
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=self.mid_feats)
        self.logits_fc = nn.Linear(in_features=self.mid_feats, 
                                   out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=self.mid_feats, 
                                  out_features=1)
    
    def forward(self, x, i=-1, normalize=True, permute=True):
        # print('Using small model forward')
        if i != -1:
            assert i < len(self.conv_seqs), f"Oob for passed in index: {i}"
            return self.conv_seqs[i](x)
        
        if normalize:
            x = x / 255.0
        if permute:
            x = x.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.conv_seqs):
            x = conv(x)
        
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

    def load_from_file(self, model_path, d=None):
        if d == None:
            self.load_state_dict(torch.load(model_path))
        

class ImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs, first_channel=16, k_size=3,
                 distill_net=None, corrs=[], no_perm=False, no_scale=False):

        # corrs array holds the indices of the conv sequences
        # that we will cache for passing to the distill_net

        super(ImpalaCNN, self).__init__()
        self.no_perm = no_perm
        self.no_scale = no_scale
        self.corrs = corrs
        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [first_channel, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, k_size=k_size)
            
            # Building Student Network
            if distill_net:
                distill_net.append(
                  SmallConvSequence(shape, out_channels, k_size=k_size),
                  conv_seq.get_output_shape()
                )

            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)

        # Compiling Student Network
        if distill_net:
            distill_net.compile()

        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def assign_distill_net(self, distill_net):
        # Useful for assignment AFTER LOADING
        self.distill_net = distill_net

    def forward_distill(self, obs, cl_func, kl_func):
        # print('Using large model forward_distill')
        # cl_func: correspondence loss function (probably MSE)
        # kl_func: KL-Diverence loss function
        assert self.distill_net != None, "Distill net not set..."
        distill_net = self.distill_net

        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        if not self.no_scale:
            x = obs / 255.0  # scale to 0-1
            x_dist = obs / 255.0
        else:
            x = obs
            x_dist = obs * 1

        if not self.no_perm:
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
            x_dist = x_dist.permute(0, 3, 1, 2)

        corr_loss = 0
        for i, conv_seq in enumerate(self.conv_seqs):
            with torch.no_grad():
                x = conv_seq(x)
            
            x_dist = distill_net(x_dist, i=i)
            if len(self.corrs) != 0 and i in self.corrs:
                corr_loss += cl_func(x_dist, x)
        
        with torch.no_grad():
          x = torch.flatten(x, start_dim=1)
          x = torch.relu(x)
          x = self.hidden_fc(x)
          x = torch.relu(x)
          logits = self.logits_fc(x)
          dist = torch.distributions.Categorical(logits=logits)
          value = self.value_fc(x)

        x_dist = torch.flatten(x_dist, start_dim=1)
        x_dist = torch.relu(x_dist)
        x_dist = distill_net.hidden_fc(x_dist)
        x_dist = torch.relu(x_dist)
        logits_dist = distill_net.logits_fc(x_dist)
        dist_dist = torch.distributions.Categorical(logits=logits_dist)
        value_dist = distill_net.value_fc(x_dist)

        kl_loss = kl_func(logits_dist, logits)
        return None, None, dist_dist, value_dist, corr_loss, kl_loss

    def forward(self, obs):
        # print('Using large model forward')
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        if not self.no_scale:
            x = obs / 255.0  # scale to 0-1
        else:
            x = obs

        if not self.no_perm:
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        for i, conv_seq in enumerate(self.conv_seqs):
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
                 proc_conv_ksize=None, proc_conv_stride=None,
                 impala_layer_init=None, init_style=None,
                 log_dir=None, d=torch.device('cuda'), grad_on=False,
                 impala_k_size=3, scale=1.0):
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
            tmp_mat = np.zeros((target_width, target_width, 3))
            h, w, c = templates[f]['img'].shape

            if h > target_width:
                psh, peh = 0, target_width
                ish = int(np.floor((h - target_width) * 0.5))
                ieh = ish + target_width
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
        self.grad_on = grad_on
        self.out_feats = out_features
        self.out_conv_feats = conv_out_features

        # Constructing TempConv
        self.filter_names = []
        custom_weight = []
        for k, v in templates.items():
            self.filter_names.append(k)
            custom_weight.append(v['img'].transpose(2, 0, 1))
        custom_weight = np.array(custom_weight) * scale
        custom_weight = torch.from_numpy(custom_weight)
        custom_weight.requires_grad_(grad_on)
        self.temp_conv = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=20, 
            kernel_size=target_width,
            padding=int((target_width - 1) * 0.5),
            bias=None
        )
        self.temp_conv.weight.data = custom_weight.float()
        self.temp_conv.requires_grad_(grad_on)

        new_shape = list(obs_space.shape)
        new_shape[-1] = len(image_list)

        print(new_shape)

        self.impala = ImpalaCNN(
          obs_space=np.zeros(new_shape),
          num_outputs=num_outputs,
          first_channel=26,
          k_size=impala_k_size,
          no_perm=True,
          no_scale=True
        )

        # Mini-Conv for Additional Processing
        # self.conv = nn.Conv2d(in_channels=len(templates),
        #                       out_channels=conv_out_features,
        #                       kernel_size=3,
        #                       padding=1)
        # self.max_pool2d = nn.MaxPool2d(kernel_size=3,
        #                                stride=2,
        #                                padding=1)

        # # Logit and Value FCs
        # out_img_size = 64 / self.pooling
        # out_max_size = int(np.floor(0.5 * (out_img_size - 1)) + 1)
        # in_features = conv_out_features * out_max_size ** 2
        # self.hidden_fc = nn.Linear(in_features=in_features,
        #                            out_features=out_features)
        # self.logits_fc = nn.Linear(in_features=out_features, 
        #                            out_features=num_outputs)
        # self.value_fc = nn.Linear(in_features=out_features, 
        #                           out_features=1)

        # # Initialize weights of logits_fc
        # nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        # nn.init.zeros_(self.logits_fc.bias)

    def template_match_torch(self, i, d=torch.device('cuda')):
        if self.grad_on:
            dst = self.temp_conv(i)
        else:
            with torch.no_grad():
                dst = self.temp_conv(i)
        # filters = F.max_pool2d(dst, (self.pooling, self.pooling))
        filters = dst
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
            filters = self.template_match_torch(x, d) # b x 20 x 64 x 64

        return self.impala(filters)

        # # Learn on the pre-computed filters
        # x = self.conv(filters)
        # x = torch.relu(x)
        # x = self.max_pool2d(x)

        # # Convert to vectors
        # x = torch.flatten(x, start_dim=1)
        # x = self.hidden_fc(x)
        # x = torch.relu(x)

        # # Get Distribution and Estimated Value
        # logits = self.logits_fc(x)
        # dist = torch.distributions.Categorical(logits=logits)
        # value = self.value_fc(x)

        # return dist, value
    
    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))
        
        
class TMPNet2(nn.Module):

    def __init__(self, obs_space, num_outputs, target_width=7,
                 pooling=2, out_features = 256, conv_out_features=32,
                 d=torch.device('cuda'), grad_on=False):
        super(TMPNet2, self).__init__()
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

        new_shape = list(obs_space.shape)
        new_shape[-1] = len(image_list) + 3 # Concats

        print(new_shape)

        self.impala = ImpalaCNN(
          obs_space=np.zeros(new_shape),
          num_outputs=num_outputs,
          first_channel=26,
          no_perm=True,
          no_scale=True
        )

    def template_match_torch(self, i, d=torch.device('cuda')):
        with torch.no_grad():
            dst = self.temp_conv(i)
        filters = dst
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
            
        # Concatenation of Feature Maps with Image
        x2 = obs / 255.0
        x2 = x2.permute(0, 3, 1, 2).float()
        filters = torch.cat([x2, filters], dim=1)
        return self.impala(filters)
    
    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))

class TMPNet3(nn.Module):

    def __init__(self, obs_space, num_outputs, target_width=7,
                 pooling=2, out_features = 256, conv_out_features=32,
                 proc_conv_ksize=3, proc_conv_stride=2,
                 d=torch.device('cuda'), grad_on=False):
        super(TMPNet3, self).__init__()
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
        self.templates_tensor = torch.from_numpy(custom_weight)
        self.templates_tensor = self.templates_tensor.to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ).float()
        
        self.proc_conv = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=3, 
            kernel_size=proc_conv_ksize,
            stride=proc_conv_stride,
            bias=None
        )

        new_shape = list(obs_space.shape)
        new_shape[-1] = len(templates)
        self.impala = ImpalaCNN(
          obs_space=np.zeros(new_shape),
          num_outputs=num_outputs,
          first_channel=26,
          no_perm=True,
          no_scale=True
        )

    def forward(self, obs):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        filters = self.proc_conv(self.templates_tensor) # 20 x 3 x 3 x 3
        # filters = torch.tanh(filters) # Transform to [-1, 1]
        x = F.conv2d(x, filters, padding=1) # b x 20 x 64 x 64
        return self.impala(x)
    
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

# from vit_pytorch.vit import ViT
        
# class VIT_Wrapper(nn.Module):
#     def __init__(self, obs_space, num_outputs, 
#                  patch_size=4, dim=128, depth=3, heads=4,
#                  dropout=0.1, pool='cls'):

#         super(VIT_Wrapper, self).__init__()

#         h, w, c = obs_space.shape
#         shape = (c, h, w)
        
#         # Fruitbot: 64x64
#         # Ablation:
#         # - Patch-Size = {4, 8, 16}
#         # - Dim = {256, 512, 1024, 2048} Constant
#         # - depth = {1, 2, 3, 4, 5, 6}
#         # - heads = {4, 8, 16, 32} Constant
#         # - dropout = {0.1, 0.2, 0.3, 0.4} Constant
#         # - pool = {'cls', 'mean'} Constant
        
#         self.vit = ViT(
#             image_size = h,
#             patch_size = patch_size,
#             num_classes = num_outputs,
#             dim = dim,
#             heads = heads,
#             depth = depth,
#             mlp_dim = 2 * dim,
#             dropout = dropout,
#             emb_dropout = dropout,
#             pool = pool
#         )
        
#         self.value_fc = nn.Linear(
#             in_features=dim, out_features=1)
        
#     def forward(self, obs):
#         assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
#         x = obs / 255.0  # scale to 0-1
#         x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
#         x, logits = self.vit(x)
        
#         dist = torch.distributions.Categorical(logits=logits)
#         value = self.value_fc(torch.relu(x))

#         # print("ppo_value:", value.shape)

#         return dist, value

#     def save_to_file(self, model_path):
#         torch.save(self.state_dict(), model_path)

#     def load_from_file(self, model_path):
#         self.load_state_dict(torch.load(model_path))
        