import torch
from torch import nn
import torch.nn.functional as F

from functools import reduce

from models.modules import *


class Encoder(nn.Module):
    """Convolutional Encoder."""

    def __init__(self, latent_dim, encoder_config):

        super(Encoder, self).__init__()
        self.config = encoder_config

        for k in range(self.config["blocks"]):
            if k == 0:
                in_channel, out_channel = self.config['input_size'][-1], self.config['channels'][0]
            else:
                in_channel, out_channel = self.config['channels'][k-1], self.config['channels'][k]
            setattr(self, 'conv{}'.format(k+1), conv_block(in_channel, out_channel, down=self.config["down"]))

        if self.config["mean_pool"]:
            self.fc_in_dim = self.config['last_conv_size'][-1]
        else:
            self.fc_in_dim = reduce(lambda a, b: a*b, self.config['last_conv_size'])
        self.fc = nn.Linear(self.fc_in_dim, latent_dim)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x = x.reshape(-1, *x.shape[2:])
        if self.config["down"] == 'maxpool_unpool':
            pool_idx, hidden_size = [], [x.shape]

        for k in range(self.config["blocks"]):
            x = getattr(self, 'conv{}'.format(k+1))(x)
            x = self.relu(x)
            if self.config["down"] == 'maxpool_unpool':
                pool_idx.append(getattr(self, 'conv{}'.format(k+1))[1].pool_idx)
                hidden_size.append(x.shape)

        if self.config["down"] == 'maxpool_unpool':
            return x, pool_idx, hidden_size
        return x


class Decoder(nn.Module):
    """Convolutional Decoder."""

    def __init__(self, latent_dim, decoder_config):

        super(Decoder, self).__init__()
        self.config = decoder_config

        fc_output = reduce(lambda a, b: a*b, self.config['first_deconv_size'])
        self.fc = nn.Linear(latent_dim, fc_output)
        self.relu = nn.LeakyReLU(0.5, inplace=True)
        self.unpool = (self.config["up"]=='unpool')

        for k in range(self.config["blocks"]):
            if k == 0:
                in_channel, out_channel = self.config['first_deconv_size'][0], self.config['channels'][0]
            else:
                in_channel, out_channel = self.config['channels'][k-1], self.config['channels'][k]

            if k != self.config["blocks"]-1:
                setattr(self, 'deconv{}'.format(k+1), deconv_block(in_channel, out_channel, up=self.config["up"]))
            else:
                setattr(self, 'deconv{}'.format(k+1), deconv_block(in_channel, out_channel, up=self.config["up"], last_layer=True))
            
    def forward(self, x):

        if self.unpool:
            x, pool_idx, hidden_size = x
        x = x.view(x.shape[0], *self.config['first_deconv_size'])
        for k in range(self.config["blocks"]):
            if self.unpool:
                x = getattr(self, 'deconv{}'.format(k+1))((x, pool_idx[-k-1], hidden_size[-k-2]))
            else:
                x = getattr(self, 'deconv{}'.format(k+1))(x)
            if k < self.config["blocks"]-1:
                x = self.relu(x)
            else:
                x = torch.sigmoid(x)
        return x


class BehaveNet(nn.Module):

    def __init__(self, config):
        super(BehaveNet, self).__init__()
        assert config["encoder"]["down"] == "maxpool_unpool"
        assert config["decoder"]["up"] == "unpool"
        assert config["encoder"]["input_size"][-1] == 2
        assert config["decoder"]["channels"][-1] == 2
        self.unpool = True
        self.config = config
        self.latent_dim = config["latent"]
        self.fpc = config["frames_per_clip"]

        self.encoder = Encoder(config["latent"], config["encoder"])
        self.decoder = Decoder(config["latent"], config["decoder"])
        self.dense1 = nn.Linear(reduce(lambda a, b: a*b, config["encoder"]['last_conv_size']), config["latent"])
        self.dense2 = nn.Linear(config["latent"], reduce(lambda a, b: a*b, config["decoder"]['first_deconv_size']))

        self.relu = nn.LeakyReLU(0.5, inplace=True)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=2)
        x, pool_idx, hidden_size = self.encoder(x)
        x = self.relu(self.dense1(x.flatten(start_dim=1)))
        self.latent = x
        x = self.relu(self.dense2(x))
        x = self.decoder((x, pool_idx, hidden_size))
        x = x.view(-1, self.config["frames_per_clip"], *x.shape[1:])
        return x[:, :, :1], x[:, :, 1:]


class DisAE(nn.Module):
    """Multi-View Disentangled Convolutional Encoder-Decoder."""

    def __init__(self, config):

        super(DisAE, self).__init__()
        if config["decoder"]["up"] == 'unpool':
            assert config["encoder"]["pool"]
        self.unpool = (config["decoder"]["up"] == 'unpool')
        self.config = config
        self.pose_dim = config["pose"]
        self.ct_dim = config["content"]
        self.fpc = config["frames_per_clip"]
        self.first_frame = config["first_frame"]

        # rollout / dynamic / pose encoder
        self.encoder1 = Encoder(config["pose"], config["encoder_ps"])
        self.decoder1 = Decoder(config["pose"], config["decoder"])

        self.encoder2 = Encoder(config["pose"], config["encoder_ps"])
        self.decoder2 = Decoder(config["pose"], config["decoder"])

        self.view_fuse = nn.Conv2d(256*2, config["pose"], 8, stride=1)

        # context / style / content encoder
        self.encoder1_ct = Encoder(config["content"], config["encoder_ct"])
        self.encoder2_ct = Encoder(config["content"], config["encoder_ct"])
        self.ct_fuse = nn.Conv2d(256*2, config["content"], 1, stride=1)

        # combine pose & content
        self.combine = nn.Conv2d(config["pose"]+config["content"], 256*2, 1, stride=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        

    def forward(self, x1, x2, pred_state=False):
        
        # context / style / content encoding
        if self.first_frame:
            c1, c2 = self.encoder1_ct(x1[:, :1]), self.encoder2_ct(x2[:, :1])
            c = self.relu(self.ct_fuse(torch.cat([c1, c2], dim=1)))
            self.content = c
            c = c.unsqueeze(dim=1).repeat_interleave(self.fpc, dim=1)
            x1, x2 = x1[:, 1:], x2[:, 1:]
        else:
            c1, c2 = self.encoder1_ct(x1[:, :n_past]), self.encoder2_ct(x2[:, :n_past])
            c = self.relu(self.ct_fuse(torch.cat([c1, c2], dim=1)))
            c = c.view(-1, n_past, *c.shape[1:])
            self.content = c
            c = torch.cat([c[:, :-1], c[:, -1:].repeat_interleave(n_future, dim=1)], dim=1)

        # rollout / dynamic / pose encoding
        p1, p2 = self.encoder1(x1), self.encoder2(x2)
        p = self.relu(self.view_fuse(torch.cat([p1, p2], dim=1)))
        self.latent = p.view(-1, self.fpc, self.pose_dim)

        # rollout / dynamic / pose decoding
        p = self.relu(self.combine(torch.cat([p.repeat(1, 1, 8, 8), c.flatten(end_dim=1)], dim=1)))
        p1, p2 = p[:, :256], p[:, 256:]

        if self.unpool:
            p1 = self.decoder1((p1, pool_idx1, hidden_size1))
            p2 = self.decoder2((p2, pool_idx2, hidden_size2))
        else:
            p1, p2 = self.decoder1(p1), self.decoder2(p2)
        p1, p2 = p1.view(-1, self.fpc, *p1.shape[1:]), p2.view(-1, self.fpc, *p2.shape[1:])


        return p1, p2


class DBE(nn.Module):
    # assume uniform prior and same covariance
    def __init__(self, config):
        super(DBE, self).__init__()
        if config["decoder"]["up"] == 'unpool':
            assert config["encoder"]["pool"]
        self.unpool = (config["decoder"]["up"] == 'unpool')
        self.config = config
        self.pose_dim = config["pose"]
        self.ct_dim = config["content"]
        self.state_dim = config["state_dim"]
        self.fpc = config["frames_per_clip"]
        self.indep = config["independent_cluster"]
        self.diffvar = config["different_vars"]
        self.first_frame = config["first_frame"]
        self.cond = config["ct_cond"]
        self.straight_through = config["straight_through"]
        self.context_sub = config["context_substraction"]

        # rollout / dynamic / pose encoder
        self.encoder1 = Encoder(config["pose"], config["encoder_ps"])
        self.decoder1 = Decoder(config["state_dim"], config["decoder"])

        self.encoder2 = Encoder(config["pose"], config["encoder_ps"])
        self.decoder2 = Decoder(config["state_dim"], config["decoder"])

        self.view_fuse = nn.Conv2d(256*2, config["pose"], 8, stride=1)
        if config["ct_cond"]:
            self.ct_cond = nn.Conv2d(config["content"], config["cond_dim"], 8, stride=1)
            self.posterior = GM_Recurrent(in_dim=config["pose"]+config["cond_dim"], out_dim=config["gaussian_dim"], n_clusters=config["n_clusters"], n_layer=config["rnn_layers"], tau=config["tau"], bid=config["bidirectional"])
        else:
            self.posterior = GM_Recurrent(in_dim=config["pose"], out_dim=config["gaussian_dim"], n_clusters=config["n_clusters"], n_layer=config["rnn_layers"], tau=config["tau"], bid=config["bidirectional"])
        self.init_state_posterior = Gaussian_MLP(in_dim=config["pose"]*config["n_past"], out_dim=config["state_dim"])
        self.state_model = nn.Linear(config["state_dim"]+config["gaussian_dim"], config["state_dim"])
        nn.init.normal_(self.state_model.weight.data, 0.0, 0.01)
        nn.init.constant_(self.state_model.bias.data, 0.0)
        self.centroids = nn.Embedding(config["n_clusters"], config["gaussian_dim"])
        if config["different_vars"]:
            self.logvars = nn.Embedding(config["n_clusters"], config["gaussian_dim"])

        if not config["independent_cluster"]:
            self.c_net = nn.Linear(config["state_dim"], config["n_clusters"])

        # context / style / content encoder
        self.encoder1_ct = Encoder(config["content"], config["encoder_ct"])
        self.encoder2_ct = Encoder(config["content"], config["encoder_ct"])
        if config["straight_through"]:
            self.ct_fuse = nn.Conv2d(256*2, config["content"], 1, stride=1)
        else:
            self.ct_fuse = nn.Conv2d(256*2, config["content"], 8, stride=1)

        # combine pose & content
        if config["straight_through"]:
            self.combine = nn.ConvTranspose2d(config["state_dim"]+config["content"], 256*2, 1, stride=1)
        else:
            self.combine = nn.ConvTranspose2d(config["state_dim"]+config["content"], 256*2, 8, stride=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x1, x2, n_past=None, n_future=None):
        
        assert self.fpc == n_past + n_future
        if self.first_frame:
            c1, c2 = self.encoder1_ct(x1[:, :1]), self.encoder2_ct(x2[:, :1])
            c = self.relu(self.ct_fuse(torch.cat([c1, c2], dim=1)))
            self.content = c
            c = c.unsqueeze(dim=1).repeat_interleave(self.fpc, dim=1)
            if self.context_sub:
                x1, x2 = x1[:, 1:]-x1[:, :1], x2[:, 1:]-x2[:, :1]
            else:
                x1, x2 = x1[:, 1:], x2[:, 1:]
        else:
            c1, c2 = self.encoder1_ct(x1[:, :n_past]), self.encoder2_ct(x2[:, :n_past])
            c = self.relu(self.ct_fuse(torch.cat([c1, c2], dim=1)))
            c = c.view(-1, n_past, *c.shape[1:])
            self.content = c
            c = torch.cat([c[:, :-1], c[:, -1:].repeat_interleave(n_future, dim=1)], dim=1)

        # rollout / dynamic / pose encoding
        p1, p2 = self.encoder1(x1), self.encoder2(x2)
        p = self.relu(self.view_fuse(torch.cat([p1, p2], dim=1)))
        p = p.view(-1, self.fpc, self.pose_dim)

        if self.cond:
            cc = self.relu(self.ct_cond(self.content))[:, :, 0, 0]
            g, mu, logvar, one_hot, prob1 = self.posterior(torch.cat([p, cc.unsqueeze(dim=1).repeat_interleave(self.fpc, dim=1)], dim=2))
        else:
            g, mu, logvar, one_hot, prob1 = self.posterior(p)

        if self.diffvar:
            g = g * torch.matmul(one_hot, self.logvars.weight.mul(0.5).exp()) + torch.matmul(one_hot, self.centroids.weight)
        else:
            g = g + torch.matmul(one_hot, self.centroids.weight)
        self.latent = mu.detach()
        self.cluster = one_hot.detach()

        s = []
        s0, mu_s0, logvar_s0 = self.init_state_posterior(p[:, :n_past].flatten(start_dim=1))
        si = s0
        s.append(si)
        for i in range(1, n_past+n_future):
            si = self.state_model(torch.cat([si, g[:, i]], dim=1))
            s.append(si)
        s = torch.stack(s, dim=1)
        if not self.indep:
            pre_softmax = self.c_net(s[:, :-1])
            if torch.any(torch.isnan(pre_softmax)):
                print('pre-softmax tensors contain nan!')
            if torch.any(torch.isinf(pre_softmax)):
                print('pre-softmax tensors contain inf!')
            prob2 = F.log_softmax(pre_softmax, dim=-1)
        p = s
        self.post_pose = p.detach()

        # rollout / dynamic / pose decoding
        if self.straight_through:
            p = self.relu(self.combine(torch.cat([p.view(-1, self.state_dim, 1, 1).repeat(1, 1, 8, 8), c.flatten(end_dim=1)], dim=1)))
        else:
            p = self.relu(self.combine(torch.cat([p.view(-1, self.state_dim, 1, 1), c.flatten(end_dim=1)], dim=1)))
        p1, p2 = p[:, :256], p[:, 256:]

        if self.unpool:
            p1 = self.decoder1((p1, pool_idx1, hidden_size1))
            p2 = self.decoder2((p2, pool_idx2, hidden_size2))
        else:
            p1, p2 = self.decoder1(p1), self.decoder2(p2)
        p1, p2 = p1.view(-1, self.fpc, *p1.shape[1:]), p2.view(-1, self.fpc, *p2.shape[1:])

        if not self.indep:
            return (p1, p2), (mu, logvar), (mu_s0, logvar_s0), (prob1[:, 1:], prob2)
        return (p1, p2), (mu, logvar), (mu_s0, logvar_s0), (prob1)

    def generate(self, x1, x2, n_past=None, n_future=None, c=None):
        assert self.fpc == n_past + n_future
        if self.first_frame:
            c1, c2 = self.encoder1_ct(x1[:, :1]), self.encoder2_ct(x2[:, :1])
            ct = self.relu(self.ct_fuse(torch.cat([c1, c2], dim=1)))
            self.content = ct
            ct = ct.unsqueeze(dim=1).repeat_interleave(self.fpc, dim=1)
            x1, x2 = x1[:, 1:], x2[:, 1:]
        else:
            c1, c2 = self.encoder1_ct(x1[:, :n_past]), self.encoder2_ct(x2[:, :n_past])
            ct = self.relu(self.ct_fuse(torch.cat([c1, c2], dim=1)))
            ct = ct.view(-1, n_past, *c.shape[1:])
            self.content = ct
            ct = torch.cat([ct[:, :-1], ct[:, -1:].repeat_interleave(n_future, dim=1)], dim=1)

        # rollout / dynamic / pose encoding
        p1, p2 = self.encoder1(x1), self.encoder2(x2)
        p = self.relu(self.view_fuse(torch.cat([p1, p2], dim=1)))
        p = p.view(-1, n_past, self.pose_dim)

        g, mu, logvar, one_hot, prob1 = self.posterior(p)
        if self.diffvar:
            g = g * torch.matmul(one_hot, self.logvars.weight.mul(0.5).exp()) + torch.matmul(one_hot, self.centroids.weight)
        else:
            g = g + torch.matmul(one_hot, self.centroids.weight)
        self.latent = mu.detach()
        self.cluster = one_hot.detach()

        s = []
        s0, mu_s0, logvar_s0 = self.init_state_posterior(p[:, :n_past].flatten(start_dim=1))
        si = s0
        s.append(si)
        for i in range(1, n_past):
            si = self.state_model(torch.cat([si, g[:, i]], dim=1))
            s.append(si)

        for i in range(0, n_future):
            if c is None:
                prob_dist = torch.distributions.Categorical(F.softmax(self.c_net(si), dim=-1)) # probs should be of size batch x classes
                print(F.softmax(self.c_net(si), dim=-1)[0].argmax())
                ci = prob_dist.sample()
            else:
                ci = c[:, i]
            gi = torch.normal(0, 1, size=(ci.shape[0], self.centroids.weight.shape[1])).to(si.device) + self.centroids.weight[ci]
            si = self.state_model(torch.cat([si, gi], dim=1))
            s.append(si)
        s = torch.stack(s, dim=1)
        p = s
        self.post_pose = p.detach()

        # rollout / dynamic / pose decoding
        p = self.relu(self.combine(torch.cat([p.view(-1, self.state_dim, 1, 1).repeat(1, 1, 8, 8), ct.flatten(end_dim=1)], dim=1)))
        p1, p2 = p[:, :256], p[:, 256:]

        if self.unpool:
            p1 = self.decoder1((p1, pool_idx1, hidden_size1))
            p2 = self.decoder2((p2, pool_idx2, hidden_size2))
        else:
            p1, p2 = self.decoder1(p1), self.decoder2(p2)
        p1, p2 = p1.view(-1, self.fpc, *p1.shape[1:]), p2.view(-1, self.fpc, *p2.shape[1:])

        return p1, p2
        