import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv_block(in_channels, out_channels, kernel_size=5, stride=1, padding=2, down='maxpool'):
    '''
    returns a conv block conv-bn-relu-pool
    set return indices to true for unpooling later
    '''
    if down == 'conv':
        stride = 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        Down2dWrapper(down), 
        nn.BatchNorm2d(out_channels), 
    )


def deconv_block(in_channels, out_channels, kernel_size=5, stride=1, padding=2, up='unpool', last_layer=False):
    '''
    returns a deconv block conv-bn-relu-unpool
    '''
    return nn.Sequential(
        Up2dWrapper(up), 
        nn.ConvTranspose2d(in_channels, out_channels*(2**2) if up=='subpixel' else out_channels, kernel_size, stride=stride, padding=padding), # asymmetric padding?
        nn.BatchNorm2d(out_channels), 
    )


class Down2dWrapper(nn.Module):
    '''
    workaround for sequential not taking multiple inputs
    '''
    def __init__(self, down='maxpool'):

        super(Down2dWrapper, self).__init__()
        self.down = down
        if down == 'maxpool':
            self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False)
        elif down == 'maxpool_unpool':
            self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        elif down == 'conv':
            pass
        else:
            raise ValueError("Downsample method not supported!")

    def forward(self, x):
        if self.down == 'maxpool_unpool':
            x, pool_idx = self.maxpool(x)
            self.pool_idx = pool_idx
            return x
        elif self.down == 'maxpool':
            return self.maxpool(x)
        elif self.down == 'conv':
            return x


class Up2dWrapper(nn.Module):
    '''
    workaround for sequential not taking multiple inputs
    '''
    def __init__(self, up='unpool'):

        super(Up2dWrapper, self).__init__()
        self.up = up
        if up == 'unpool':
            self.unpool = nn.MaxUnpool2d(2, stride=2)
        elif up == 'upsample':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        elif up == 'subpixel':
            self.subpixel = nn.PixelShuffle(upscale_factor=2)
        else:
            raise ValueError("Upsample method not supported!")

    def forward(self, x):
        if self.up == 'unpool':
            x, pool_idx, hidden_size = x
            return self.unpool(x, pool_idx, hidden_size)
        elif self.up == 'upsample':
            return self.upsample(x)
        elif self.up == 'subpixel':
            return self.subpixel(x)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layer=2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Gaussian_MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layer=2):
        super(Gaussian_MLP, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.mu_net = nn.Linear(hidden_dim, out_dim)
        self.logvar_net = nn.Linear(hidden_dim, out_dim)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        h = F.relu(self.linear(x))
        mu = self.mu_net(h)
        logvar = self.logvar_net(h)
        if self.training:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        return mu, mu, logvar

class Recurrent(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layer=1, mod='GRU'):
        super(Recurrent, self).__init__()
        self.rnn = getattr(nn, mod)(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        x = self.linear(h)
        return x

class Recurrent_Cell(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layer=1, mod='GRU'):
        super(Recurrent_Cell, self).__init__()
        self.mod = mod
        self.n_layer = n_layer
        self.rnn_cell = nn.ModuleList([getattr(nn, '{}Cell'.format(mod))(in_dim if i==0 else hidden_dim, hidden_dim, n_layer, batch_first=True) for i in range(n_layer)])
        self.linear = nn.Linear(hidden_dim, out_dim)

    def init_hidden(self, device):
        hidden = []
        for i in range(self.n_layer):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)),
                        #    Variable(torch.zeros(self.batch_size, self.hidden_size).to(device))
                           ))
        return hidden

    def forward(self, x):
        h = x
        for i in range(self.n_layer):
            self.hidden[i] = self.rnn_cell[i](h, self.hidden[i])
            h = self.hidden[i][0]
        x = self.linear(h)
        return x


class Gaussian_Recurrent(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layer=1, mod='GRU'):
        super(Gaussian_Recurrent, self).__init__()
        self.rnn = getattr(nn, mod)(in_dim, hidden_dim, n_layer, batch_first=True)
        self.mu_net = nn.Linear(hidden_dim, out_dim)
        self.logvar_net = nn.Linear(hidden_dim, out_dim)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        h, _ = self.rnn(x)
        mu = self.mu_net(h)
        logvar = self.logvar_net(h)
        if self.training:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        return mu, mu, logvar
        
class Gaussian_Recurrent_Cell(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layer=1, mod='GRU'):
        super(Gaussian_Recurrent_Cell, self).__init__()
        self.mod = mod
        self.n_layer = n_layer
        self.rnn_cell = nn.ModuleList([getattr(nn, '{}Cell'.format(mod))(in_dim if i==0 else hidden_dim, hidden_dim, n_layer, batch_first=True) for i in range(n_layer)])
        self.mu_net = nn.Linear(hidden_dim, out_dim)
        self.logvar_net = nn.Linear(hidden_dim, out_dim)

    def init_hidden(self, device):
        hidden = []
        for i in range(self.n_layer):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)),
                        #    Variable(torch.zeros(self.batch_size, self.hidden_size).to(device))
                           ))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x):
        h = x
        for i in range(self.n_layer):
            self.hidden[i] = self.rnn_cell[i](h, self.hidden[i])
            h = self.hidden[i][0]
        mu = self.mu_net(h)
        logvar = self.logvar_net(h)
        if self.training:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        return mu, mu, logvar
        

class GM_Recurrent(nn.Module):

    def __init__(self, in_dim, out_dim, n_clusters, tau=1, hidden_dim=128, n_layer=1, mod='GRU', bid=False):
        super(GM_Recurrent, self).__init__()
        self.mod = mod
        self.tau = tau
        self.n_layer = n_layer
        self.n_clusters = n_clusters
        self.rnn = getattr(nn, mod)(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=bid)

        if bid:
            self.mu_net = nn.Linear(hidden_dim*2, out_dim)
            self.logvar_net = nn.Linear(hidden_dim*2, out_dim)

            self.c_net = nn.Linear(hidden_dim*2, n_clusters)
        else:
            self.mu_net = nn.Linear(hidden_dim, out_dim)
            self.logvar_net = nn.Linear(hidden_dim, out_dim)

            self.c_net = nn.Linear(hidden_dim, n_clusters)

    def reparameterize_gumbel(self, p):
        # Gumbel-Softmax for discrete sampling
        sampled_one_hot = F.gumbel_softmax(p, tau=self.tau, hard=True, dim=-1)
        return sampled_one_hot

    def reparameterize_gaussian(self, mu, logvar):
        # Gaussian sampling
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)
    
    def forward(self, x):
        h, _ = self.rnn(x)
        # gaussian inference
        mu = self.mu_net(h)
        logvar = self.logvar_net(h)
        # gumbel inference
        p = self.c_net(h)
        prob = F.log_softmax(p, dim=-1)
        if self.training:
            z = self.reparameterize_gaussian(mu, logvar)
            sampled_one_hot = self.reparameterize_gumbel(p)
            return z, mu, logvar, sampled_one_hot, prob
        size = prob.shape
        prob = prob.flatten(end_dim=1)
        one_hot = torch.zeros_like(prob)
        one_hot[torch.arange(prob.shape[0]), prob.argmax(dim=-1)] = 1
        prob, one_hot = prob.view(*size[:2], self.n_clusters), one_hot.view(*size[:2], self.n_clusters)
        return mu, mu, logvar, one_hot, prob
