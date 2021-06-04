
import torch
import torch.nn as nn
import torch.nn.functional as F

def recon(output, view, normalize=False):
    return F.mse_loss(output, view)

# def pred(output, view, normalize=False):
#     if normalize:
#         beta = view.std((0,1))
#         return F.mse_loss(output[:, :-1]/beta, view[:, 1:]/beta)
#     if output.shape[1] == view.shape[1]:
#         return F.mse_loss(output[:, :-1], view[:, 1:])
#     elif output.shape[1] == view.shape[1]-1:
#         return F.mse_loss(output, view[:, 1:])

def ct_sim(content):
    return F.mse_loss(content.mean(dim=1, keepdim=True), content)

def semi_cls(output, states, print_acc=False):
    # states: list of dicts
    labeled_frames, tar_states = get_labeled_frames(output, states)
    if len(labeled_frames) == 0:
        return torch.tensor(0.)
    labeled_frames = torch.cat(labeled_frames, dim=0)
    tar_states = torch.LongTensor(tar_states).to(output.device)
    if print_acc:
        acc = (labeled_frames.argmax(dim=1) == tar_states).sum().float() / len(tar_states)
        print('state acc: ', acc.item())
    return F.cross_entropy(labeled_frames, tar_states)

def get_labeled_frames(output, states):
    labeled_frames, tar_states = [], []
    for i, vid in enumerate(states):
        for tar_state, state_indices in vid.items():
            labeled_frames.append(output[i, state_indices])
            tar_states += [tar_state] * len(state_indices)
    return labeled_frames, tar_states

def kl(mu, logvar):
    # kld of fixed prior
    mu = mu.flatten(end_dim=1) if mu.dim()==3 else mu
    logvar = logvar.flatten(end_dim=1) if logvar.dim()==3 else logvar
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= mu.shape[0]
    return KLD

def kl_general(mu1, logvar1, mu2, logvar2):
    # kld of learned prior
    mu1 = mu1.flatten(end_dim=1) if mu1.dim()==3 else mu1
    mu2 = mu2.flatten(end_dim=1) if mu2.dim()==3 else mu2
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    KLD = kld.sum() / mu1.shape[0]
    return KLD

def kl_cat(prob):
    # kld of uniform prior
    prob = prob.flatten(end_dim=1) if prob.dim()==3 else prob
    c = prob.shape[1]
    KLD = torch.sum(prob * (prob * c).log())
    KLD /= prob.shape[0]
    return KLD

def kl_cat_log(prob):
    # kld of uniform prior
    prob = prob.flatten(end_dim=1) if prob.dim()==3 else prob
    c = prob.shape[1]
    KLD = torch.sum(prob.exp() * prob)
    KLD /= prob.shape[0]
    return KLD

def kl_cat_general_log(prob1, prob2):
    # kld of learned prior
    prob1 = prob1.flatten(end_dim=1) if prob1.dim()==3 else prob1
    prob2 = prob2.flatten(end_dim=1) if prob2.dim()==3 else prob2
    KLD = torch.sum(prob1 * (prob1 / prob2).log())
    KLD /= prob1.shape[0]
    return KLD

def kl_cat_general_log(prob1, prob2):
    # kld of learned prior probs are log_softmax
    prob1 = prob1.flatten(end_dim=1) if prob1.dim()==3 else prob1
    prob2 = prob2.flatten(end_dim=1) if prob2.dim()==3 else prob2
    KLD = torch.sum(prob1.exp() * (prob1 - prob2))
    KLD /= prob1.shape[0]
    return KLD

def vq(mu, quantized, commitment_cost=0.25):
    e_latent_loss = F.mse_loss(quantized.detach(), mu)
    q_latent_loss = F.mse_loss(quantized, mu.detach())
    return q_latent_loss + commitment_cost * e_latent_loss

def kmeans(centroids):
    # encourage separation between centroids (vector quantization)
    z_dim = centroids.shape[1]
    return -torch.cdist(centroids, centroids, p=1).mean() * z_dim / 2
    # return -(torch.cdist(centroids, centroids, p=2)**2).mean() * z_dim / 2

def svd(z, k=30, lmbda=1):
    batch_size = z.shape[0]
    gram_matrix = (z @ z.T) / batch_size
    _ ,sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:k])
    return lmbda * torch.sum(sv)
    
def beta_annealing(epoch, step=1000):
    return (epoch+1)/step if epoch<step else 1

def entropy_annealing(epoch, step=1000):
    return 1 - epoch/step if epoch<step else 0

def smoothing_annealing(epoch, step=1000):
    return 1 - epoch/step if epoch<step else 0