import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

#code adapted from https://github.com/tanelp/tiny-diffusion

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print(device)


class SinusoidalEmbedding(nn.Module):
    
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = emb.to(device, dtype=torch.float)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1
      
      
      
class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

    def __len__(self):
        return 1
      

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size: int, num_timesteps: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(emb_size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(num_timesteps, **kwargs)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)



class MLP(nn.Module):
    def __init__(self, nin, nout, num_timesteps, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        
        self.time_mlp = PositionalEmbedding(emb_size, num_timesteps, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, num_timesteps, input_emb)
        
        self.net = []
        #hs = [nin] + hidden_sizes + [nout]
        
        if time_emb == 'sinusoidal':
          t_size = emb_size
        else:
          t_size = 1
          
        if input_emb == 'sinusoidal':
          i_size = emb_size
        else:
          i_size = nin 
          
        input_size = t_size + i_size
        
        hs = [input_size]
        
        for i in range(0,hidden_layers):
          hs.append(hidden_size)
          
        hs.append(nout)
        
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])
                
        self.net.pop() # pop ReLU
        
        self.net = nn.Sequential(*self.net)
        
        

    def forward(self, x, t):
      
        x_emb = self.input_mlp(x)    
        t_emb = self.time_mlp(t)
                
        x = torch.cat((x_emb, t_emb),dim=-1).to(device, dtype=torch.float)
                                
        return self.net(x)



class MLP_dist_emb(nn.Module):
    def __init__(self, ndist, ndist_emb, ncoords, nout, num_timesteps, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.proj = nn.Linear(ndist, ndist_emb)
        self.time_mlp = PositionalEmbedding(emb_size, num_timesteps, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, num_timesteps, input_emb)
        
        self.net = []
        #hs = [nin] + hidden_sizes + [nout]
        
        if time_emb == 'sinusoidal':
          t_size = emb_size
        else:
          t_size = 1
          
        if input_emb == 'sinusoidal':
          i_size = emb_size
        else:
          i_size = ncoords  
          
        input_size = t_size + i_size + ndist_emb 
        
        hs = [input_size]
        
        for i in range(0,hidden_layers):
          hs.append(hidden_size)
          
        hs.append(nout)
        
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])
                
        self.net.pop() # pop ReLU
        
        self.net = nn.Sequential(*self.net)
        
        

    def forward(self, d, x, t):
      
        d_emb = self.proj(d)
        x_emb = self.input_mlp(x)    
        t_emb = self.time_mlp(t)
                
        x = torch.cat((d_emb, x_emb, t_emb),dim=-1).to(device, dtype=torch.float)
                                
        return self.net(x)



class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.1,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2


        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
                
        #guided diffusion 
        self.sqrt_one_minus_alphas_cumprod_prev = (1 - self.alphas_cumprod_prev) ** 0.5
        
        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.alphas = self.alphas.to(device, dtype=torch.float)
        self.alphas_cumprod = self.alphas_cumprod.to(device, dtype=torch.float)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device, dtype=torch.float)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device, dtype=torch.float)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device, dtype=torch.float)
        self.sqrt_one_minus_alphas_cumprod_prev = self.sqrt_one_minus_alphas_cumprod_prev.to(device, dtype=torch.float)
        self.betas = self.betas.to(device, dtype=torch.float)
        self.sqrt_inv_alphas_cumprod = self.sqrt_inv_alphas_cumprod.to(device, dtype=torch.float)
        self.sqrt_inv_alphas_cumprod_minus_one = self.sqrt_inv_alphas_cumprod_minus_one.to(device, dtype=torch.float)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device, dtype=torch.float)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device, dtype=torch.float)
        
    def get_noise_w_gradient(self, noise, t, gradient, scale_factor):
        s1 = self.sqrt_one_minus_alphas_cumprod[t]
        s1 = s1.reshape(-1,1)
        return noise - s1*scale_factor*gradient
    
    def step_guided(self, x_t, t, noise_w_gradient):
        s1 = self.alphas_cumprod_prev[t]
        s2 = self.sqrt_one_minus_alphas_cumprod[t]
        s3 = self.sqrt_alphas_cumprod[t]
        s4 = self.sqrt_one_minus_alphas_cumprod_prev[t]
        s1 = s1.reshape(-1,1)
        s2 = s2.reshape(-1,1)
        s3 = s3.reshape(-1,1)
        s4 = s4.reshape(-1,1)
        return s1*((x_t-(s2*noise_w_gradient))/s3) + s4*noise_w_gradient
    
    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, t, sample):
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps




def eval_loss(model, noise_scheduler, sample_conditional, test_loader):

    model.eval()
    test_losses = [] 
    
    with torch.no_grad():

        for step, x in enumerate(test_loader):
            
            if step % 1000 == 0:
                print(step)
            
            xyz_coeff = x[0]
            pairwise_dist = x[1]
                        
            xyz_coeff = xyz_coeff.to(device, dtype=torch.float)
            pairwise_dist = pairwise_dist.to(device, dtype=torch.float)

            noise = torch.randn(xyz_coeff.shape).to(device, dtype=torch.float)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (xyz_coeff.shape[0],)).to(device, dtype=torch.long)
            noisy = noise_scheduler.add_noise(xyz_coeff, noise, timesteps)

            if sample_conditional:
                noisy = torch.cat((noisy,pairwise_dist), axis=1)

            noise_pred = model.forward(noisy, timesteps)
            loss = F.l1_loss(noise_pred, noise)
            test_losses.append(loss.detach().item())


    return (np.mean(test_losses))


def eval_loss_w_demb(model, noise_scheduler, sample_conditional, test_loader):

    model.eval()
    test_losses = [] 
    
    with torch.no_grad():

        for step, x in enumerate(test_loader):
            
            if step % 1000 == 0:
                print(step)
            
            xyz_coeff = x[0]
            pairwise_dist = x[1]
                        
            xyz_coeff = xyz_coeff.to(device, dtype=torch.float)
            pairwise_dist = pairwise_dist.to(device, dtype=torch.float)

            noise = torch.randn(xyz_coeff.shape).to(device, dtype=torch.float)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (xyz_coeff.shape[0],)).to(device, dtype=torch.long)
            noisy = noise_scheduler.add_noise(xyz_coeff, noise, timesteps)

            noise_pred = model.forward(pairwise_dist, noisy, timesteps)

            loss = F.l1_loss(noise_pred, noise)
            test_losses.append(loss.detach().item())


    return (np.mean(test_losses))


def eval_loss_w_demb_independent_dim(model, noise_scheduler, sample_conditional, test_loader, idx_start, idx_end):

    model.eval()
    test_losses = [] 
    
    with torch.no_grad():

        for step, x in enumerate(test_loader):
            
            if step % 1000 == 0:
                print(step)
            
            xyz_coeff = x[0]
            pairwise_dist = x[1]
                        
            xyz_coeff = xyz_coeff.to(device, dtype=torch.float)
            pairwise_dist = pairwise_dist.to(device, dtype=torch.float)

            xyz_coeff_dimi = xyz_coeff[:,idx_start:idx_end]

            noise = torch.randn(xyz_coeff_dimi.shape).to(device, dtype=torch.float)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (xyz_coeff_dimi.shape[0],)).to(device, dtype=torch.long)
            noisy = noise_scheduler.add_noise(xyz_coeff_dimi, noise, timesteps)

            noise_pred = model.forward(pairwise_dist, noisy, timesteps)

            loss = F.l1_loss(noise_pred, noise)
            test_losses.append(loss.detach().item())


    return (np.mean(test_losses))





def apply_reverse_norm(data, mean_data, std_data):
	return ((data*std_data) + mean_data)

def apply_norm(data, mean_data, std_data):
	return (data - mean_data)/std_data

