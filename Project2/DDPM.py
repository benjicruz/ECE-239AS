import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet

device = torch.device("mps" if torch.mps.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        beta_t = beta_1 + (beta_T - beta_1) * ((t_s - 1) / (T - 1)) # 1
        sqrt_beta_t = torch.sqrt(beta_t) # 2
        alpha_t = 1 - beta_t #3
        oneover_sqrt_alpha = torch.divide(1, torch.sqrt(alpha_t)) #5

        time = torch.arange(0, T + 1, device=device)
        beta_bar = beta_1 + (beta_T - beta_1) * ((time) / (T - 1))
        alpha_s = 1 - beta_bar
        alpha_t_bar_temp = torch.cumprod(alpha_s, dim=0) # 5
        alpha_t_bar = alpha_t_bar_temp[t_s - 1] # 6
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar) # 4
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)
        





        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  
        # ==================================================== #
        
        # Step 0, set variables
        p_uncond = self.dmconfig.mask_p
        mask_value = self.dmconfig.condition_mask_value
        B = images.shape[0]
        num_classes = self.dmconfig.num_classes
        device = images.device
        #print(p_uncond, mask_value, B, num_classes)

        # Step 1, get the samples, theyre given! 

        # Step 2, do masking on the probability, also do one hot encoding here
        one_hot = F.one_hot(conditions, num_classes).float().to(device)
        mask = torch.rand(B, device=device) < p_uncond
        one_hot[mask] = mask_value

        # Step 3 Sample T, get scheduler values too
        t_s = torch.randint(1, T+1, (B,), device = device)
        scheduler_values = self.scheduler(t_s)
        sqrt_alpha = scheduler_values['sqrt_alpha_bar']
        sqrt_one_minus = scheduler_values['sqrt_oneminus_alpha_bar']
        # Step 4, sample noise
        episilon = torch.randn_like(images)

        # Step 5, corrupt data to sampled time steps

        #print(sqrt_alpha.shape, images.shape, sqrt_one_minus.shape, episolon.shape)
        x_t = sqrt_alpha.view(B, 1, 1, 1) * images + sqrt_one_minus.view(B, 1, 1, 1) * episilon

        #Step 6, gradient descent
        network_output = self.network(x_t, t_s/T, one_hot)
        noise_loss = self.loss_fn(network_output, episilon)

        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        # Step 0, set variables
        B = conditions.shape[0]
        num_classes = self.dmconfig.num_classes
        device = conditions.device
        one_hot_cond = F.one_hot(conditions, num_classes).float()
        one_hot_uncond = torch.full_like(one_hot_cond, fill_value=-1.0).float()
        #Step 1, sample x_T
        X_t = torch.randn(B, 1, 28, 28, device=device)

        # Step 2, for loop from T to 1
        with torch.no_grad():
            for t in range(T, 0, -1):
                #step 3 sample z
                if t > 1: 
                    z = torch.randn_like(X_t)
                else:
                    z = torch.zeros_like(X_t)
                #step 4 get e_t
                schedule_input = (torch.ones(B) * t).long().to(device)
                schedule_values = self.scheduler(schedule_input)
                esp_c = self.network(X_t, schedule_input/T, one_hot_cond)
                esp_uc = self.network(X_t, schedule_input/T, one_hot_uncond)
                e_t = (1 + omega) * esp_c - omega * esp_uc

                #step 5
                oneover_sqrt_alpha = schedule_values['oneover_sqrt_alpha'].view(B, 1, 1, 1)
                alpha_t = schedule_values['alpha_t'].view(B, 1, 1, 1)
                sqrt_one_minus = schedule_values['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)
                beta_t = schedule_values['beta_t'].view(B, 1, 1, 1)
                X_t = oneover_sqrt_alpha * (X_t - (1 - alpha_t) / sqrt_one_minus * e_t) + torch.sqrt(beta_t) * z
            
        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images