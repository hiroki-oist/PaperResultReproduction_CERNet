import torch
from torch import nn
import time
import numpy as np
import pickle as pk
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torchvision
import os
import torch.nn.functional as F 

class CERNet(nn.Module):
    
    def __init__(self, 
                 causes_dim: int, 
                 states_dim: list[int], 
                 output_dim: int, 
                 tau_h: list[float], 
                 alpha_x: float, 
                 alpha_h: list[float],
                 save_every_n_iteration: int = None,
                 save_directory: str = None            
                ) -> None:
        
        super(CERNet, self).__init__()
        
        self.causes_dim = causes_dim
        self.output_dim = output_dim
        self.states_dim = states_dim
        self.tau_h = tau_h
        self.alpha_x = alpha_x
        self.alpha_h = alpha_h
        
        self.decay_epochs = 8000
        self.alpha_x_base = alpha_x        
        self.alpha_schedule = lambda e: max(0.0, self.alpha_x_base * (1 - e / self.decay_epochs))
        
        self.save_every_n_iteration = save_every_n_iteration
        self.save_directory = save_directory
        if self.save_directory is not None:
            os.makedirs(self.save_directory, exist_ok=True)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.t = 0
        
        self.last_h_post = [None] * len(states_dim)
        self.last_c      = None
        
        self.m_w_o = torch.zeros(output_dim, states_dim[0])
        self.v_w_o = torch.zeros(output_dim, states_dim[0])

        self.m_b_o = torch.zeros(output_dim)
        self.v_b_o = torch.zeros(output_dim)

        self.m_w_c = torch.zeros(states_dim[-1], causes_dim)
        self.v_w_c = torch.zeros(states_dim[-1], causes_dim)

        self.m_w_r = [torch.zeros(s, s) for s in states_dim]
        self.v_w_r = [torch.zeros(s, s) for s in states_dim]

        self.m_b_r = [torch.zeros(s) for s in states_dim]
        self.v_b_r = [torch.zeros(s) for s in states_dim]

        self.m_w_hh = [torch.zeros(states_dim[i], states_dim[i+1]) for i in range(len(states_dim) - 1)]
        self.v_w_hh = [torch.zeros(states_dim[i], states_dim[i+1]) for i in range(len(states_dim) - 1)]
        
        # Output weights initialization
        self.w_o = torch.randn(self.output_dim, self.states_dim[0]) / (self.states_dim[0] * 10)
        self.b_o = torch.randn(self.output_dim) / 100
        self.w_c = torch.randn(self.states_dim[-1], self.causes_dim) / (self.causes_dim * 10)
        self.w_r = nn.ParameterList([nn.Parameter(torch.randn(s, s) / (s * 10)) for s in states_dim])
        self.b_r = nn.ParameterList([nn.Parameter(torch.randn(s) / 100) for s in states_dim])
        self.w_hh = nn.ParameterList()
        for i in range(len(states_dim) - 1):
            w = nn.Parameter(torch.randn(states_dim[i], states_dim[i+1]) / (states_dim[i] * 10))
            self.w_hh.append(w)
        
        # Predictions, states and errors are temporarily stored for batch learning
        # Learning can be performed online, but computations are slower
        self.x_pred = None
        self.error = None
        self.h_prior = [None] * len(states_dim) 
        self.h_post  = [None] * len(states_dim)
        self.c = None
        
        self.predictions = np.zeros((100, self.output_dim)) # n_timesteps * output_dim
        self.observations = np.zeros((100, self.output_dim))
        self.online_h_posts = [torch.zeros(100, dim) for dim in self.states_dim]
        self.online_h_priors = [torch.zeros(100, dim) for dim in self.states_dim]
        
    def forward(self, x, c_init, h_init=0, epoch=0, store=True):
        """
        Pass through the network : forward (prediction) and backward (inference) passes are 
         performed at the same time. Online learning could be performed here, but to improve
         computations speed, we use the seq_len as a batch dimension in a separate function.
        Parameters :        
        - x : target sequences, Tensor of shape (seq_len, batch_size, output_dim)
        - h_init : states of the sequences, Tensor of shape (batch_size, states_dim)
        - store : whether to store the prediction as an object attribute, Boolean
        """
        
        self.alpha_x = self.alpha_schedule(epoch)
        seq_len, batch_size, _ = x.shape
        
        if h_init is None:
            h_init = [torch.zeros(batch_size, dim) for dim in self.states_dim]
        
        # Temporary storing of the predictions, states and errors
        if store:
            x_preds = torch.zeros(seq_len, batch_size, self.output_dim)
            cs = torch.zeros(seq_len, batch_size, self.causes_dim)
            h_priors = [torch.zeros(seq_len, batch_size, dim) 
                        for dim in self.states_dim]
            h_posts = [torch.zeros(seq_len, batch_size, dim) 
                    for dim in self.states_dim]
            error_hs = [torch.zeros(seq_len, batch_size, dim) 
                        for dim in self.states_dim]
        else:
            x_preds = None
            cs = None
            h_priors = None
            h_posts = None
            error_hs = None
        errors = torch.zeros(seq_len, batch_size, self.output_dim)
        
        # Initialise hidden state and hidden causes
        old_h_post = h_init
        c = c_init
        n_layers = len(self.states_dim)
        h_prior = [torch.zeros( batch_size, dim) for dim in self.states_dim]
        h_post = [torch.zeros (batch_size, dim) for dim in self.states_dim]
        error_h = [torch.zeros( batch_size, dim) for dim in self.states_dim]
        
        for t in range(seq_len):
            # Forward propagation
            # Top layer
            h_prior[n_layers - 1] = (1-1/self.tau_h[n_layers - 1]) * old_h_post[n_layers - 1] + (1/self.tau_h[n_layers - 1]) * (
                torch.mm(F.leaky_relu(old_h_post[n_layers - 1], negative_slope=0.01), self.w_r[n_layers - 1].T) + 
                torch.mm(c, self.w_c.T) + 
                self.b_r[n_layers - 1].unsqueeze(0).repeat(batch_size, 1)
            )
            if store:
                h_priors[n_layers - 1][t] = h_prior[n_layers - 1].detach()
                
            # Second layer to Bottom layer
            for layer in reversed(range(n_layers - 1)):
                # Compute h_prior according to past h_post and higher layer h
                h_prior[layer] = (1-1/self.tau_h[layer]) * old_h_post[layer] + (1/self.tau_h[layer]) * (
                    torch.mm(F.leaky_relu(old_h_post[layer], negative_slope=0.01), self.w_r[layer].T) + 
                    torch.mm(F.leaky_relu(h_prior[layer + 1], negative_slope=0.01), self.w_hh[layer].T) + 
                    self.b_r[layer].unsqueeze(0).repeat(batch_size, 1)
                )
                if store:
                    h_priors[layer][t] = h_prior[layer].detach()
            
            # Bottom layer: Compute x_pred according to h_prior
            x_pred = torch.mm(F.leaky_relu(h_prior[0], negative_slope=0.01), self.w_o.T) + self.b_o.unsqueeze(0).repeat(batch_size, 1)
            if store:
                x_preds[t] = x_pred.detach()
            # Compute the error on the sensory level
            error = x_pred - x[t]
            errors[t] = error
            
            # Backward Propagation
            if self.alpha_x > 0:
                # Bottom Layer
                # Infer h_post according to h_prior and the error on the sensory level
                derivative = torch.ones_like(h_prior[0])
                derivative[h_prior[0] < 0] = 0.01
                h_post[0] = h_prior[0] - self.alpha_x * derivative * torch.mm(error, self.w_o)
                if store:
                    h_posts[0][t] = h_post[0].detach()
                h_post[0] = torch.clamp(h_post[0], -2, 2)
                
                # Compute the error on the hidden state level
                error_h[0] = h_prior[0] - h_post[0]
                if store:
                    error_hs[0][t] = error_h[0].detach()
                    
                # Second bottom layer to Top layer
                for layer in range(1, n_layers):
                    derivative = torch.ones_like(h_prior[layer])
                    derivative[h_prior[layer] < 0] = 0.01
                    h_post[layer] = h_prior[layer] - self.alpha_h[layer-1] * derivative * torch.mm(error_h[layer - 1], self.w_hh[layer - 1])
                    error_h[layer] = h_prior[layer] - h_post[layer]
                    if store:
                        error_hs[layer][t] = error_h[layer].detach()

                # Top layer: Infer c according to its past value and the error on the hidden state level
                # c = c - self.alpha_h[n_layers - 1]*torch.mm(error_h[n_layers - 1], self.w_c)
                # As mentioned in the manuscript, c is NOT updated during the training
                
                old_h_post = h_post
            
            else:
                # print("skipping backprop")
                old_h_post = h_prior
                
            if store:
                cs[t] = c
                
        if store:
            self.error = errors.detach()
            self.x_pred = x_preds
            self.error_h = error_hs
            self.h_prior = h_priors
            self.h_post = h_posts
            self.c = cs
                                         
        return errors, x_preds
    
    def learn(self, lr_o: float, lr_h: list[float], lr_c: float,
          max_grad_norm: float = 1.0) -> None:
        """
        Adam optimiser + global norm clipping
        """
        self.t += 1  
        seq_len, batch_size, _ = self.x_pred.shape
        n_layers = len(self.states_dim)

        with torch.no_grad():
            # --- 勾配計算 ---
            activated_prior = F.leaky_relu(self.h_prior[0].reshape(seq_len*batch_size, self.states_dim[0]), negative_slope=0.01)
            delta_w_o = torch.mm(
                self.error.reshape(seq_len*batch_size, self.output_dim).T, 
                activated_prior
            )
            delta_b_o = torch.sum(self.error.reshape(seq_len*batch_size, self.output_dim), dim=0)
            delta_w_c = torch.mm(
                self.error_h[n_layers - 1].reshape(seq_len*batch_size, self.states_dim[n_layers - 1]).T,
                self.c.reshape(seq_len*batch_size, self.causes_dim)
            )

            # Recurrent Weights
            delta_w_hh_list = []
            for layer in range(n_layers - 1):
                activated_next_layer = F.leaky_relu(self.h_prior[layer + 1][:-1], negative_slope=0.01)
                delta_w_hh = torch.mm(
                    self.error_h[layer][1:].reshape((seq_len-1)*batch_size, self.states_dim[layer]).T,
                    activated_next_layer.reshape((seq_len-1)*batch_size, self.states_dim[layer + 1])
                )
                delta_w_hh_list.append(delta_w_hh)

            delta_w_r_list = []
            delta_b_r_list = []
            for layer in range(n_layers):
                activated_post = F.leaky_relu(self.h_post[layer][:-1], negative_slope=0.01)
                delta_w_r = torch.mm(
                    self.error_h[layer][1:].reshape((seq_len-1)*batch_size, self.states_dim[layer]).T,
                    activated_post.reshape((seq_len-1)*batch_size, self.states_dim[layer])
                )
                delta_b_r = torch.sum(self.error_h[layer].reshape(seq_len*batch_size, self.states_dim[layer]), dim=0)
                delta_w_r_list.append(delta_w_r)
                delta_b_r_list.append(delta_b_r)

            # global norm clipping
            all_grads = [
                delta_w_o, delta_b_o, delta_w_c,
            ]
            all_grads.extend(delta_w_hh_list)
            all_grads.extend(delta_w_r_list)
            all_grads.extend(delta_b_r_list)
            
            total_norm = 0.0
            for g in all_grads:
                total_norm += g.norm(2).item()**2
            total_norm = total_norm ** 0.5
            if total_norm > max_grad_norm:
                clip_coef = max_grad_norm / (total_norm + 1e-6)
            else:
                clip_coef = 1.0

            if clip_coef < 1.0:
                for i in range(len(all_grads)):
                    all_grads[i] = all_grads[i] * clip_coef

            delta_w_o = all_grads[0]
            delta_b_o = all_grads[1]
            delta_w_c = all_grads[2]

            idx = 3
            for layer in range(n_layers - 1):
                delta_w_hh_list[layer] = all_grads[idx]
                idx += 1

            for layer in range(n_layers):
                delta_w_r_list[layer] = all_grads[idx]
                idx += 1
            for layer in range(n_layers):
                delta_b_r_list[layer] = all_grads[idx]
                idx += 1
            # global norm clipping end

            # --- Compute Adam ---
            def adam_update(param, m, v, grad, lr):
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                param -= lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                return param, m, v

            self.w_o, self.m_w_o, self.v_w_o = adam_update(self.w_o, self.m_w_o, self.v_w_o, delta_w_o, lr_o)
            self.b_o, self.m_b_o, self.v_b_o = adam_update(self.b_o, self.m_b_o, self.v_b_o, delta_b_o, lr_o)
            self.w_c, self.m_w_c, self.v_w_c = adam_update(self.w_c, self.m_w_c, self.v_w_c, delta_w_c, lr_c)

            for layer in range(n_layers - 1):
                self.w_hh[layer], self.m_w_hh[layer], self.v_w_hh[layer] = adam_update(
                    self.w_hh[layer], self.m_w_hh[layer], self.v_w_hh[layer],
                    delta_w_hh_list[layer], lr_h[layer]
                )

            for layer in range(n_layers):
                self.w_r[layer], self.m_w_r[layer], self.v_w_r[layer] = adam_update(
                    self.w_r[layer], self.m_w_r[layer], self.v_w_r[layer],
                    delta_w_r_list[layer], lr_h[layer]
                )
                self.b_r[layer], self.m_b_r[layer], self.v_b_r[layer] = adam_update(
                    self.b_r[layer], self.m_b_r[layer], self.v_b_r[layer],
                    delta_b_r_list[layer], lr_h[layer]
                )

        if self.save_every_n_iteration is not None and (self.t % self.save_every_n_iteration == 0):
            self.save_snapshot(self.t)

            
    def save_snapshot(self, iteration: int):
        """
        Save model parameters and future prediction results in dictionary style
        every n iterations.
        """
        folder = os.path.join(self.save_directory, f"iteration_{iteration}")
        os.makedirs(folder, exist_ok=True)
        snapshot = {
            'iteration': iteration,
            'w_o': self.w_o.cpu().detach().numpy(),
            'b_o': self.b_o.cpu().detach().numpy(),
            'w_c': self.w_c.cpu().detach().numpy(),
            'w_r': [w.cpu().detach().numpy() for w in self.w_r],
            'b_r': [b.cpu().detach().numpy() for b in self.b_r],
            'w_hh': [w.cpu().detach().numpy() for w in self.w_hh],
            'x_pred': self.x_pred.cpu().detach().numpy() if self.x_pred is not None else None,
            'error': self.error.cpu().detach().numpy() if self.error is not None else None,
            'h_prior': [h.cpu().detach().numpy() for h in self.h_prior] if self.h_prior is not None else None,
            'h_post': [h.cpu().detach().numpy() for h in self.h_post] if self.h_post is not None else None,
            'c': self.c.cpu().detach().numpy() if self.c is not None else None,
            't': self.t,
        }
        with open(os.path.join(folder, "snapshot.pkl"), "wb") as f:
            pk.dump(snapshot, f)
    

    def vfe(self):
        """
        Computes the variational free-energy associated with the last seen values, up to a constant term
        """
        e_x = self.error 
        mse_x = torch.mean(torch.sum(e_x**2, dim=2))  # (seq_len,batch)->scalar
        vfe_x = 0.5 * self.alpha_x * mse_x
        
        vfe_h_sum = 0.0
        for i, e_h_i in enumerate(self.error_h):
            # e_h_i: (seq_len, batch_size, states_dim[i])
            mse_h_i = torch.mean(torch.sum(e_h_i**2, dim=2))  # scalar
            vfe_h_sum += 0.5 * self.alpha_h[i] * mse_h_i
        return vfe_x + vfe_h_sum
    
    def one_step_prediction(self, t, x_t=None, prev_x_pred=None, store=True, horizon: int = 20 ):
        """
        Using the last_h_post and self.last_c, it computes/predicts the next timestep's x_pred
        """
        if any(h is None for h in self.last_h_post):
            batch_size = 1 if x_t is None else x_t.shape[0]
            self.last_h_post = [torch.zeros(batch_size, dim) for dim in self.states_dim]

        n_layers = len(self.states_dim)
        error_h = [None] * n_layers
        
        # ---------- backward/inference (1 step分) ----------
        if (x_t is not None) and (self.alpha_x > 0) and (t != 0):
            error_t = prev_x_pred - x_t  # (batch_size, output_dim)
            h_post = [None] * n_layers

            # Bottom layer
            derivative = torch.ones_like(self.h_prior[0])
            derivative[self.h_prior[0] < 0] = 0.01
            h_post[0] = self.h_prior[0] - self.alpha_x * derivative * torch.mm(error_t, self.w_o)
            error_h[0] = self.h_prior[0] - h_post[0]

            # Middle to Top layer
            for layer in range(1, n_layers):
                # print(f"alpha_h[{layer}]: ", self.alpha_h[layer-1])
                derivative = torch.ones_like(self.h_prior[layer])
                derivative[self.h_prior[layer] < 0] = 0.01

                h_post[layer] = self.h_prior[layer] \
                    - self.alpha_h[layer - 1] * derivative \
                    * torch.mm(error_h[layer - 1], self.w_hh[layer - 1])
                error_h[layer] = self.h_prior[layer] - h_post[layer]
                # print(f"error_h[{layer}]: ", error_h[layer])
            
            # Update c at the top layer
            # self.c = self.c - self.alpha_h[n_layers - 1]*torch.mm(error_h[n_layers - 1], self.w_c)

            if store:
                self.last_h_post = h_post
        elif (t != 0):
            if store:
                self.last_h_post = self.h_prior
                
        old_h_post = self.last_h_post
        
        # ---------- 1 step forward propagation ----------
        # top layer
        self.h_prior[n_layers - 1] = (1 - 1/self.tau_h[n_layers - 1]) * old_h_post[n_layers - 1] \
                                + (1/self.tau_h[n_layers - 1]) * (
                                    torch.mm(F.leaky_relu(old_h_post[n_layers - 1], negative_slope=0.01), self.w_r[n_layers - 1].T)
                                    + torch.mm(self.c, self.w_c.T)
                                    + self.b_r[n_layers - 1].unsqueeze(0)
                                )
        
        for layer in reversed(range(n_layers - 1)):
            self.h_prior[layer] = (1 - 1/self.tau_h[layer]) * old_h_post[layer] \
                             + (1/self.tau_h[layer]) * (
                                 torch.mm(F.leaky_relu(old_h_post[layer], negative_slope=0.01), self.w_r[layer].T)
                                 + torch.mm(F.leaky_relu(self.h_prior[layer + 1], negative_slope=0.01), self.w_hh[layer].T)
                                 + self.b_r[layer].unsqueeze(0)
                               )
        
        x_pred = torch.mm(F.leaky_relu(self.h_prior[0], negative_slope=0.01), self.w_o.T) + self.b_o.unsqueeze(0)
                
        return x_pred, error_h
    
    @torch.no_grad()
    def predict_horizon(self, t: int, prev_x_pred: torch.Tensor, steps: int = 20):
        """
        現在時刻 t の内部状態から steps ステップ先まで
        オープンループ予測し、全ステップ分を返す。
        Given internal states at timestep t, computes open-loop prediction for {steps} step future.
        """
        h_bk   = [h.clone() for h in self.h_prior]
        c_bk   = self.c.clone()
        post_bk = [h.clone() for h in self.last_h_post]

        preds = []                    
        cur_pred = prev_x_pred
        for k in range(steps):
            preds.append(cur_pred)     
            cur_pred, _ = self.one_step_prediction(
                t + k + 1,
                x_t=None,             
                prev_x_pred=cur_pred,
                store=True             
            )

        self.h_prior      = h_bk
        self.c            = c_bk
        self.last_h_post  = post_bk

        return torch.stack(preds, dim=0) 
    
    def reset_state(self, batch_size:int = 1, class_id:int|None = None, rand_seed: int = 1):
        torch.manual_seed(rand_seed)
        self.last_h_post = [torch.zeros(batch_size, d) for d in self.states_dim]
        if class_id is None:
            # self.c = torch.zeros(batch_size, self.causes_dim)
            self.c = torch.randn(batch_size, self.causes_dim, device=self.w_o.device) * 0.05
        else:
            self.c = F.one_hot(
                torch.tensor([class_id]*batch_size), num_classes=self.causes_dim
            ).float()

    @torch.no_grad()
    def step(self, t:int, x_t: torch.Tensor, prev_x_pred: torch.Tensor, feedback: bool = True):
        future_traj = [0]
        if not feedback:         # α=0 
            alpha_backup, self.alpha_x = self.alpha_x, 0.0
            x_pred, error_h = self.one_step_prediction(t, x_t, prev_x_pred, store=True)
            self.alpha_x = alpha_backup
            future_traj = self.predict_horizon(
                t=t,
                prev_x_pred=x_pred,  
                steps=20            
            )
        else:
            x_pred, error_h = self.one_step_prediction(t, x_t, prev_x_pred, store=True)
            future_traj = self.predict_horizon(
                t=t,
                prev_x_pred=x_pred,   
                steps=100 - t        
            )
            
        return x_pred, future_traj, error_h
    
    def past_reconstruction(self, t: int):
        n_layers = len(self.states_dim)
        delta_c_sum = 0
        mse_sum = 0
        
        for timestep in range(t):
            prev_x_pred = self.predictions[timestep]
            x_t = self.observations[timestep]
            h_prior = [prior[timestep] for prior in self.online_h_priors]
        
            # Update C and hiddens at each t to minimise error
            error_t = torch.tensor(prev_x_pred - x_t, dtype=torch.float32).unsqueeze(0)  # (batch_size, output_dim)
            mse_sum += torch.mean(error_t.pow(2)).item()
            h_post = [None] * n_layers
            error_h = [None] * n_layers

            derivative = torch.ones_like(h_prior[0])
            derivative[h_prior[0] < 0] = 0.01
            h_post[0] = h_prior[0] - self.alpha_x * derivative * torch.mm(error_t, self.w_o)
            error_h[0] = h_prior[0] - h_post[0]

            for layer in range(1, n_layers):
                derivative = torch.ones_like(h_prior[layer])
                derivative[h_prior[layer] < 0] = 0.01

                h_post[layer] = h_prior[layer] \
                    - self.alpha_h[layer - 1] * derivative \
                    * torch.mm(error_h[layer - 1], self.w_hh[layer - 1])
                error_h[layer] = h_prior[layer] - h_post[layer]
            
            delta_c_sum += torch.mm(error_h[n_layers-1], self.w_c)         
        
        self.c = self.c - (self.alpha_h[n_layers - 1] / t) * delta_c_sum
        
        return mse_sum, delta_c_sum
        
    def prediction(self, t, until_end = False):
        n_layers = len(self.states_dim)
        batch_size = 1
        x_preds = np.zeros((100, self.output_dim), dtype=np.float32)
        
        # for timestep in range(t + 1): 
        until_t = t + 1 if until_end == False else 100           
        for timestep in range(until_t):            
            # Set old_h_post which is the input for current timestep from the hidde layer of the previous timestep
            if any(h is None for h in self.last_h_post) or timestep == 0:
                batch_size = 1 
                self.last_h_post = [torch.zeros(batch_size, dim) for dim in self.states_dim]
            if timestep != 0:
                self.last_h_post = self.h_prior 
            old_h_post = self.last_h_post
            
            # ---------- one step forward ----------
            # top layer
            self.h_prior[n_layers - 1] = (1 - 1/self.tau_h[n_layers - 1]) * old_h_post[n_layers - 1] \
                                    + (1/self.tau_h[n_layers - 1]) * (
                                        torch.mm(F.leaky_relu(old_h_post[n_layers - 1], negative_slope=0.01), self.w_r[n_layers - 1].T)
                                        + torch.mm(self.c, self.w_c.T)
                                        + self.b_r[n_layers - 1].unsqueeze(0)
                                    )
            
            for layer in reversed(range(n_layers - 1)):
                self.h_prior[layer] = (1 - 1/self.tau_h[layer]) * old_h_post[layer] \
                                + (1/self.tau_h[layer]) * (
                                    torch.mm(F.leaky_relu(old_h_post[layer], negative_slope=0.01), self.w_r[layer].T)
                                    + torch.mm(F.leaky_relu(self.h_prior[layer + 1], negative_slope=0.01), self.w_hh[layer].T)
                                    + self.b_r[layer].unsqueeze(0)
                                )
            
            x_pred = torch.mm(F.leaky_relu(self.h_prior[0], negative_slope=0.01), self.w_o.T) + self.b_o.unsqueeze(0)
            
            
            x_preds[timestep, :] = x_pred.squeeze(0).cpu().numpy()  
                
            for l in range(n_layers):
                self.online_h_priors[l][timestep] = self.h_prior[l].squeeze(0).detach()
            
        return x_preds
    
    @torch.no_grad()
    def detect_class(self, t: int, x_t: torch.Tensor, feedback: bool = True):
        mse_sum = 0
        n_iterations = 15
        if t > 0:
            self.observations[t-1] = x_t
            for i in range(n_iterations):
                mse_sum, delta_c_sum = self.past_reconstruction(t)    
                self.predictions = self.prediction(t) if i != n_iterations - 1 else self.prediction(t, True)
                total = delta_c_sum.abs().sum().item()
                if total < 1.0e-7:
                    print("iteration:", i)
                    print("mse_sum: ", mse_sum)
                    print("delta_c_sum: ", total)
                    print("C optimised enough! ")
                    break
                if i == 0 or i == 19:
                    print("iteration:", i)
                    print("mse_sum: ", mse_sum)
                    print("delta_c_sum: ", total)
                
        else: 
            self.predictions = self.prediction(t) 
            
        return mse_sum, self.predictions
    