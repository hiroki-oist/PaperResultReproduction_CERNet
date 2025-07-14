import os
import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from dotenv import load_dotenv
load_dotenv()

from model.CERNet import CERNet
from utils import u2_tomlConfig as toml_config
from utils.x1_path_utils import resolve_path

def run_task(config_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = toml_config.TOMLConfigLoader(config_path)

    dataset_path = resolve_path(cfg["dataset"]["dataset_path"])
    causes_dim   = cfg["training"]["causes_dim"]
    states_dim   = cfg["training"]["states_dim"]
    output_dim   = cfg["training"]["output_dim"]
    tau_h        = cfg["training"]["tau_h"]
    alpha_x      = cfg["training"]["alpha_x"]
    alpha_h      = cfg["training"]["alpha_h"]
    iterations   = cfg["training"]["iterations"]
    save_every_n_iteration   = cfg["training"]["save_every_n_iteration"]

    lr_o = cfg["training"]["lr"]["lr_o"]
    lr_h = cfg["training"]["lr"]["lr_h"]
    lr_c = cfg["training"]["lr"]["lr_c"]

    train_save_dir = resolve_path(cfg["training"]["save_directory"])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Data"))
    base_save_dir = os.environ.get("CERNET_SAVE_DIR", default_base_dir)
    full_save_path = os.path.join(base_save_dir, train_save_dir, f"seed{seed}")
    os.makedirs(full_save_path, exist_ok=True)
    if os.path.exists(full_save_path):
        shutil.rmtree(full_save_path)
    os.makedirs(full_save_path, exist_ok=True)

    data_np = np.load(dataset_path)  # shape: (num_seq, seq_len, output_dim)
    print("Loaded data shape:", data_np.shape)

    num_seq, seq_len, out_dim = data_np.shape
    if out_dim != output_dim:
        raise ValueError(f"Mismatch: config output_dim={output_dim} but data has {out_dim}.")
    
    data_min = data_np.min()
    data_max = data_np.max()

    data_np_normalized = -0.9 + (data_np - data_min) * (1.8 / (data_max - data_min))

    data_tensor = torch.tensor(data_np_normalized, dtype=torch.float32).permute(1, 0, 2)

    pcnet = CERNet(
        causes_dim = causes_dim,
        states_dim = states_dim,
        output_dim = output_dim,
        tau_h      = tau_h,
        alpha_x    = alpha_x,
        alpha_h    = alpha_h,
        save_every_n_iteration=save_every_n_iteration,
        save_directory=full_save_path
    )

    vfes   = torch.zeros(iterations, 2)
    losses = torch.zeros(iterations)
    free_losses = torch.zeros(iterations)   

    h_init = [torch.zeros(num_seq, d) for d in states_dim]
    c_init = torch.eye(causes_dim)
    x_preds = None

    for epoch in tqdm(range(iterations)):

        # forward
        x_error, x_preds = pcnet.forward(data_tensor, c_init, h_init, epoch=epoch)

        loss = torch.mean(torch.sum(x_error**2, dim=2))  
        losses[epoch] = loss.item()

        pcnet.learn(lr_o, lr_h, lr_c)

        vfes[epoch] = pcnet.vfe()
 
        print(
            f"Epoch:{epoch:5d} "
            f"train_loss:{loss.item():.4e}  "
            f"VFE:{vfes[epoch].sum().item():.4e}"
        )
            
    
    print(x_preds.shape)

    fig_loss = plt.figure()
    plt.plot(losses, label="corrected-loss")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction error (avg)')
    plt.title('Training Loss (entire dataset)')
    loss_plot_path = os.path.join(full_save_path, "losses.png")
    fig_loss.savefig(loss_plot_path)

    fig_vfe = plt.figure()
    plt.plot(vfes[:, 0], label="VFE_x")
    plt.plot(vfes[:, 1], label="VFE_h")
    plt.plot(torch.sum(vfes, dim=1), label="VFE_sum")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('VFE')
    plt.legend()
    plt.title('Variational Free Energy (entire dataset)')
    vfe_plot_path = os.path.join(full_save_path, "vfe.png")
    fig_vfe.savefig(vfe_plot_path)
    
    fig_x_preds = plt.figure()
    plt.plot(x_preds[:, 0, :], label = "output")
    plt.plot(data_np_normalized[0, :, :], label = "target", color="black")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Output and target')
    plt.title('Output and Target of last epoch')

    torch.save(losses, os.path.join(full_save_path, "losses.pt"))
    torch.save(vfes,   os.path.join(full_save_path, "vfes.pt"))
    np.save(os.path.join(full_save_path, "losses.npy"), losses.cpu().numpy())
    np.save(os.path.join(full_save_path, "vfes.npy"),   vfes.cpu().numpy())

    model_save_path = os.path.join(full_save_path, "pcnet_state_dict.pth")
    torch.save(pcnet.state_dict(), model_save_path)

    print(f"\nAll training results have been saved to: {full_save_path}\n")
    
    colors=['tab:blue', 'tab:orange', 'tab:green']
    labels=['a', 'b', 'c']
    
    tmp = pcnet.alpha_x
    pcnet.alpha_x = 0.
    _, x_preds_after_training = pcnet.forward(data_tensor, c_init, h_init)
    pcnet.alpha_x = tmp
    
    fig, ax = plt.subplots(figsize=(7, 2.5))

    timesteps = np.arange(data_tensor.shape[0])
    timesteps_2 = np.arange(100)

    colors = ['blue', 'red', 'green'] 

    for k in range(1):
        ax.plot(timesteps, data_tensor[:, k, 0], '--', label='Observed x', c='black')  
        ax.plot(timesteps, data_tensor[:, k, 1], '--', label='Observed y', c='gray')  
        
        ax.plot(timesteps_2, pcnet.x_pred[:, k, 0], label='Predicted x', c=colors[k])  
        ax.plot(timesteps_2, pcnet.x_pred[:, k, 1], label='Predicted y', linestyle="dotted", c=colors[k])  
        
        ax.plot(timesteps, x_preds_after_training[:, k, 0], label='Prediction from pcnet forward return')
        ax.plot(timesteps, x_preds_after_training[:, k, 1], label='Prediction from pcnet forward return')
        
    ax.set_xlabel("Time step") 
    ax.set_ylabel("Value")  
    ax.grid(True)
    ax.legend()

    # plt.show()
    
    min_loss_value = torch.min(losses).item()
    min_loss_epoch = torch.argmin(losses).item()
    print(f"\nLowest Loss: {min_loss_value:.6f} at Epoch {min_loss_epoch}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run PC_RNN_HC_A training with config.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration toml file')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    run_task(args.config, args.seed)

if __name__ == "__main__":
    main()
