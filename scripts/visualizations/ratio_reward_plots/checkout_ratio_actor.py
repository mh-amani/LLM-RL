import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def extract_trajectory_stats(attempted_ratios_list):
    """Convert raw trajectory data to padded matrices for portion and reward."""
    num_samples = len(attempted_ratios_list)
    max_iters = max(len(sample) for sample in attempted_ratios_list)

    portion_matrix = np.full((num_samples, max_iters), np.nan)
    reward_matrix = np.full((num_samples, max_iters), np.nan)

    for i, sample_history in enumerate(attempted_ratios_list):
        for t, entry in enumerate(sample_history):
            portion_matrix[i, t] = entry['portion']
            reward_matrix[i, t] = np.mean(entry['reward'])
        reward_matrix[i, t:] = reward_matrix[i, t]
        portion_matrix[i, t:] = portion_matrix[i, t]
    return portion_matrix, reward_matrix

def plot_portion_and_reward_trajectories(portion_matrix, reward_matrix, sample_ids=None, path=None):
    """Plot two subplots showing trajectories for portion and reward with mean ± std and selected samples."""
    num_samples, max_iters = portion_matrix.shape
    iterations = np.arange(max_iters)

    # Compute mean and std (ignore NaNs)
    portion_mean = np.nanmean(portion_matrix, axis=0)
    portion_std = np.nanstd(portion_matrix, axis=0)
    reward_mean = np.nanmean(reward_matrix, axis=0)
    reward_std = np.nanstd(reward_matrix, axis=0)

    if sample_ids is None:
        sample_ids = random.sample(range(num_samples), min(10, num_samples))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot portion
    axes[0].plot(iterations, portion_mean, label='Mean Portion', linestyle='--')
    axes[0].fill_between(iterations, portion_mean - portion_std, portion_mean + portion_std, alpha=0.2)
    for sid in sample_ids:
        axes[0].plot(iterations, portion_matrix[sid], alpha=0.7, linewidth=1)
    axes[0].set_ylabel("Supervision Ratio (Portion)")
    axes[0].set_title("Supervision Ratio Over Iterations")
    axes[0].legend()
    axes[0].grid(True)

    # Plot reward
    axes[1].plot(iterations, reward_mean, label='Mean Reward', linestyle='--', color='tab:green')
    axes[1].fill_between(iterations, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color='tab:green')
    for sid in sample_ids:
        axes[1].plot(iterations, reward_matrix[sid], alpha=0.7, linewidth=1, color='tab:green')
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Reward Over Iterations")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
    plt.show()


def plot_mean_trajectories_only(portion_matrix, reward_matrix, path=None):
    """Plot only the mean ± std for portion and reward in a single figure with two subplots."""
    iterations = np.arange(portion_matrix.shape[1])

    portion_mean = np.nanmean(portion_matrix, axis=0)
    portion_std = np.nanstd(portion_matrix, axis=0)

    reward_mean = np.nanmean(reward_matrix, axis=0)
    reward_std = np.nanstd(reward_matrix, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    lower_portion_bound = np.clip(portion_mean - portion_std, 0, 1)
    upper_portion_bound = np.clip(portion_mean + portion_std, 0, 1)

    lower_reward_bound = np.clip(reward_mean - reward_std, 0, 1)
    upper_reward_bound = np.clip(reward_mean + reward_std, 0, 1)

    # Portion subplot
    axes[0].plot(iterations, portion_mean, label='Mean Portion', linestyle='--')
    axes[0].fill_between(iterations, lower_portion_bound, upper_portion_bound, alpha=0.2)
    axes[0].set_ylabel("Supervision Ratio (Portion)")
    axes[0].set_title("Mean Supervision Ratio Over Iterations")
    axes[0].legend()
    axes[0].grid(True)

    # Reward subplot
    axes[1].plot(iterations, reward_mean, label='Mean Reward', linestyle='--', color='tab:green')
    axes[1].fill_between(iterations, lower_reward_bound, upper_reward_bound, alpha=0.2, color='tab:green')
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Mean Reward Over Iterations")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
    plt.show()

def plot_combined_mean_trajectory_paper_style(portion_matrix, reward_matrix, path_pdf=None):
    """Plot both mean portion and reward over iterations in one subplot with shaded std, for paper use."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1,
        "lines.markersize": 4,
    })

    iterations = np.arange(portion_matrix.shape[1])

    portion_mean = np.nanmean(portion_matrix, axis=0)
    portion_std = np.nanstd(portion_matrix, axis=0)

    reward_mean = np.nanmean(reward_matrix, axis=0)
    reward_std = np.nanstd(reward_matrix, axis=0)

    lower_portion_bound = np.clip(portion_mean - portion_std, 0, 1)
    upper_portion_bound = np.clip(portion_mean + portion_std, 0, 1)

    lower_reward_bound = np.clip(reward_mean - reward_std, 0, 1)
    upper_reward_bound = np.clip(reward_mean + reward_std, 0, 1)

    fig, ax = plt.subplots(figsize=(5.0, 2.5))

    # Portion (blue)
    ax.plot(iterations, portion_mean, label='Mean Portion', linestyle='--', color='tab:blue')
    ax.fill_between(iterations, lower_portion_bound, upper_portion_bound, alpha=0.2, color='tab:blue')

    # Reward (green)
    ax.plot(iterations, reward_mean, label='Mean Reward', linestyle='--', color='tab:green')
    ax.fill_between(iterations, lower_reward_bound, upper_reward_bound, alpha=0.2, color='tab:green')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Mean Portion and Reward Over Iterations", pad=4)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.45), frameon=False)
    ax.grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout(pad=1.0)
    if path_pdf:
        plt.savefig(path_pdf, format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load state dict
    ratio_actor_path = './LLM-RL/scripts/scratch/ratio_actor-2.pt'
    state_dict = torch.load(ratio_actor_path, weights_only=False)

    attempted_ratios_list = state_dict['attempted_ratios_list']
    portion_matrix, reward_matrix = extract_trajectory_stats(attempted_ratios_list)

    # Save plot in same directory as script
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories_summary.png")
    plot_portion_and_reward_trajectories(portion_matrix, reward_matrix, path=save_path, sample_ids=[0, 1, 2])


    # Also create a plot with only the aggregate trends
    avg_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mean_trajectories_only.png")
    plot_mean_trajectories_only(portion_matrix, reward_matrix, path=avg_plot_path)

    # Also create a plot with only the aggregate trends
    avg_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mean_trajectories_only_paper_style.pdf")
    plot_combined_mean_trajectory_paper_style(portion_matrix, reward_matrix, path_pdf=avg_plot_path)




