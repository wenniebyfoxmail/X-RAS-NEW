import numpy as np
import matplotlib.pyplot as plt

def plot_results(npz_file):
    # 1. 加载数据
    data = np.load(npz_file)
    d = data['d']
    load_steps = data['load_steps']
    reactions = data['reactions']
    coords = data['coords_d']

    print("Analyzing max damage at each step:")
    max_d_per_step = np.max(d, axis=0)
    for i, max_val in enumerate(max_d_per_step):
        print(f"Step {i+1}: Max Damage = {max_val:.6f}")

    # 2. 绘制反力-位移曲线
    plt.figure(figsize=(8, 6))
    plt.plot(load_steps, reactions, 'b-o', markersize=4, linewidth=2)
    plt.xlabel('Displacement (mm)', fontsize=12)
    plt.ylabel('Reaction Force (N)', fontsize=12)
    plt.title('Reaction Force - Displacement Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reaction_force_curve.png', dpi=300)
    plt.close()
    print("Saved reaction_force_curve.png")

    # 3. 绘制损伤图 (Steps 5, 10, 15, 20)
    target_steps = [5, 10, 15, 20]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, step in enumerate(target_steps):
        idx = step - 1
        if idx < d.shape[1]:
            d_val = d[:, idx]
            # s=10 增大点的大小以便肉眼观察
            sc = axes[i].scatter(coords[:,0], coords[:,1], c=d_val, cmap='jet', s=10, vmin=0, vmax=1)
            axes[i].set_title(f'Step {step} (Max d={d_val.max():.2f})')
            axes[i].axis('equal')
            axes[i].axis('off')
            plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04, label='Damage')
        else:
            axes[i].text(0.5, 0.5, 'Step out of range', ha='center')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('damage_evolution.png', dpi=300)
    plt.close()
    print("Saved damage_evolution.png")

    # 4. 单独绘制最终裂纹路径
    plt.figure(figsize=(7, 7))
    d_final = d[:, -1]
    plt.scatter(coords[:,0], coords[:,1], c=d_final, cmap='jet', s=10, vmin=0, vmax=1)
    plt.colorbar(label='Damage')
    plt.axis('equal')
    plt.title(f'Final Crack Path (Step {d.shape[1]})')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.tight_layout()
    plt.savefig('final_crack_path.png', dpi=300)
    plt.close()
    print("Saved final_crack_path.png")

if __name__ == "__main__":
    plot_results('data/fe_sent_phasefield.npz')