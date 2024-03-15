import matplotlib.pyplot as plt
import numpy as np
import torch

plt.style.use('seaborn')
use_rel = True  # False # Plot either relative or absolute reconstruction error
# plt.figure(figsize=(4, 2.7))
plt.figure(figsize=(3.2, 2.7))

losses_critical = []
for depth in [1, 2, 3]:
    saved_data = torch.load(
        f'output_july23/depth={depth}-ntrain=2000_rank1=10_rank2=5_epochCont=30000_seed=1_initScale=0.001_lr=0.2_opt=sgd/info.pth')
    print(saved_data['losses_critical'])
    T_critical_list = saved_data['args'].T_critical_list
    losses_critical.append(saved_data['losses_critical'])
    if use_rel:
        plt.plot(T_critical_list, saved_data['losses_critical'][0] - np.array(saved_data['losses_critical']),
                 marker='o',
                 markersize=5, label=f'D= {depth}')
    else:
        plt.plot(T_critical_list, np.array(saved_data['losses_critical']), marker='o',
                 markersize=5, label=f'D= {depth}')

plt.title(r'Effect of Depth (Rank 10 $\to$ Rank 5)')
plt.xlabel('Deficit Removal (Epoch)')
if use_rel:
    plt.ylabel('Rel. Recon. Error')
else:
    plt.ylabel('Recon. Error')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# plt.legend(fontsize=10, loc='best', frameon=False)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.55), fontsize=10, frameon=False)

plt.tight_layout()
if use_rel:
    plt.savefig('plots/critical-period-completion-depth-rel_v2.pdf', bbox_inches='tight')
else:
    plt.savefig('plots/critical-period-completion-depth.pdf', bbox_inches='tight')
