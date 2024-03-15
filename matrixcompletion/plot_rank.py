import matplotlib.pyplot as plt
import torch

plt.style.use('seaborn')
# plt.figure(figsize=(4, 2.7))
plt.figure(figsize=(3.2, 2.7))

losses_critical = []
for rank1 in [2, 5, 10, 15]:
    saved_data = torch.load(
        f'output_july23/depth=3-ntrain=2000_rank1={rank1}_rank2=5_epochCont=30000_seed=1_initScale=0.001_lr=0.2_opt=sgd/info.pth')
    print(saved_data['losses_critical'])
    T_critical_list = saved_data['args'].T_critical_list
    losses_critical.append(saved_data['losses_critical'])
    plt.plot(T_critical_list, saved_data['losses_critical'], marker='o', markersize=5, label=f'R= {rank1}')

plt.title('Effect of Rank (Task 1)')
plt.xlabel('Deficit Removal (Epoch)')
plt.ylabel('Recon. Error')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=10, frameon=False)

# plt.legend(fontsize=10, loc='best', frameon=False)
plt.tight_layout()
plt.savefig('plots/critical-period-completion-rank_v2.pdf', bbox_inches='tight')
