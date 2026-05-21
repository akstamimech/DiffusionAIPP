
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

maps = list(range(31, 50))

diffusion = {
    31: 16.8798,
    32: 24.2293,
    33: 15.1882,
    34: 13.6741,
    35: 5.8435,
    36: 36.0959,
    37: 38.2846,
    38: 43.7513,
    39: 13.8261,
    40: 27.7982,
    41: 51.9428,
    42: 18.2256,
    43: 11.1662,
    44: 32.8830,
    45: 25.2199,
    46: 5.7386,
    47: 20.4951,
    48: 51.0421,
    49: 60.2259,
}

classic = {
    31: 59.3564,
    32: 28.9149,
    33: 6.3172,
    34: 19.4249,
    35: 0.7776,
    36: 40.9506,
    37: 54.6519,
    38: 18.8863,
    39: 16.5652,
    40: 14.1806,
    41: 55.4862,
    42: 10.5805,
    43: 14.2272,
    44: 22.6193,
    45: 14.2480,
    46: 4.5603,
    47: 30.6692,
    48: 55.0902,
    49: 60.8915,
}

diff_vals = np.array([diffusion[m] for m in maps])
classic_vals = np.array([classic[m] for m in maps])
diff_avg = diff_vals.mean()
classic_avg = classic_vals.mean()
delta = diff_vals - classic_vals

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11.56, 7.68), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

ax1.plot(maps, diff_vals, marker='o', linewidth=2, markersize=6, label='Diffusion')
ax1.plot(maps, classic_vals, marker='s', linewidth=2, markersize=6, label='Classic')
ax1.axhline(diff_avg, linestyle='--', linewidth=1.6, alpha=0.55, label=f'Diffusion avg = {diff_avg:.2f}%')
ax1.axhline(classic_avg, linestyle=':', linewidth=1.6, alpha=0.8, label=f'Classic avg = {classic_avg:.2f}%')
ax1.set_title('Task Completion by Map: Diffusion vs Classic')
ax1.set_ylabel('Task Completion (%)')
ax1.grid(True, alpha=0.25)
ax1.legend(loc='upper left')

colors = ['#2ca02c' if d >= 0 else '#d62728' for d in delta]
ax2.bar(maps, delta, color=colors)
ax2.axhline(0, color='black', linewidth=1)
ax2.set_ylabel('Diff - Classic')
ax2.set_xlabel('Map ID')
ax2.set_xticks(maps)
ax2.set_xticklabels([str(m) for m in maps])
ax2.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
fig.savefig('diffusion_vs_classic_task_completion_matched.png', dpi=200)

plt.show()
