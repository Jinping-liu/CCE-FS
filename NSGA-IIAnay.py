import matplotlib.pyplot as plt
import numpy as np

# Given data
data = [
    {'validity': 1.0, 'sparsity_cont': 0.390625, 'sparsity_cat': np.nan, 'lof_scores': 0.925,
     'cont_proximity': 21.180330951521476, 'cont_proximity_mad': -1.2914281479533134, "time": 8},
    {'validity': 1.0, 'sparsity_cont': 0.396875, 'sparsity_cat': np.nan, 'lof_scores': 0.9625,
     'cont_proximity': 13.327485408132873, 'cont_proximity_mad': -1.2102913563959523, "time": 31},
    {'validity': 1.0, 'sparsity_cont': 0.403125, 'sparsity_cat': np.nan, 'lof_scores': 0.975,
     'cont_proximity': 10.26622043384126, 'cont_proximity_mad': -1.1707830926046383, "time": 61},
    {'validity': 1.0, 'sparsity_cont': 0.4046875, 'sparsity_cat': np.nan, 'lof_scores': 0.98125,
     'cont_proximity': 8.527909856084236, 'cont_proximity_mad': -1.1374483423579176, "time": 123},
    {'validity': 1.0, 'sparsity_cont': 0.40625, 'sparsity_cat': np.nan, 'lof_scores': 0.985,
     'cont_proximity': 7.416743643179205, 'cont_proximity_mad': -1.115017636193744, "time": 187}
]

# Population sizes
population_sizes = [10, 50, 100, 200, 300]

# Extract data for plotting
validity = [entry['validity'] for entry in data]
sparsity_cont = [entry['sparsity_cont'] for entry in data]
lof_scores = [entry['lof_scores'] for entry in data]
cont_proximity_mad = [entry['cont_proximity_mad'] for entry in data]
processing_time = [entry['time'] for entry in data]

fig, ax1 = plt.subplots(figsize=(12, 8))

# Define colors and markers for each metric
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', 'D', '^']

# Create a uniform x-tick range
uniform_xticks = np.arange(len(population_sizes))

# Plotting each metric with distinct color and marker on the left y-axis
ax1.plot(uniform_xticks, validity, label='Validity', marker=markers[0], color=colors[0], linestyle='-')
ax1.plot(uniform_xticks, sparsity_cont, label='Continuous-Sparsity', marker=markers[1], color=colors[1], linestyle='-')
ax1.plot(uniform_xticks, lof_scores, label='LOF-NR', marker=markers[2], color=colors[2], linestyle='-')
ax1.plot(uniform_xticks, cont_proximity_mad, label='Continuous-Proximity', marker=markers[3], color=colors[3], linestyle='-')

ax1.set_xlabel('Population Size')
ax1.set_ylabel('Metric Value')
ax1.set_title('Performance Analysis vs Population Size', fontsize=16)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Set x-axis ticks to uniform_xticks and relabel them with population_sizes
ax1.set_xticks(uniform_xticks)
ax1.set_xticklabels(population_sizes)

# Create a second y-axis to plot the processing time
ax2 = ax1.twinx()
ax2.plot(uniform_xticks, processing_time, label='Processing Time', marker='x', color='purple', linestyle='--')
ax2.set_ylabel('Processing Time (seconds)')
ax2.set_yticks([0, 50, 100, 150, 200])

# Adjust right y-axis limit to better reflect the non-linear nature of processing time
ax2.set_ylim(0, 200)

# Annotate processing time data points
for i, txt in enumerate(processing_time):
    ax2.annotate(txt, (uniform_xticks[i], processing_time[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='purple')

ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig("./PerformanceAnalysis.pdf", dpi=600)
plt.show()
