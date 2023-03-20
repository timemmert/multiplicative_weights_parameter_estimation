import json

from matplotlib import pyplot as plt

with open("data_use-best-cov=False.json", "r") as fp:
    results = json.load(fp)
    noises_plot = results["noises"]
    times_convergence_plot = results["times_convergence"]
    times_stddevs_plot = results["times_stddevs"]

plt.errorbar(noises_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o', label="Rand draw matrix")

with open("data_use-best-cov=True.json", "r") as fp:
    results = json.load(fp)
    noises_plot = results["noises"]
    times_convergence_plot = results["times_convergence"]
    times_stddevs_plot = results["times_stddevs"]

plt.errorbar(noises_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o', label="Use best matrix")
plt.legend()
plt.show()
