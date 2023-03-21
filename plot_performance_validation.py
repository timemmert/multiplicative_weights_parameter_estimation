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

plt.title("Convergence time vs noise")
plt.xlabel("noise")
plt.ylabel("Convergence time")
plt.legend()
plt.show()

with open("data_use-best-cov=False_multi_candidates.json", "r") as fp:
    results = json.load(fp)
    candidates_plot = results["noises"]
    times_convergence_plot = results["times_convergence"]
    times_stddevs_plot = results["times_stddevs"]
plt.errorbar(candidates_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o', label="Rand draw matrix")

with open("data_use-best-cov=True_multi_candidates.json", "r") as fp:
    results = json.load(fp)
    candidates_plot = results["noises"]
    times_convergence_plot = results["times_convergence"]
    times_stddevs_plot = results["times_stddevs"]
plt.errorbar(candidates_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o', label="Use best matrix")



plt.title("Convergence time vs n_l candidates")
plt.xlabel("n_l candidates")
plt.ylabel("Convergence time")
plt.legend()
plt.show()

with open("data_use-best-cov=True_epsilon.json", "r") as fp:
    results = json.load(fp)
    epsilons_plot = results["epsilons"]
    times_convergence_plot = results["times_convergence"]
    times_stddevs_plot = results["times_stddevs"]
    correctness_percentage = results["correctness_percentage"]

plt.errorbar(epsilons_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o', label="Use best matrix")

for i, txt in enumerate(correctness_percentage):
    plt.annotate(txt, (epsilons_plot[i], times_convergence_plot[i] + 3))


with open("data_use-best-cov=False_epsilon.json", "r") as fp:
    results = json.load(fp)
    epsilons_plot = results["epsilons"]
    times_convergence_plot = results["times_convergence"]
    times_stddevs_plot = results["times_stddevs"]
    correctness_percentage = results["correctness_percentage"]
plt.errorbar(epsilons_plot, times_convergence_plot, times_stddevs_plot, linestyle='None', marker='o', label="Rand draw matrix")

for i, txt in enumerate(correctness_percentage):
    plt.annotate(txt, (epsilons_plot[i], times_convergence_plot[i]))

plt.title("Convergence time vs epsilon candidates")
plt.xlabel("Epsilon")
plt.ylabel("Convergence time")
plt.legend()
plt.show()
