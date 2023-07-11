import matplotlib.pyplot as plt

# plt.xscale("log")

plt.plot([0.001, 0.1, 1, 10], [5, 10, 22, 33], label=f"minDCF(pi_tilde)")
plt.plot([0.001, 0.1, 1, 10], [50, 100, 252, 353], label=f"minDCF(pi_tilde)")

plt.xlabel('Lambda')
plt.ylabel('minDCF')
plt.title("Plot of Lambda with regard to minDCF value")
plt.legend()
plt.xscale("log")
plt.show()