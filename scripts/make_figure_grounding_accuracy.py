import matplotlib.pyplot as plt
import numpy as np

# Condições
conditions = ["Baseline", "L1", "L2", "L3", "L2-Ref"]
x = np.arange(len(conditions))

# ===== Dados (substitua pelos seus valores finais) =====
gpt41 = [31.1, 50.0, 43.3, 48.9, 44.4]
qwen_235b = [27.0, 43.5, 49.0, 47.0, 48.0]
qwen_32b = [32.0, 41.0, 38.0, 45.0, 37.0]
qwen_8b = [30.0, 34.0, 29.0, 29.0, 27.0]

# ===== Estilo IEEE =====
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9
})

plt.figure(figsize=(3.5, 2.5))  # largura típica IEEE (coluna única)

# Linhas com marcadores distintos (funciona em preto e branco)
plt.plot(x, gpt41, marker='o', linestyle='-', linewidth=1.5, label='GPT-4.1')
plt.plot(x, qwen_235b, marker='D', linestyle='--', linewidth=1.5, label='Qwen3-VL-235B')
plt.plot(x, qwen_32b, marker='s', linestyle='-.', linewidth=1.5, label='Qwen3-VL-32B')
plt.plot(x, qwen_8b, marker='^', linestyle=':', linewidth=1.5, label='Qwen3-VL-8B')

# Eixos
plt.xticks(x, conditions)
plt.ylabel("Accuracy (%)")
plt.xlabel("Condition")

# Grid leve (opcional, IEEE aceita discreto)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

# Legenda compacta
plt.legend(fontsize=7, loc='best', frameon=False)

plt.tight_layout()

# Salvar em alta qualidade
plt.savefig("ieee_grounding_plot.pdf", bbox_inches='tight')
plt.savefig("ieee_grounding_plot.png", dpi=300, bbox_inches='tight')

plt.show()