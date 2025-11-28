"""
Visualización detallada de una capa oculta del MLP
Muestra el flujo: FC → BatchNorm → ReLU → Dropout
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def create_hidden_layer_diagram():
    """Crea un diagrama detallado de una capa oculta."""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colores
    color_input = "#D4E9D7"  # Verde claro
    color_fc = "#FFE5B4"  # Naranja claro
    color_bn = "#B4D7FF"  # Azul claro
    color_relu = "#FFD4E5"  # Rosa claro
    color_dropout = "#E8E8E8"  # Gris claro
    color_neuron = "#4A90E2"  # Azul para neuronas

    # Función para dibujar neurona
    def draw_neuron(x, y, radius=0.15, color=color_neuron, label=""):
        circle = Circle(
            (x, y), radius, color=color, ec="black", linewidth=1.5, zorder=3
        )
        ax.add_patch(circle)
        if label:
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
                color="white",
                zorder=4,
            )

    # Función para dibujar caja de operación
    def draw_operation_box(x, y, width, height, color, text, fontsize=11):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08",
            edgecolor="black",
            facecolor=color,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            zorder=3,
        )

    # Función para dibujar flecha
    def draw_arrow(x1, y1, x2, y2, style="->", color="black", linewidth=1.5):
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=linewidth,
            mutation_scale=15,
            zorder=1,
        )
        ax.add_patch(arrow)

    # Título
    ax.text(
        7,
        9.3,
        "Detalle de Capa Oculta del MLP",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        7,
        8.8,
        "Ejemplo: Capa con 128 neuronas",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        color="#555",
    )

    # ========== CAPA DE ENTRADA ==========
    input_x = 1.5
    input_neurons_y = [7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5]

    ax.text(
        input_x,
        8.2,
        "Entrada\n(69 dim)",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#333",
    )

    # Dibujar neuronas de entrada
    for i, y in enumerate(input_neurons_y[:3]):
        draw_neuron(input_x, y, radius=0.12, color="#7CB342")

    # Puntos suspensivos
    ax.text(
        input_x,
        3.5,
        "...",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color="#555",
    )

    # Últimas neuronas
    for i, y in enumerate(input_neurons_y[-2:]):
        draw_neuron(input_x, y, radius=0.12, color="#7CB342")

    # ========== FULLY CONNECTED (FC) ==========
    fc_x = 3.5
    fc_neurons_y = [7.5, 6.8, 6.1, 5.4, 4.7, 4.0, 3.3, 2.6, 1.9, 1.2]

    ax.text(
        fc_x,
        8.2,
        "Fully Connected\n(128 neuronas)",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#333",
    )

    # Dibujar algunas neuronas FC
    for i, y in enumerate(fc_neurons_y[:4]):
        draw_neuron(fc_x, y, radius=0.12, color="#FF8C00")
        # Conexiones desde entrada
        for input_y in [input_neurons_y[0], input_neurons_y[1], input_neurons_y[-1]]:
            draw_arrow(
                input_x + 0.12,
                input_y,
                fc_x - 0.12,
                y,
                style="-",
                color="#BBB",
                linewidth=0.5,
            )

    # Puntos suspensivos
    ax.text(
        fc_x,
        3.0,
        "...",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color="#555",
    )

    # Últimas neuronas FC
    for i, y in enumerate(fc_neurons_y[-3:]):
        draw_neuron(fc_x, y, radius=0.12, color="#FF8C00")
        for input_y in [input_neurons_y[0], input_neurons_y[-1]]:
            draw_arrow(
                input_x + 0.12,
                input_y,
                fc_x - 0.12,
                y,
                style="-",
                color="#BBB",
                linewidth=0.5,
            )

    # Fórmula FC
    ax.text(
        fc_x,
        0.6,
        r"$y = Wx + b$",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_fc, alpha=0.7),
    )

    # ========== BATCH NORMALIZATION ==========
    bn_x = 5.8

    draw_operation_box(bn_x, 4.5, 1.8, 7.0, color_bn, "Batch\nNorm", fontsize=12)

    # Conexiones FC → BN
    for y in fc_neurons_y[:4]:
        draw_arrow(fc_x + 0.12, y, bn_x - 0.9, y, color="#666", linewidth=1.2)
    for y in fc_neurons_y[-3:]:
        draw_arrow(fc_x + 0.12, y, bn_x - 0.9, y, color="#666", linewidth=1.2)

    # Fórmula BN
    ax.text(
        bn_x,
        0.6,
        r"$\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_bn, alpha=0.7),
    )

    # ========== RELU ==========
    relu_x = 8.1

    draw_operation_box(relu_x, 4.5, 1.8, 7.0, color_relu, "ReLU", fontsize=12)

    # Conexiones BN → ReLU
    for y in fc_neurons_y[:4]:
        draw_arrow(bn_x + 0.9, y, relu_x - 0.9, y, color="#666", linewidth=1.2)
    for y in fc_neurons_y[-3:]:
        draw_arrow(bn_x + 0.9, y, relu_x - 0.9, y, color="#666", linewidth=1.2)

    # Fórmula ReLU
    ax.text(
        relu_x,
        0.6,
        r"$max(0, x)$",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color_relu, alpha=0.7),
    )

    # ========== DROPOUT ==========
    dropout_x = 10.4
    dropout_neurons_y = fc_neurons_y.copy()

    ax.text(
        dropout_x,
        8.2,
        "Dropout (p=0.2)\n20% desactivadas",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#333",
    )

    # Dibujar neuronas después de dropout
    # Algunas activas (azul) y algunas desactivadas (gris)
    active_indices = [0, 1, 3, -3, -2, -1]  # Índices de neuronas activas

    for i, y in enumerate(dropout_neurons_y[:4]):
        if i in active_indices:
            draw_neuron(dropout_x, y, radius=0.12, color="#4CAF50")
            draw_arrow(
                relu_x + 0.9, y, dropout_x - 0.12, y, color="#666", linewidth=1.2
            )
        else:
            draw_neuron(dropout_x, y, radius=0.12, color="#CCC")
            draw_arrow(
                relu_x + 0.9,
                y,
                dropout_x - 0.12,
                y,
                color="#DDD",
                linewidth=0.8,
                style="-",
            )
            # Marcar con X
            ax.plot(
                [dropout_x - 0.08, dropout_x + 0.08],
                [y - 0.08, y + 0.08],
                "r-",
                linewidth=2,
                zorder=5,
            )
            ax.plot(
                [dropout_x - 0.08, dropout_x + 0.08],
                [y + 0.08, y - 0.08],
                "r-",
                linewidth=2,
                zorder=5,
            )

    ax.text(
        dropout_x,
        3.0,
        "...",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color="#555",
    )

    for i, y in enumerate(dropout_neurons_y[-3:]):
        idx = len(dropout_neurons_y) - 3 + i
        if idx in active_indices or i >= 1:  # Mantener últimas activas
            draw_neuron(dropout_x, y, radius=0.12, color="#4CAF50")
            draw_arrow(
                relu_x + 0.9, y, dropout_x - 0.12, y, color="#666", linewidth=1.2
            )
        else:
            draw_neuron(dropout_x, y, radius=0.12, color="#CCC")
            draw_arrow(
                relu_x + 0.9,
                y,
                dropout_x - 0.12,
                y,
                color="#DDD",
                linewidth=0.8,
                style="-",
            )
            ax.plot(
                [dropout_x - 0.08, dropout_x + 0.08],
                [y - 0.08, y + 0.08],
                "r-",
                linewidth=2,
                zorder=5,
            )
            ax.plot(
                [dropout_x - 0.08, dropout_x + 0.08],
                [y + 0.08, y - 0.08],
                "r-",
                linewidth=2,
                zorder=5,
            )

    # ========== SALIDA DE LA CAPA ==========
    output_x = 12.5

    ax.text(
        output_x,
        8.2,
        "Salida\n(128 dim)",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#333",
    )

    for i, y in enumerate(dropout_neurons_y[:4]):
        if i in active_indices:
            draw_neuron(output_x, y, radius=0.12, color="#9C27B0")
            draw_arrow(
                dropout_x + 0.12, y, output_x - 0.12, y, color="#666", linewidth=1.2
            )

    ax.text(
        output_x,
        3.0,
        "...",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color="#555",
    )

    for i, y in enumerate(dropout_neurons_y[-3:]):
        idx = len(dropout_neurons_y) - 3 + i
        if idx in active_indices or i >= 1:
            draw_neuron(output_x, y, radius=0.12, color="#9C27B0")
            draw_arrow(
                dropout_x + 0.12, y, output_x - 0.12, y, color="#666", linewidth=1.2
            )

    # Leyenda
    legend_y = 0.3
    legend_elements = [
        ("Neurona activa", "#4CAF50"),
        ("Neurona desactivada (Dropout)", "#CCC"),
    ]

    legend_x_start = 4.5
    for i, (label, color) in enumerate(legend_elements):
        x_pos = legend_x_start + i * 3.5
        draw_neuron(x_pos - 0.5, legend_y, radius=0.1, color=color)
        ax.text(x_pos + 0.3, legend_y, label, ha="left", va="center", fontsize=9)

    plt.tight_layout()

    # Guardar
    output_path = "output/hidden_layer_detail.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Diagrama detallado guardado en: {output_path}")

    plt.show()


if __name__ == "__main__":
    create_hidden_layer_diagram()
