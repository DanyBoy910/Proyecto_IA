"""
Visualización de la arquitectura del MLP para predicción de calidad del aire
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def create_architecture_diagram():
    """Crea un diagrama de la arquitectura del MLP."""

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Colores
    color_input = "#D4E9D7"  # Verde claro
    color_hidden = "#F9E5A3"  # Amarillo claro
    color_output = "#F4C2C2"  # Rosa claro
    color_dropout = "#E8E8E8"  # Gris claro

    # Parámetros de posición
    x_center = 5
    box_width = 4
    box_height = 0.8
    spacing = 0.3
    dropout_height = 0.4

    # Posiciones Y (de arriba hacia abajo)
    y_positions = {
        "input": 12.5,
        "fc1": 10.8,
        "dropout1": 9.7,
        "fc2": 8.6,
        "dropout2": 7.5,
        "fc3": 6.4,
        "dropout3": 5.3,
        "output": 3.6,
    }

    # Función para crear caja
    def create_box(x, y, width, height, color, text, fontsize=14, bold=True):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)

        weight = "bold" if bold else "normal"
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight=weight,
            family="sans-serif",
        )

    # Función para crear flecha
    def create_arrow(y_start, y_end, label=""):
        arrow = FancyArrowPatch(
            (x_center, y_start),
            (x_center, y_end),
            arrowstyle="->,head_width=0.4,head_length=0.3",
            color="black",
            linewidth=2,
            mutation_scale=20,
        )
        ax.add_patch(arrow)

        if label:
            y_mid = (y_start + y_end) / 2
            ax.text(
                x_center + 2.2,
                y_mid,
                label,
                ha="left",
                va="center",
                fontsize=11,
                style="italic",
                color="#333333",
            )

    # Función para crear caja de dropout
    def create_dropout_box(y, label):
        # Líneas punteadas del contenedor
        dropout_box = FancyBboxPatch(
            (x_center - box_width / 2 - 0.3, y - 0.9),
            box_width + 0.6,
            1.8,
            boxstyle="round,pad=0.05",
            edgecolor="#9370DB",
            facecolor="none",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(dropout_box)

        # Etiqueta de dropout
        ax.text(
            x_center + box_width / 2 + 1.8,
            y,
            label,
            ha="left",
            va="center",
            fontsize=11,
            style="italic",
            color="#333333",
        )

    # INPUT LAYER
    create_box(
        x_center,
        y_positions["input"],
        box_width,
        box_height,
        color_input,
        "Input ×69",
        fontsize=16,
    )

    # PRIMERA CAPA OCULTA (128)
    create_dropout_box(y_positions["fc1"], "Dropout= 0.2")
    create_arrow(
        y_positions["input"] - box_height / 2, y_positions["fc1"] + box_height / 2
    )
    create_box(
        x_center,
        y_positions["fc1"],
        box_width,
        box_height,
        color_hidden,
        "FC ×128",
        fontsize=15,
    )

    # SEGUNDA CAPA OCULTA (64)
    create_dropout_box(y_positions["fc2"], "Dropout= 0.2")
    create_arrow(
        y_positions["fc1"] - box_height / 2, y_positions["fc2"] + box_height / 2
    )
    create_box(
        x_center,
        y_positions["fc2"],
        box_width,
        box_height,
        color_hidden,
        "FC ×64",
        fontsize=15,
    )

    # TERCERA CAPA OCULTA (32)
    create_dropout_box(y_positions["fc3"], "Dropout= 0.2")
    create_arrow(
        y_positions["fc2"] - box_height / 2, y_positions["fc3"] + box_height / 2
    )
    create_box(
        x_center,
        y_positions["fc3"],
        box_width,
        box_height,
        color_hidden,
        "FC ×32",
        fontsize=15,
    )

    # OUTPUT LAYER
    create_arrow(
        y_positions["fc3"] - box_height / 2, y_positions["output"] + box_height / 2
    )
    create_box(
        x_center,
        y_positions["output"],
        box_width,
        box_height,
        color_output,
        "Output ×3",
        fontsize=16,
    )

    # Título
    ax.text(
        x_center,
        13.5,
        "Arquitectura del MLP\nPredicción de Calidad del Aire",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        family="sans-serif",
    )

    # Leyenda de componentes
    legend_y = 2.0
    ax.text(
        x_center,
        legend_y,
        "Componentes de cada capa oculta:",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        style="italic",
    )
    ax.text(
        x_center,
        legend_y - 0.4,
        "FC → BatchNorm1d → ReLU → Dropout",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", alpha=0.7),
    )

    # Información adicional
    info_y = 0.8
    info_text = (
        "Input: 69 características (11 vars + 55 lags + 3 temporales)\n"
        + "Output: 3 contaminantes (PM2.5, O3, CO)"
    )
    ax.text(
        x_center,
        info_y,
        info_text,
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFACD", alpha=0.5),
    )

    plt.tight_layout()

    # Guardar
    output_path = "output/mlp_architecture_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Diagrama guardado en: {output_path}")

    plt.show()


if __name__ == "__main__":
    create_architecture_diagram()
