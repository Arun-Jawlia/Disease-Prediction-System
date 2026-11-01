import matplotlib.pyplot as plt


def save_confusion_matrix(
    cm,
    dataset_name,
    model_name,
    output_dir,
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(cm)

    ax.set_title(
        f"{dataset_name.title()} - "
        f"{model_name.replace('_', ' ').title()}"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Low Risk",
        "High Risk",
    ])

    ax.set_yticklabels([
        "Low Risk",
        "High Risk",
    ])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{dataset_name}_{model_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()