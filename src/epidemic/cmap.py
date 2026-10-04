from matplotlib.colors import ListedColormap

def create_cmap_and_labels(q):
    
    max_infected = q // 2
    num_recovered = q - 1 - max_infected
    colors = ["#1f77b4"]
    labels = ["S (0)"]

    for i in range(1, max_infected + 1):
        ratio = (i - 1) / max(1, max_infected - 1) if max_infected > 1 else 0
        colors.append((1.0, 0.8 - 0.6 * ratio, 0.0))
        labels.append(f"I_{i} ({i})")

    for i in range(max_infected + 1, q):
        ratio = (i - max_infected - 1) / max(1, num_recovered - 1) if num_recovered > 1 else 0
        colors.append((0.2, 0.9 - 0.5 * ratio, 0.2))
        labels.append(f"R_{i - max_infected} ({i})")

    return ListedColormap(colors), labels