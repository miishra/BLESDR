import pandas as pd
import matplotlib.pyplot as plt

csv_path = "apple2.csv"  # change this
cols = ["cfo_equal_00_hz", "cfo_equal_11_hz", "cfo_jump_10_hz", "cfo_jump_01_hz"]

df = pd.read_csv(csv_path)

n = len(df)
mid = n // 2

first_half = df.iloc[:mid][cols].dropna(how="all")
second_half = df.iloc[mid:][cols].dropna(how="all")

def plot_half(data, title_prefix):
    # Boxplot
    plt.figure(figsize=(8, 5))
    data.boxplot(column=cols)
    plt.title(f"{title_prefix} - CFO Equalization / Jumps (Hz) - Boxplot")
    plt.ylabel("Frequency Offset [Hz]")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    # Violin plot
    values = [data[c].values for c in cols]
    plt.figure(figsize=(8, 5))
    plt.violinplot(values, showmeans=True, showmedians=True)
    plt.title(f"{title_prefix} - CFO Equalization / Jumps (Hz) - Violin Plot")
    plt.ylabel("Frequency Offset [Hz]")
    plt.xticks(range(1, len(cols) + 1), cols, rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

# First half
plot_half(first_half, "First Half")

# Second half
plot_half(second_half, "Second Half")