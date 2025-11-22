#!/usr/bin/env python3
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV_PATH = r"\\wsl.localhost\Ubuntu-22.04\home\yying2\SpaceTeamsROS\depth_bands.csv"

# ===== 精度控制区域 =====
# 拟合收敛精度（数值越小 → 要求越严格 → 更慢，可能不收敛）
FTOL = 1e-12   # 目标函数相对变化阈值
XTOL = 1e-12   # 参数相对变化阈值
MAXFEV = 100000  # 最大函数评估次数（迭代上限）

# 输出小数位控制（比如 4 表示保留4位小数）
PRINT_DIGITS = 4
# =======================

def load_last_row_depth_bands(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    last = rows[-1]

    h_vals = []
    d_vals = []

    for col_name, value in zip(header, last):
        if not col_name.startswith("h_"):
            continue

        try:
            h = float(col_name.split("_")[1])
            d = float(value)
        except:
            continue

        if np.isnan(d):
            continue

        h_vals.append(h)
        d_vals.append(d)

    return np.array(h_vals, dtype=float), np.array(d_vals, dtype=float)

# 指数模型：d(h) = a * exp(b*h) + c
def exp_model(h, a, b, c):
    return a * np.exp(b * h) + c

def main():
    # 读取数据
    h_vals, d_vals = load_last_row_depth_bands(CSV_PATH)

    # 初值
    init_guess = [d_vals.max(), -5.0, d_vals.min()]

    # 指数拟合 + 精度控制
    popt, pcov = curve_fit(
        exp_model,
        h_vals,
        d_vals,
        p0=init_guess,
        ftol=FTOL,
        xtol=XTOL,
        maxfev=MAXFEV,
    )

    a, b, c = popt

    d = PRINT_DIGITS
    print("\n==== Exponential fit result ====")
    print(f"a = {a:.{d}f}")
    print(f"b = {b:.{d}f}")
    print(f"c = {c:.{d}f}")
    print("\nModel:")
    print(f"d(h) = {a:.{d}f} * exp({b:.{d}f} * h) + {c:.{d}f}")

    # 画图
    h_fit = np.linspace(h_vals.min(), h_vals.max(), 300)
    d_fit = exp_model(h_fit, *popt)

    plt.figure()
    plt.scatter(h_vals, d_vals, label="原始数据", s=25)
    plt.plot(h_fit, d_fit, "r-", label="指数拟合", linewidth=2)
    plt.xlabel("h (垂直归一化高度)")
    plt.ylabel("min depth (m)")
    plt.title("指数拟合：d(h) = a * exp(b*h) + c")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
