import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# =====================================================================
# ВАРІАНТ № 9 (n = 56)
# =====================================================================
VAR_9_X = [2, 3, 4, 5, 6, 7, 8, 9]
VAR_9_N = [1, 2, 17, 16, 11, 4, 4, 1]


def input_custom_data():
    """Введення довільної дискретної вибірки користувачем"""
    print("\n--- ВВЕДЕННЯ НОВОЇ ВИБІРКИ ---")
    k = int(input("Введіть кількість варіант (значень x_i): "))
    x_vals, n_vals = [], []
    for i in range(k):
        x = float(input(f"  x_{i+1}: "))
        n_i = int(input(f"  n_{i+1} (частота): "))
        x_vals.append(x)
        n_vals.append(n_i)
    return x_vals, n_vals


def process_lab_data(x_vals, n_vals):
    x_vals = np.array(x_vals, dtype=float)
    n_vals = np.array(n_vals, dtype=int)
    n_total = int(np.sum(n_vals))

    # 1. Формування впорядкованого варіаційного ряду
    raw_sample = []
    for x, n in zip(x_vals, n_vals):
        raw_sample.extend([x] * n)
    raw_sample = np.array(sorted(raw_sample))

    # 2. Статистичний розподіл
    w_vals = n_vals / n_total                  # Відносні частоти w_i
    F_cum = np.cumsum(w_vals)                  # Накопичені відносні частоти F*(x)

    # 3. Характеристики вихідної вибірки
    mean_orig = np.mean(raw_sample)
    var_orig = np.var(raw_sample, ddof=0)
    std_orig = np.sqrt(var_orig)
    median_orig = np.median(raw_sample)
    mode_orig = x_vals[np.argmax(n_vals)]

    # Функція обчислення квантиля через лінійну інтерполяцію
    def calc_quantile(sample, m_quant, K_quant=100):
        n_len = len(sample)
        L_m = (n_len + 1) * (m_quant / K_quant)
        idx = L_m - 1
        floor_idx = int(np.floor(idx))
        frac_L = idx - floor_idx

        if floor_idx < 0:
            return sample[0]
        elif floor_idx >= n_len - 1:
            return sample[-1]
        else:
            x_floor = sample[floor_idx]
            x_next = sample[floor_idx + 1]
            return x_floor + frac_L * (x_next - x_floor)

    # 4. Перерахунок квантилів до m = 100
    m_total = 100
    y_samples = np.array([calc_quantile(raw_sample, k, m_total) for k in range(1, m_total + 1)])

    # Формування сітки 10x10 для Перерахованої вибірки (n = 100)
    y_matrix = y_samples.reshape(10, 10)
    
    formatted_matrix = []
    for row in y_matrix:
        formatted_row = []
        for val in row:
            if val.is_integer():
                formatted_row.append(f"{int(val)}")
            else:
                formatted_row.append(f"{val:.2f}".rstrip('0').rstrip('.').replace('.', ','))
        formatted_matrix.append(formatted_row)

    df_sample_100 = pd.DataFrame(formatted_matrix)

    # 5. Характеристики перерахованої вибірки (m = 100)
    mean_new = np.mean(y_samples)
    var_new = np.var(y_samples, ddof=0)
    std_new = np.sqrt(var_new)
    median_new = np.median(y_samples)

    y_round = np.round(y_samples)
    vals, counts = np.unique(y_round, return_counts=True)
    mode_new = vals[np.argmax(counts)]

    # Формула різниці у відсотках за модулем: |A_перерах - A_вих| / A_вих * 100%
    def calc_abs_pct_str(new_val, orig_val):
        if orig_val == 0:
            return "0"
        pct = abs(new_val - orig_val) / orig_val * 100
        if pct == 0:
            return "0"
        return f"{pct:.2f}"

    # Форматування чисел (дрібні з 4 знаками, цілі без дробової частини)
    def fmt_val(val):
        if float(val).is_integer():
            return str(int(val))
        return f"{val:.4f}"

    # Порівняльна таблиця (Таблиця 10 як на скріншоті)
    df_comparison = pd.DataFrame({
        'Характеристика': [
            'Вибіркове середнє x_bar',
            'Вибіркова дисперсія D',
            'Середнє квадратичне відхилення sigma',
            'Медіана Me',
            'Мода Mo'
        ],
        f'Вихідна n = {n_total}': [
            fmt_val(mean_orig),
            fmt_val(var_orig),
            fmt_val(std_orig),
            fmt_val(median_orig),
            fmt_val(mode_orig)
        ],
        f'Перерахована n = {m_total}': [
            fmt_val(mean_new),
            fmt_val(var_new),
            fmt_val(std_new),
            fmt_val(median_new),
            fmt_val(mode_new)
        ],
        'Різниця, %': [
            calc_abs_pct_str(mean_new, mean_orig),
            calc_abs_pct_str(var_new, var_orig),
            calc_abs_pct_str(std_new, std_orig),
            calc_abs_pct_str(median_new, median_orig),
            calc_abs_pct_str(mode_new, mode_orig)
        ]
    })

    # --- ВИВІД ТАБЛИЦЬ В КОНСОЛЬ ---
    print("\n" + "="*75)
    print("                Перерахована вибірка (n = 100)")
    print("="*75)
    print(tabulate(df_sample_100, tablefmt='grid', stralign='center', numalign='center', headers=[], showindex=False))

    print("\n" + "="*75)
    print("   10. Порівняльна таблиця")
    print("="*75)
    print(tabulate(df_comparison, headers='keys', tablefmt='grid', stralign='left', numalign='center', showindex=False))

    # Побудова графіків
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Полігон частот
    axes[0].plot(x_vals, n_vals, marker='o', color='#1f77b4', linewidth=1.8, markersize=6)
    axes[0].set_title('1. Полігон частот (x_i, n_i)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('x_i')
    axes[0].set_ylabel('Частоти (n_i)')
    axes[0].grid(True, linestyle='-', alpha=0.4)

    # 2. Полігон відносних частот
    axes[1].plot(x_vals, w_vals, marker='o', color='#1f77b4', linewidth=1.8, markersize=6)
    axes[1].set_title('2. Полігон відносних частот (x_i, w_i)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('x_i')
    axes[1].set_ylabel('Відносні частоти (w_i)')
    axes[1].grid(True, linestyle='-', alpha=0.4)

    # 3. Ступінчаста функція розподілу F*(x)
    x_min_p = min(x_vals) - 1
    x_max_p = max(x_vals) + 2

    axes[2].plot([x_min_p, x_vals[0]], [0, 0], color='#d62728', linewidth=2)
    for i in range(len(x_vals) - 1):
        axes[2].plot([x_vals[i], x_vals[i+1]], [F_cum[i], F_cum[i]], color='#d62728', linewidth=2)
        axes[2].scatter(x_vals[i+1], F_cum[i], color='#d62728', facecolors='none', s=25, zorder=3)
        axes[2].scatter(x_vals[i], F_cum[i], color='#d62728', s=25, zorder=3)

    axes[2].plot([x_vals[-1], x_max_p], [1, 1], color='#d62728', linewidth=2)
    axes[2].scatter(x_vals[-1], F_cum[-1], color='#d62728', s=25, zorder=3)

    axes[2].set_title('3. Емпірична функція розподілу F*(x)', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('F*(x)')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('lab_results.png', dpi=300)
    print("\n[ІНФО] Графіки успішно збережено у файл 'lab_results.png'.")
    plt.show()


def main():
    while True:
        print("\n" + "="*45)
        print("     ОБРОБКА СТАТИСТИЧНОЇ ВИБІРКИ     ")
        print("="*45)
        print("1. Запустити Варіант 9")
        print("2. Ввести нову вибірку вручну")
        print("3. Вийти з програми")
        print("="*45)

        choice = input("Оберіть пункт (1-3): ").strip()

        if choice == '1':
            print("\n[ІНФО] Обчислення для Варіанта 9...")
            process_lab_data(VAR_9_X, VAR_9_N)
        elif choice == '2':
            x_vals, n_vals = input_custom_data()
            process_lab_data(x_vals, n_vals)
        elif choice == '3':
            print("\nПрограму завершено!")
            break
        else:
            print("\n[ПОМИЛКА] Введіть число 1, 2 або 3.")


if __name__ == "__main__":
    main()