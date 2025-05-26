import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, lambdify, simplify

# --- Аналитическое решение ---
def solve_analytical(model, x0_val, k_val, M_val=None):
    t = symbols('t')
    x = Function('x')(t)
    k = symbols('k')
    x0 = symbols('x0')
    M = symbols('M')

    if model == 'Мальтус':
        ode = Eq(x.diff(t), k * x)
        ic = {x.subs(t, 0): x0}
        sol = dsolve(ode, x, ics=ic).rhs.subs({k: k_val, x0: x0_val})
    else:
        ode = Eq(x.diff(t), k * x * (1 - x / M))
        ic = {x.subs(t, 0): x0}
        sol = simplify(dsolve(ode, x, ics=ic)).rhs.subs({k: k_val, x0: x0_val, M: M_val})

    return lambdify(t, sol, 'numpy')

# --- Численные методы ---
def euler(f, x0, t):
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + f(t[i - 1], x[i - 1]) * (t[i] - t[i - 1])
    return x

def rk2(f, x0, t):
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = f(t[i - 1], x[i - 1])
        k2 = f(t[i - 1] + h / 2, x[i - 1] + h * k1 / 2)
        x[i] = x[i - 1] + h * k2
    return x

def rk4(f, x0, t):
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = f(t[i - 1], x[i - 1])
        k2 = f(t[i - 1] + h / 2, x[i - 1] + h * k1 / 2)
        k3 = f(t[i - 1] + h / 2, x[i - 1] + h * k2 / 2)
        k4 = f(t[i - 1] + h, x[i - 1] + h * k3)
        x[i] = x[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

# --- Построение графика ---
def run_simulation(params):
    model = params['model']
    method = params['method']
    x0 = float(params['x0'])
    k = float(params['k'])
    M = float(params['M']) if model == 'Ферхюльст' else None
    t_max = float(params['t_max'])
    dt = float(params['dt'])

    t = np.arange(0, t_max + dt, dt)

    # Определим правую часть ОДУ
    if model == 'Мальтус':
        f = lambda t, x: k * x
    else:
        f = lambda t, x: k * x * (1 - x / M)

    # Аналитическое решение
    x_analytic_func = solve_analytical(model, x0, k, M)
    x_analytic = x_analytic_func(t)

    # Численное решение
    if method == 'Эйлер':
        x_numeric = euler(f, x0, t)
    elif method == 'РК2':
        x_numeric = rk2(f, x0, t)
    else:
        x_numeric = rk4(f, x0, t)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(t, x_analytic, label='Аналитическое решение', linewidth=2)
    plt.plot(t, x_numeric, '--', label=f'Численный метод: {method}')
    plt.xlabel('Время')
    plt.ylabel('Население')
    plt.title(f'Модель {model}')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Интерфейс GUI ---
def launch_gui():
    root = tk.Tk()
    root.title("Модель роста населения")

    entries = {}

    def add_entry(label_text, row):
        label = tk.Label(root, text=label_text)
        label.grid(row=row, column=0, sticky='w')
        entry = tk.Entry(root)
        entry.grid(row=row, column=1)
        entries[label_text] = entry

    add_entry("Начальное население x0", 0)
    add_entry("Коэффициент роста k", 1)
    add_entry("Предельное население M", 2)
    add_entry("Макс. время t_max", 3)
    add_entry("Шаг dt", 4)

    model_var = tk.StringVar(value='Мальтус')
    method_var = tk.StringVar(value='Эйлер')

    tk.Label(root, text="Модель:").grid(row=5, column=0, sticky='w')
    ttk.Combobox(root, textvariable=model_var, values=['Мальтус', 'Ферхюльст']).grid(row=5, column=1)

    tk.Label(root, text="Метод:").grid(row=6, column=0, sticky='w')
    ttk.Combobox(root, textvariable=method_var, values=['Эйлер', 'РК2', 'РК4']).grid(row=6, column=1)

    def on_run():
        params = {
            'x0': entries["Начальное население x0"].get(),
            'k': entries["Коэффициент роста k"].get(),
            'M': entries["Предельное население M"].get(),
            't_max': entries["Макс. время t_max"].get(),
            'dt': entries["Шаг dt"].get(),
            'model': model_var.get(),
            'method': method_var.get(),
        }
        run_simulation(params)

    tk.Button(root, text="Рассчитать", command=on_run).grid(row=7, column=0, columnspan=2)

    root.mainloop()

if __name__ == '__main__':
    launch_gui()
