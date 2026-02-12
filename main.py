import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QLabel, QPushButton, QDoubleSpinBox,
    QHeaderView, QGroupBox, QLineEdit, QMessageBox, QStatusBar, QCheckBox,
    QDialog, QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid

class FileNameDialog(QDialog):
    """Диалог для ввода имени файла без расширения."""
    def __init__(self, parent=None, default_name="schrodinger_plot"):
        super().__init__(parent)
        self.setWindowTitle("Save Plot as PNG")
        self.setModal(True)
        
        layout = QFormLayout()
        
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("Enter filename without extension")
        layout.addRow("Filename:", self.name_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
    
    def get_filename(self):
        return self.name_edit.text().strip()

class SchrodingerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("1D Schrödinger Equation Solver")
        self.setGeometry(100, 100, 1300, 850)
        self.statusBar().showMessage("Ready")
        self.x = None                # сетка координат
        self.V = None                # потенциал на сетке
        self.energies = None         # собственные значения
        self.wavefuncs = None        # собственные функции
        self.requested_indices = []  # реальные номера уровней
        self.y_limits_manual = False # флаг ручного масштаба
        self.y_min = None            # сохранённые лимиты
        self.y_max = None
        self.setup_ui()
    
    def setup_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # === Панель ввода параметров ===
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        
        # Таблица для задания потенциала с панелью управления
        table_layout = QVBoxLayout()
        table_layout.addWidget(QLabel("Potential definition (piecewise):"))
        
        self.potential_table = QTableWidget(0, 3)
        self.potential_table.setHorizontalHeaderLabels(
            ["Start (x)", "End (x)", "V(x) expression"]
        )
        self.potential_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Панель управления строками
        row_control_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("+ Add interval")
        self.remove_row_btn = QPushButton("- Remove selected")
        row_control_layout.addWidget(self.add_row_btn)
        row_control_layout.addWidget(self.remove_row_btn)
        row_control_layout.addStretch()
        
        table_layout.addWidget(self.potential_table)
        table_layout.addLayout(row_control_layout)
        
        # Шаг дискретизации
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Spatial step (dx):"))
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.001, 1.0)
        self.dx_spin.setSingleStep(0.01)
        self.dx_spin.setValue(0.05)
        step_layout.addWidget(self.dx_spin)
        step_layout.addStretch()
        
        # Выбор уровней энергии
        levels_layout = QHBoxLayout()
        levels_layout.addWidget(QLabel("Energy levels to compute (e.g., 0,1,2 or 0-3):"))
        self.levels_edit = QLineEdit("0,1,2,3,4")
        levels_layout.addWidget(self.levels_edit)
        levels_layout.addStretch()
        
        # Управление масштабом по Y и сохранение
        control_layout = QHBoxLayout()
        
        # Масштаб по Y
        control_layout.addWidget(QLabel("Y-axis limits:"))
        self.y_auto_checkbox = QCheckBox("Auto-scale")
        self.y_auto_checkbox.setChecked(True)
        self.y_auto_checkbox.stateChanged.connect(self.toggle_y_manual)
        
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e6, 1e6)
        self.ymin_spin.setSingleStep(1.0)
        self.ymin_spin.setValue(-10.0)
        self.ymin_spin.setEnabled(False)
        
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-1e6, 1e6)
        self.ymax_spin.setSingleStep(1.0)
        self.ymax_spin.setValue(10.0)
        self.ymax_spin.setEnabled(False)
        
        self.apply_y_btn = QPushButton("Apply Y-limits")
        self.apply_y_btn.setEnabled(False)
        self.apply_y_btn.clicked.connect(self.apply_y_limits)
        
        control_layout.addWidget(self.y_auto_checkbox)
        control_layout.addWidget(QLabel("Y min:"))
        control_layout.addWidget(self.ymin_spin)
        control_layout.addWidget(QLabel("Y max:"))
        control_layout.addWidget(self.ymax_spin)
        control_layout.addWidget(self.apply_y_btn)
        
        # Кнопка сохранения
        self.save_btn = QPushButton("Save Plot as PNG")
        self.save_btn.clicked.connect(self.save_plot)
        self.save_btn.setEnabled(False)  # активируется после решения
        control_layout.addWidget(self.save_btn)
        control_layout.addStretch()
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        self.build_btn = QPushButton("Build Potential")
        self.solve_btn = QPushButton("Solve Schrödinger Equation")
        buttons_layout.addWidget(self.build_btn)
        buttons_layout.addWidget(self.solve_btn)
        
        input_layout.addLayout(table_layout)
        input_layout.addLayout(step_layout)
        input_layout.addLayout(levels_layout)
        input_layout.addLayout(control_layout)
        input_layout.addLayout(buttons_layout)
        input_group.setLayout(input_layout)
        
        # === Панель вывода графиков ===
        output_group = QGroupBox("Visualization")
        output_layout = QHBoxLayout()
        
        # График потенциала
        self.potential_fig = Figure(figsize=(6, 4))
        self.potential_canvas = FigureCanvas(self.potential_fig)
        self.potential_ax = self.potential_fig.add_subplot(111)
        self.potential_ax.set_title("Potential V(x)")
        self.potential_ax.set_xlabel("x")
        self.potential_ax.set_ylabel("V(x)")
        self.potential_ax.grid(True, alpha=0.3)
        
        # Результирующий график
        self.result_fig = Figure(figsize=(6, 4))
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_title("Wavefunctions ψₙ(x) + Energy Levels")
        self.result_ax.set_xlabel("x")
        self.result_ax.set_ylabel("ψₙ(x) + Eₙ")
        self.result_ax.grid(True, alpha=0.3)
        
        output_layout.addWidget(self.potential_canvas)
        output_layout.addWidget(self.result_canvas)
        output_group.setLayout(output_layout)
        
        # Сборка интерфейса
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Подключение сигналов
        self.add_row_btn.clicked.connect(self.add_table_row)
        self.remove_row_btn.clicked.connect(self.remove_selected_row)
        self.build_btn.clicked.connect(self.build_potential)
        self.solve_btn.clicked.connect(self.solve)
        
        # Инициализация таблицы примером (прямоугольная яма)
        self.add_table_row(-5.0, -1.0, "0.0")
        self.add_table_row(-1.0, 1.0, "-50.0")
        self.add_table_row(1.0, 5.0, "0.0")
    
    def toggle_y_manual(self, state):
        """Переключение между автоматическим и ручным масштабированием по Y."""
        is_auto = (state == 0)  # Qt.Unchecked = 0
        self.y_limits_manual = not is_auto
        self.ymin_spin.setEnabled(not is_auto)
        self.ymax_spin.setEnabled(not is_auto)
        self.apply_y_btn.setEnabled(not is_auto)
        
        if is_auto and self.wavefuncs is not None:
            self.draw_result(auto_scale=True)
            self.statusBar().showMessage("Y-axis: auto-scale enabled")
    
    def apply_y_limits(self):
        """Применяет заданные пользователем лимиты по оси Y без повторного решения."""
        if self.wavefuncs is None:
            QMessageBox.warning(self, "Warning", "No results to rescale. Solve first.")
            return
        
        try:
            ymin = self.ymin_spin.value()
            ymax = self.ymax_spin.value()
            if ymin >= ymax:
                raise ValueError("Y min must be less than Y max")
            
            self.y_min = ymin
            self.y_max = ymax
            self.draw_result(auto_scale=False)
            self.statusBar().showMessage(f"Y-axis scaled to [{ymin:.2f}, {ymax:.2f}]")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid Y limits:\n{e}")
    
    def save_plot(self):
        """Сохраняет результирующий график в PNG после запроса имени файла."""
        if self.wavefuncs is None:
            QMessageBox.warning(self, "Warning", "No results to save. Solve first.")
            return
        
        # Диалог для ввода имени файла
        dialog = FileNameDialog(self, default_name=f"schrodinger_n{self.requested_indices[0]}")
        if dialog.exec_() != QDialog.Accepted:
            return  # пользователь отменил
        
        filename_base = dialog.get_filename()
        if not filename_base:
            QMessageBox.warning(self, "Warning", "Filename cannot be empty.")
            return
        
        # Валидация имени файла
        invalid_chars = r'<>:"/\|?*'
        if any(c in filename_base for c in invalid_chars):
            QMessageBox.critical(
                self, "Error",
                f"Filename contains invalid characters: {invalid_chars}\n"
                f"Please use only letters, numbers, underscores and hyphens."
            )
            return
        
        # Формирование полного пути
        filename = f"{filename_base}.png"
        filepath = os.path.join(os.getcwd(), filename)
        
        # Сохранение графика
        try:
            # Увеличиваем разрешение для публикаций (300 DPI)
            self.result_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.statusBar().showMessage(f"Plot saved to: {filepath}")
            QMessageBox.information(
                self, "Success",
                f"Plot successfully saved as:\n{filepath}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save plot:\n{type(e).__name__}: {e}"
            )
            self.statusBar().showMessage(f"Save failed: {e}")
    
    def add_table_row(self, start=None, end=None, expr=None):
        row = self.potential_table.rowCount()
        self.potential_table.insertRow(row)
        if start is not None:
            self.potential_table.setItem(row, 0, QTableWidgetItem(str(start)))
        if end is not None:
            self.potential_table.setItem(row, 1, QTableWidgetItem(str(end)))
        if expr is not None:
            self.potential_table.setItem(row, 2, QTableWidgetItem(str(expr)))
    
    def remove_selected_row(self):
        current = self.potential_table.currentRow()
        if current >= 0:
            self.potential_table.removeRow(current)
    
    def build_potential(self):
        """Строит потенциал из таблицы и отображает его на графике."""
        # Валидация и парсинг таблицы
        intervals = []
        try:
            for row in range(self.potential_table.rowCount()):
                start_item = self.potential_table.item(row, 0)
                end_item = self.potential_table.item(row, 1)
                expr_item = self.potential_table.item(row, 2)
                
                if not all([start_item, end_item, expr_item]):
                    raise ValueError(f"Row {row}: incomplete data")
                
                start = float(start_item.text())
                end = float(end_item.text())
                expr = expr_item.text().strip()
                if start >= end:
                    raise ValueError(f"Row {row}: start ({start}) >= end ({end})")
                intervals.append((start, end, expr))
            
            if not intervals:
                raise ValueError("No intervals defined")
        
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid table \n{e}")
            self.statusBar().showMessage(f"Error: {e}")
            return
        
        # Определение расчётной области СТРОГО по заданным интервалам
        domain_min = min(iv[0] for iv in intervals)
        domain_max = max(iv[1] for iv in intervals)
        
        # Создание сетки
        dx = self.dx_spin.value()
        self.x = np.arange(domain_min, domain_max + dx/2, dx)
        
        # Вычисление потенциала на сетке
        self.V = np.zeros_like(self.x)
        safe_env = {
            "__builtins__": {},
            "np": np,
            "x": 0.0,  
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": np.abs
        }
        
        for i, x_val in enumerate(self.x):
            assigned = False
            for start, end, expr in intervals:
                if start <= x_val <= end:
                    try:
                        safe_env['x'] = x_val
                        self.V[i] = eval(expr, {"__builtins__": {}}, safe_env)
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Evaluation Error",
                            f"Error at x={x_val:.3f} for expression '{expr}':\n{e}"
                        )
                        self.statusBar().showMessage(f"Evaluation error at x={x_val:.3f}")
                        return
                    assigned = True
                    break
            
            if not assigned:
                self.V[i] = 0.0
        
        # Отрисовка потенциала
        self.potential_ax.clear()
        self.potential_ax.plot(self.x, self.V, 'b-', linewidth=2, label='V(x)')
        self.potential_ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self.potential_ax.set_title("Potential V(x)")
        self.potential_ax.set_xlabel("x")
        self.potential_ax.set_ylabel("V(x)")
        self.potential_ax.grid(True, alpha=0.3)
        self.potential_ax.legend()
        self.potential_canvas.draw()
        
        self.statusBar().showMessage(f"Potential built on {len(self.x)} grid points (domain: [{domain_min:.2f}, {domain_max:.2f}])")
    
    def solve(self):
        """Решает уравнение Шредингера методом конечных разностей."""
        if self.x is None or self.V is None:
            QMessageBox.critical(self, "Error", "Build potential first (click 'Build Potential')!")
            self.statusBar().showMessage("Error: Build potential first")
            return
        
        # Парсинг запрошенных уровней
        try:
            level_str = self.levels_edit.text().strip()
            if not level_str:
                raise ValueError("Level list is empty")
            
            requested_indices = set()
            for part in level_str.split(','):
                part = part.strip()
                if '-' in part:
                    a, b = map(int, part.split('-'))
                    if a > b:
                        raise ValueError(f"Invalid range {a}-{b} (start > end)")
                    requested_indices.update(range(a, b + 1))
                else:
                    requested_indices.add(int(part))
            
            self.requested_indices = sorted(requested_indices)
            if not self.requested_indices:
                raise ValueError("No valid levels specified")
            
            max_available = len(self.x) - 3
            if max(self.requested_indices) > max_available:
                raise ValueError(
                    f"Requested level {max(self.requested_indices)} exceeds maximum available ({max_available})\n"
                    f"Decrease dx or request lower levels."
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid level specification:\n{e}")
            self.statusBar().showMessage(f"Level spec error: {e}")
            return
        
        # Построение трёхдиагональной матрицы (атомные единицы: ħ = m = 1)
        dx = self.dx_spin.value()
        dx2_inv = 1.0 / dx**2
        V_internal = self.V[1:-1]
        
        diag = dx2_inv + V_internal          # 1/h² + V_i
        offdiag = -0.5 * dx2_inv * np.ones(len(diag) - 1)  # -1/(2h²)
        
        # Решение задачи на собственные значения
        try:
            idx_min, idx_max = min(self.requested_indices), max(self.requested_indices)
            energies_full, eigvecs_full = eigh_tridiagonal(
                diag, offdiag,
                select='i',
                select_range=(idx_min, idx_max)
            )
            
            self.energies = np.array([
                energies_full[i - idx_min] for i in self.requested_indices
            ])
            self.wavefuncs = np.zeros((len(self.x), len(self.requested_indices)))
            
            # Восстановление граничных условий и нормировка
            for j, idx in enumerate(self.requested_indices):
                vec = eigvecs_full[:, idx - idx_min]
                self.wavefuncs[1:-1, j] = vec
                
                # Нормировка через trapezoid
                norm = np.sqrt(trapezoid(np.abs(self.wavefuncs[:, j])**2, self.x))
                if norm > 1e-12:
                    self.wavefuncs[:, j] /= norm
                else:
                    self.wavefuncs[:, j] = 0.0
                    QMessageBox.warning(
                        self, "Warning",
                        f"Wavefunction for level {idx} has near-zero norm (possibly unbound state)"
                    )
        
        except Exception as e:
            QMessageBox.critical(self, "Solver Error", f"Eigenvalue solver failed:\n{type(e).__name__}: {e}")
            self.statusBar().showMessage(f"Solver error: {type(e).__name__}")
            return
        
        # Визуализация с автоматическим масштабом
        self.y_auto_checkbox.setChecked(True)
        self.y_limits_manual = False
        self.draw_result(auto_scale=True)
        self.save_btn.setEnabled(True)  # активируем кнопку сохранения
        
        energy_summary = ", ".join([f"n={idx}: {E:.4f}" for idx, E in zip(self.requested_indices, self.energies)])
        self.statusBar().showMessage(f"Computed {len(self.energies)} levels: {energy_summary}")
    
    def draw_result(self, auto_scale=True):
        """Визуализирует волновые функции и потенциал с корректными номерами уровней."""
        if self.wavefuncs is None or self.x is None:
            return
        
        self.result_ax.clear()
        
        # Потенциал (полупрозрачный)
        self.result_ax.plot(self.x, self.V, 'k-', alpha=0.4, linewidth=2, label='V(x)')
        
        # Волновые функции со смещением на E_n
        colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(self.energies))))
        for plot_idx, (real_idx, E, psi) in enumerate(zip(self.requested_indices, self.energies, self.wavefuncs.T)):
            self.result_ax.plot(
                self.x, psi + E, 
                color=colors[plot_idx], 
                linewidth=1.5, 
                label=f'n={real_idx}, E={E:.3f}'
            )
            self.result_ax.axhline(E, color=colors[plot_idx], linestyle='--', alpha=0.5)
        
        self.result_ax.set_title("Wavefunctions ψₙ(x) + Energy Levels")
        self.result_ax.set_xlabel("x")
        self.result_ax.set_ylabel("ψₙ(x) + Eₙ")
        self.result_ax.grid(True, alpha=0.3)
        
        # Применение лимитов по Y
        if not auto_scale and self.y_limits_manual:
            self.result_ax.set_ylim(self.y_min, self.y_max)
        else:
            # Автоматический масштаб с небольшим отступом
            all_y = np.concatenate([self.V, *(self.wavefuncs[:, i] + self.energies[i] for i in range(len(self.energies)))])
            y_min_auto = np.min(all_y)
            y_max_auto = np.max(all_y)
            margin = 0.1 * (y_max_auto - y_min_auto)
            self.result_ax.set_ylim(y_min_auto - margin, y_max_auto + margin)
        
        if len(self.energies) <= 6:
            self.result_ax.legend(loc='upper right', fontsize='small')
        
        self.result_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SchrodingerApp()
    window.show()
    sys.exit(app.exec_())