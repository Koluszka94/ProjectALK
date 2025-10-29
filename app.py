# ===============================
# Patient Explorer
# Główne okno aplikacji PyQt6
# Obsługa danych pacjentów, filtrów, statystyk i wykresów
# ===============================

from __future__ import annotations
import sys
import os
import math
import numpy as np
import pandas as pd

# Importy PyQt6 — interfejs użytkownika
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTableView,
    QComboBox, QMessageBox, QPlainTextEdit, QTabWidget,
    QFormLayout, QInputDialog, QCheckBox, QGroupBox, QAbstractScrollArea, QStackedWidget, QSpinBox,
    QToolButton, QMenu, QSizePolicy
)
from PyQt6.QtGui import QAction, QGuiApplication
from PyQt6.QtCore import Qt, QAbstractTableModel, QVariant, QTimer

# Import matplotlib (rysowanie wykresów)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

# Własne funkcje logiki (core.py)
from core import (
    load_data, load_sqlite, add_bp_numbers, filter_patients,
    basic_summary
)

# Wymiary pól, przycisków, formularzy
FIELD_H   = 24
FIELD_W   = 240
HALF_W    = 115
BUTTON_W  = 110
BUTTON_H  = 28
FORM_MAX_W = 560
LOG_MAX_H = 80 


# Adapter DataFrame -> QTableView
# Pozwala wyświetlać dane pandas w tabeli PyQt6
class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def set_df(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=None):
        return 0 if self._df is None else len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        val = self._df.iat[index.row(), index.column()]
        return "" if pd.isna(val) else str(val)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or self._df is None:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])


# Canvas dla matplotliba w GUI (rysowanie histogramów, boxplotów, scatterów)
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)
        self.ax = self.figure.add_subplot(111)

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)


# ======================
# Główne okno aplikacji 
# ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Explorer")
        self.resize(1200, 800)

        self.df: pd.DataFrame | None = None
        self.df_view: pd.DataFrame | None = None

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Zakładki
        self.tab_data = QWidget(); tabs.addTab(self.tab_data, "Data")
        self._build_tab_data()

        self.tab_filters = QWidget(); tabs.addTab(self.tab_filters, "Filters")
        self._build_tab_filters()

        self.tab_stats = QWidget(); tabs.addTab(self.tab_stats, "Statistics")
        self._build_tab_stats()

        self.tab_plots = QWidget(); tabs.addTab(self.tab_plots, "Plots")
        self._build_tab_plots()

        self.tab_export = QWidget(); tabs.addTab(self.tab_export, "Export")
        self._build_tab_export()

        # ----------- Logi (mniejsze) -----------
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(LOG_MAX_H)
        self.log.setPlaceholderText("Log…")

        layout_root = QVBoxLayout()
        layout_root.addWidget(tabs)
        layout_root.addWidget(self.log)
        wrapper = QWidget(); wrapper.setLayout(layout_root)
        self.setCentralWidget(wrapper)
        self._log("> Aplikacja gotowa.")

    # ---------- Helpers UI ----------
    def _set_pair_sizes(self, a: QLineEdit, b: QLineEdit):
        a.setFixedSize(HALF_W, FIELD_H)
        b.setFixedSize(HALF_W, FIELD_H)

    def _set_field_size(self, w):
        w.setFixedSize(FIELD_W, FIELD_H)

    def _centered_container(self, widget: QWidget) -> QWidget:
        wrapper = QWidget()
        wrapper.setMaximumWidth(FORM_MAX_W)
        lay = QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)
        lay.addWidget(widget)
        lay.addStretch(1)
        return wrapper

    def _h(self, *widgets):
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for w in widgets:
            lay.addWidget(w)
        lay.addStretch(1)
        c = QWidget(); c.setLayout(lay)
        return c

    def _get_int(self, le: QLineEdit):
        t = le.text().strip()
        return None if t == "" else int(t)

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    # ---------- Jednolity styl przycisków ----------
    def _style_button(self, w):
        w.setFixedSize(BUTTON_W, BUTTON_H)
        w.setStyleSheet("""
            QPushButton, QToolButton {
                background: #2D7DFF;
                color: #FFFFFF;
                font-weight: 600;
                border: 0;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover, QToolButton:hover { background: #2468D6; }
            QPushButton:pressed, QToolButton:pressed { background: #1F5ABD; }
            QPushButton:disabled, QToolButton:disabled { background: #9DBDFF; }
        """)
        return w
# Zakładka "Data"
# Pozwala wczytać plik CSV lub bazę SQLite z danymi pacjentów
# i wyświetla zawartość w tabeli
    def _build_tab_data(self):
        v = QVBoxLayout(self.tab_data)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.le_path = QLineEdit()
        self.le_path.setPlaceholderText("Wybierz plik CSV lub SQLite…")
        self.le_path.setMinimumHeight(FIELD_H)

        btn_browse = self._style_button(QToolButton(self)); btn_browse.setText("Browse…")
        btn_browse.clicked.connect(self.on_browse_any)

        btn_load = self._style_button(QPushButton("Load data"))
        btn_load.clicked.connect(self.on_load_any)

        row.addStretch(1)
        row.addWidget(self.le_path)
        row.addWidget(btn_browse)
        row.addWidget(btn_load)
        row.addStretch(1)
        v.addLayout(row)

        # Tabela danych
        self.table = QTableView()
        self.model = PandasModel()
        self.table.setModel(self.model)
        v.addWidget(self.table)

# Zakładka "Filters"
# Umożliwia filtrowanie pacjentów po wieku, płci, ciśnieniu, HR i objawach
    def _build_tab_filters(self):
        outer = QVBoxLayout(self.tab_filters)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        form_widget = QWidget()
        form_widget.setMaximumWidth(FORM_MAX_W)
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        # Age
        self.le_age_min = QLineEdit(); self.le_age_min.setPlaceholderText("min")
        self.le_age_max = QLineEdit(); self.le_age_max.setPlaceholderText("max")
        self._set_pair_sizes(self.le_age_min, self.le_age_max)
        form.addRow("Age", self._h(self.le_age_min, self.le_age_max))

        # Gender (All/Female/Male)
        self.cb_gender = QComboBox()
        self.cb_gender.addItems(["All patients", "Female", "Male"])
        self._set_field_size(self.cb_gender)
        form.addRow("Gender", self.cb_gender)

        # Systolic
        self.le_sys_min = QLineEdit(); self.le_sys_min.setPlaceholderText("min")
        self.le_sys_max = QLineEdit(); self.le_sys_max.setPlaceholderText("max")
        self._set_pair_sizes(self.le_sys_min, self.le_sys_max)
        form.addRow("Systolic", self._h(self.le_sys_min, self.le_sys_max))

        # Diastolic
        self.le_dia_min = QLineEdit(); self.le_dia_min.setPlaceholderText("min")
        self.le_dia_max = QLineEdit(); self.le_dia_max.setPlaceholderText("max")
        self._set_pair_sizes(self.le_dia_min, self.le_dia_max)
        form.addRow("Diastolic", self._h(self.le_dia_min, self.le_dia_max))

        # Heart Rate
        self.le_hr_min = QLineEdit(); self.le_hr_min.setPlaceholderText("min")
        self.le_hr_max = QLineEdit(); self.le_hr_max.setPlaceholderText("max")
        self._set_pair_sizes(self.le_hr_min, self.le_hr_max)
        form.addRow("Heart Rate", self._h(self.le_hr_min, self.le_hr_max))

        # Symptoms
        self.cb_symptoms = QComboBox()
        self.cb_symptoms.addItems(["All patients", "With symptoms", "Without symptoms"])
        self._set_field_size(self.cb_symptoms)
        form.addRow("Symptoms", self.cb_symptoms)

        # Przyciski pod panelem filtrów
        btn_apply = self._style_button(QPushButton("Apply filters"))
        btn_apply.clicked.connect(self.apply_filter)
        btn_reset = self._style_button(QPushButton("Reset filters"))
        btn_reset.clicked.connect(self.reset_filters)

        btns_line = QHBoxLayout()
        btns_line.setContentsMargins(0, 0, 0, 0)
        btns_line.setSpacing(12)
        btns_line.addWidget(btn_apply)
        btns_line.addWidget(btn_reset)
        btns_line.addStretch(1)
        btns_widget = QWidget(); btns_widget.setLayout(btns_line)

        form.addRow("", btns_widget)

        outer.addWidget(self._centered_container(form_widget), alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)