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
        # Wymuś ten sam font i rozmiar w pt (stabilny na macOS/Retina)
        app_font = QGuiApplication.font()
        size_pt = app_font.pointSize()
        if size_pt <= 0:
            size_pt = 12  # fallback, gdyby pointSize był -1
        w.setFont(app_font)

        w.setFixedSize(BUTTON_W, BUTTON_H)
        w.setStyleSheet(f"""
            QPushButton, QToolButton {{
                background: #2D7DFF;
                color: #FFFFFF;
                font-weight: 600;
                font-size: {size_pt}pt;
                border: 0;
                border-radius: 6px;
                padding: 6px 12px;
            }}
            QPushButton:hover, QToolButton:hover {{ background: #2468D6; }}
            QPushButton:pressed, QToolButton:pressed {{ background: #1F5ABD; }}
            QPushButton:disabled, QToolButton:disabled {{ background: #9DBDFF; }}
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

        # ZAMIANA: Browse… jako QPushButton (zamiast QToolButton)
        btn_browse = self._style_button(QPushButton("Browse…"))
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

    # Zakładka "Statistics"
    # Pozwala generować tabele z podsumowaniami, grupowaniami i porównaniami
    def _build_tab_stats(self):
        v = QVBoxLayout(self.tab_stats)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        self.auto_run_enabled = False
        self._actions = {}

        def make_line(scope_prefix, title, extra_widgets=None, run_slot=None, model_attr=None):
            # hidden state
            self._make_hidden_metric_checkboxes(scope_prefix)
            self._make_hidden_stat_checkboxes(scope_prefix)

            line = QHBoxLayout()
            lbl = QLabel(title)

            # Metrics & Statistics
            btn_metrics = self._style_button(QToolButton(self)); btn_metrics.setText("Metrics ▾")
            metrics_menu = self._action_menu_metrics(scope_prefix)  # has Select All / Reset
            btn_metrics.setMenu(metrics_menu); btn_metrics.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

            btn_stats   = self._style_button(QToolButton(self)); btn_stats.setText("Statistics ▾")
            stats_menu = self._action_menu_stats(scope_prefix)    # has Select All / Reset
            btn_stats.setMenu(stats_menu); btn_stats.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

            btn_run     = self._style_button(QPushButton("Run"))
            if run_slot: btn_run.clicked.connect(run_slot)

            btn_export  = self._style_button(QToolButton(self)); btn_export.setText("Export")
            m_export = QMenu(self)
            m_export.addAction(QAction("Copy to clipboard", self, triggered=lambda: self._copy_table_to_clipboard(getattr(self, model_attr))))
            m_export.addAction(QAction("Export CSV…", self, triggered=lambda: self._export_table_to_csv(getattr(self, model_attr))))
            btn_export.setMenu(m_export); btn_export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

            line.addWidget(lbl); line.addSpacing(8)
            if extra_widgets:
                for w in extra_widgets: line.addWidget(w)
                line.addSpacing(8)
            line.addWidget(btn_metrics); line.addWidget(btn_stats)
            line.addStretch(1); line.addWidget(btn_run); line.addWidget(btn_export)
            return line

        # SUMMARY
        sum_line = make_line("sum", "Summary", run_slot=self._run_summary_section, model_attr="model_stats_summary")
        v.addLayout(sum_line)
        self.table_stats_summary = QTableView()
        self.model_stats_summary = PandasModel()
        self.table_stats_summary.setModel(self.model_stats_summary)
        self._tune_table(self.table_stats_summary)
        v.addWidget(self.table_stats_summary)

        # GROUP BY
        self.grp_by = QComboBox()
        self.grp_by.addItems(["Gender","AgeGroup","Symptoms"])
        self.grp_by.currentIndexChanged.connect(lambda _: (self.auto_run_enabled and self._debounced_recompute()))
        grp_line = make_line("grp", "Group by", extra_widgets=[QLabel("by:"), self.grp_by], run_slot=self._run_group_section, model_attr="model_stats_group")
        v.addLayout(grp_line)
        self.table_stats_group = QTableView()
        self.model_stats_group = PandasModel()
        self.table_stats_group.setModel(self.model_stats_group)
        self._tune_table(self.table_stats_group)
        v.addWidget(self.table_stats_group)

        # COMPARE
        cmp_line = make_line("cmp", "Compare symptoms", run_slot=self._run_compare_section, model_attr="model_stats_compare")
        v.addLayout(cmp_line)
        self.table_stats_compare = QTableView()
        self.model_stats_compare = PandasModel()
        self.table_stats_compare.setModel(self.model_stats_compare)
        self._tune_table(self.table_stats_compare)
        v.addWidget(self.table_stats_compare)

        auto_box = QHBoxLayout()
        self.chk_auto = QCheckBox("Auto-run")
        self.chk_auto.toggled.connect(lambda val: setattr(self, "auto_run_enabled", bool(val)))
        auto_box.addWidget(self.chk_auto); auto_box.addStretch(1)
        v.addLayout(auto_box)

    # Helpers for managing checkbox/action binding
    def _ensure_actions_bucket(self, scope):
        if scope not in self._actions:
            self._actions[scope] = {'metrics': {}, 'stats': {}}

    def _get_scope_maps(self, scope_prefix: str):
        metrics = {
            "HeartRate": getattr(self, f"{scope_prefix}_hr"),
            "Systolic": getattr(self, f"{scope_prefix}_sys"),
            "Diastolic": getattr(self, f"{scope_prefix}_dia"),
        }
        stats = {name: getattr(self, f"{scope_prefix}_{name}") for name in ["count","mean","median","std","min","q1","q3","max"]}
        return metrics, stats

    def _bulk_set(self, scope: str, kind: str, checked: bool):
        maps = self._get_scope_maps(scope)[0 if kind=='metrics' else 1]
        for name, cb in maps.items():
            cb.setChecked(checked)
        self._ensure_actions_bucket(scope)
        for name, act in self._actions[scope][kind].items():
            act.blockSignals(True)
            act.setChecked(maps[name].isChecked())
            act.blockSignals(False)
        if getattr(self, "auto_run_enabled", False):
            self._debounced_recompute()

    # ---------- Tuning QTableView ----------
    def _tune_table(self, view: QTableView):
        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        vh = view.verticalHeader()
        hh = view.horizontalHeader()
        view.setWordWrap(False)
        view.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        view.setVerticalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        vh.setVisible(False)
        hh.setStretchLastSection(False)

    # ---------- Common helpers ----------
    def _make_hidden_metric_checkboxes(self, scope_prefix: str):
        setattr(self, f"{scope_prefix}_hr", QCheckBox("HeartRate"))
        setattr(self, f"{scope_prefix}_sys", QCheckBox("Systolic"))
        setattr(self, f"{scope_prefix}_dia", QCheckBox("Diastolic"))
        for name in ("hr","sys","dia"):
            getattr(self, f"{scope_prefix}_{name}").setChecked(True)

    def _make_hidden_stat_checkboxes(self, scope_prefix: str):
        names = ["count","mean","median","std","min","q1","q3","max"]
        for n in names:
            setattr(self, f"{scope_prefix}_{n}", QCheckBox(n))
        for n in ["count","mean","median"]:
            getattr(self, f"{scope_prefix}_{n}").setChecked(True)

    def _action_for_checkbox(self, scope: str, kind: str, label: str, checkbox: QCheckBox) -> QAction:
        self._ensure_actions_bucket(scope)
        act = QAction(label, self)
        act.setCheckable(True)
        act.setChecked(checkbox.isChecked())
        def sync_from_action(checked):
            checkbox.setChecked(checked)
            if getattr(self, "auto_run_enabled", False):
                self._debounced_recompute()
        act.toggled.connect(sync_from_action)
        self._actions[scope][kind][label] = act
        return act

    def _action_menu_metrics(self, scope_prefix: str) -> QMenu:
        m = QMenu("Metrics", self)
        metrics_map, _ = self._get_scope_maps(scope_prefix)
        act_all = QAction("Select all metrics", self)
        act_all.triggered.connect(lambda: self._bulk_set(scope_prefix, 'metrics', True))
        act_none = QAction("Reset metrics", self)
        act_none.triggered.connect(lambda: self._bulk_set(scope_prefix, 'metrics', False))
        m.addAction(act_all); m.addAction(act_none); m.addSeparator()
        for label, cb in metrics_map.items():
            m.addAction(self._action_for_checkbox(scope_prefix, 'metrics', label, cb))
        return m

    def _action_menu_stats(self, scope_prefix: str) -> QMenu:
        m = QMenu("Statistics", self)
        _, stats_map = self._get_scope_maps(scope_prefix)
        act_all = QAction("Select all statistics", self)
        act_all.triggered.connect(lambda: self._bulk_set(scope_prefix, 'stats', True))
        act_none = QAction("Reset statistics", self)
        act_none.triggered.connect(lambda: self._bulk_set(scope_prefix, 'stats', False))
        m.addAction(act_all); m.addAction(act_none); m.addSeparator()
        for label, cb in stats_map.items():
            m.addAction(self._action_for_checkbox(scope_prefix, 'stats', label, cb))
        return m

    def _debounced_recompute(self):
        if not hasattr(self, "_recompute_timer"):
            self._recompute_timer = QTimer(self)
            self._recompute_timer.setSingleShot(True)
            self._recompute_timer.timeout.connect(self._run_current_stats_section)
        self._recompute_timer.start(350)

    def _run_current_stats_section(self):
        self._run_summary_section()
        self._run_group_section()
        self._run_compare_section()

    def _export_table_to_csv(self, model: QAbstractTableModel):
        if model is None or model.rowCount() == 0:
            QMessageBox.information(self, "Export", "Brak danych do eksportu.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV (*.csv)")
        if not path: return
        data = []
        headers = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())]
        for r in range(model.rowCount()):
            row = [model.data(model.index(r, c), Qt.ItemDataRole.DisplayRole) for c in range(model.columnCount())]
            data.append(row)
        pd.DataFrame(data, columns=headers).to_csv(path, index=False)
        QMessageBox.information(self, "Export", f"Zapisano: {path}")

    def _copy_table_to_clipboard(self, model: QAbstractTableModel):
        if model is None or model.rowCount() == 0:
            return
        data = []
        headers = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())]
        data.append(headers)
        for r in range(model.rowCount()):
            row = [model.data(model.index(r, c), Qt.ItemDataRole.DisplayRole) for c in range(model.columnCount())]
            data.append(row)
        tsv = "\n".join(["\t".join(map(str, row)) for row in data])
        QGuiApplication.clipboard().setText(tsv)
        self._log("> Table copied to clipboard.")

    # ----------------- STATISTICS actions -----------------
    def _collect_metrics(self, flags: tuple[QCheckBox, QCheckBox, QCheckBox]) -> list[str]:
        hr, sys, dia = flags
        cols = []
        if hr.isChecked():  cols.append("HeartRate")
        if sys.isChecked(): cols.append("Systolic")
        if dia.isChecked(): cols.append("Diastolic")
        if self.df_view is not None:
            cols = [c for c in cols if c in self.df_view.columns]
        return cols

    def _collect_stats(self, boxes: tuple[QCheckBox, ...]):
        fns = []
        if boxes[0].isChecked(): fns.append("count")
        if boxes[1].isChecked(): fns.append("mean")
        if boxes[2].isChecked(): fns.append("median")
        if boxes[3].isChecked(): fns.append("std")
        if boxes[4].isChecked(): fns.append("min")
        if boxes[5].isChecked():
            def q1(x): return x.quantile(0.25)
            q1.__name__ = "q1"; fns.append(q1)
        if boxes[6].isChecked():
            def q3(x): return x.quantile(0.75)
            q3.__name__ = "q3"; fns.append(q3)
        if boxes[7].isChecked(): fns.append("max")
        return fns

    def _round2_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors="ignore").round(2)

    def _ensure_agegroup(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Age" not in df.columns:
            return df
        ages = pd.to_numeric(df["Age"], errors="coerce").dropna()
        if ages.empty:
            return df
        max_age = int(math.ceil(ages.max()))
        top = max(20, int(math.ceil(max_age / 20.0) * 20))
        bins = list(range(0, top + 1, 20))
        labels = []
        for i in range(1, len(bins)):
            lo = 0 if i == 1 else bins[i-1] + 1
            hi = bins[i]
            labels.append(f"{lo}–{hi}")
        out = df.copy()
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
        out["AgeGroup"] = pd.cut(out["Age"], bins=bins, labels=labels, right=True, include_lowest=True)
        return out

    def _run_summary_section(self):
        if self.df_view is None or self.df_view.empty:
            return
        cols = self._collect_metrics((self.sum_hr, self.sum_sys, self.sum_dia))
        if not cols:
            return
        fns = self._collect_stats((self.sum_count, self.sum_mean, self.sum_median,
                                   self.sum_std, self.sum_min, self.sum_q1,
                                   self.sum_q3, self.sum_max))
        if not fns:
            return
        out = self.df_view[cols].agg(fns)
        if isinstance(out, pd.Series):
            out = out.to_frame().T
        if out.index.nlevels == 1:
            out = out.T
        out = self._round2_df(out).reset_index().rename(columns={"index": "Metric"})
        self.model_stats_summary.set_df(out)
        self._log("> Summary computed.")

    def _run_group_section(self):
        if self.df_view is None or self.df_view.empty:
            return
        by = self.grp_by.currentText()
        df = self._ensure_agegroup(self.df_view) if by == "AgeGroup" else self.df_view
        cols = self._collect_metrics((self.grp_hr, self.grp_sys, self.grp_dia))
        if not cols:
            return
        fns = self._collect_stats((self.grp_count, self.grp_mean, self.grp_median,
                                   self.grp_std, self.grp_min, self.grp_q1,
                                   self.grp_q3, self.grp_max))
        if not fns:
            return
        g = df.groupby(by, dropna=False)[cols].agg(fns)
        g.columns = ["_".join([c for c in col if c]).strip() if isinstance(col, tuple) else str(col)
                     for col in g.columns.values]
        g = self._round2_df(g).reset_index()
        self.model_stats_group.set_df(g)
        self._log(f"> Group by '{by}' computed.")

    def _run_compare_section(self):
        if self.df_view is None or self.df_view.empty:
            return
        cols = self._collect_metrics((self.cmp_hr, self.cmp_sys, self.cmp_dia))
        if not cols:
            return
        fns = self._collect_stats((self.cmp_count, self.cmp_mean, self.cmp_median,
                                   self.cmp_std, self.cmp_min, self.cmp_q1,
                                   self.cmp_q3, self.cmp_max))
        if not fns:
            return
        df = self.df_view.assign(
            SymptomStatus=self.df_view["Symptoms"].notna().map({True: "With symptoms", False: "Without symptoms"})
        )
        g = df.groupby("SymptomStatus", dropna=False)[cols].agg(fns)
        g.columns = ["_".join([c for c in col if c]).strip() if isinstance(col, tuple) else str(col)
                     for col in g.columns.values]
        g = self._round2_df(g).reset_index()
        self.model_stats_compare.set_df(g)
        self._log("> Compare symptoms computed.")

    # ----------------- PLOTS -----------------
    def _build_tab_plots(self):
        layout = QHBoxLayout(self.tab_plots)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Panel LEWY (opcje)
        controls_box = QGroupBox("Plots")
        controls_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        controls = QVBoxLayout(controls_box)
        controls.setSpacing(40)
        controls.setContentsMargins(8, 8, 8, 8)

        # Segment wyboru typu
        segment_row = QHBoxLayout()
        segment_row.setContentsMargins(0, 0, 0, 0)
        segment_row.setSpacing(6)

        self.btn_hist = self._style_button(QToolButton(self)); self.btn_hist.setText("Histogram"); self.btn_hist.setCheckable(True)
        self.btn_scat = self._style_button(QToolButton(self)); self.btn_scat.setText("Scatter");   self.btn_scat.setCheckable(True)
        self.btn_box  = self._style_button(QToolButton(self)); self.btn_box.setText("Boxplot");    self.btn_box.setCheckable(True)

        def set_choice(which):
            self.btn_hist.setChecked(which=="hist")
            self.btn_scat.setChecked(which=="scat")
            self.btn_box.setChecked(which=="box")
            idx = {"hist":0,"scat":1,"box":2}[which]
            self.custom_stack.setCurrentIndex(idx)
            self.preset_stack.setCurrentIndex(idx)
            self._plot_choice = which
            # tryb custom -> bez dekad
            self._hist_use_decade_bins = False
            self._clear_preset_only_ui()

        self.btn_hist.clicked.connect(lambda: set_choice("hist"))
        self.btn_scat.clicked.connect(lambda: set_choice("scat"))
        self.btn_box.clicked.connect(lambda: set_choice("box"))
        segment_row.addWidget(self.btn_hist); segment_row.addWidget(self.btn_scat); segment_row.addWidget(self.btn_box)
        controls.addLayout(segment_row)

        # ===== Panel PRESET (pod przyciskami) =====
        self.preset_stack = QStackedWidget()
        self.preset_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.preset_stack.setContentsMargins(0, 0, 0, 0)

        # --- Presety: Histogram ---
        preset_hist = QWidget()
        ph_row = QHBoxLayout(preset_hist)
        ph_row.setContentsMargins(0, 0, 0, 0)
        ph_row.setSpacing(6)
        self.lbl_hist_presets = QLabel("Presets:")
        ph_row.addWidget(self.lbl_hist_presets)
        self.cb_hist_preset = QComboBox()
        self.cb_hist_preset.addItems([
            "— select preset —",
            "Age — decades dynamic",
            "Age — 4 equal-width bins",
            "Systolic — 10 bins",
            "Diastolic — 10 bins",
            "HeartRate — 10 bins",
        ])
        self.cb_hist_preset.setFixedHeight(FIELD_H)
        btn_clear_hist = self._style_button(QToolButton(self)); btn_clear_hist.setText("Clear preset")
        btn_clear_hist.clicked.connect(lambda: self._clear_preset("hist"))
        self.cb_hist_preset.currentIndexChanged.connect(lambda _: self._on_preset_change("hist"))
        ph_row.addWidget(self.cb_hist_preset); ph_row.addWidget(btn_clear_hist); ph_row.addStretch(1)
        self.preset_stack.addWidget(preset_hist)

        # --- Presety: Scatter ---
        preset_scat = QWidget()
        ps_row = QHBoxLayout(preset_scat)
        ps_row.setContentsMargins(0, 0, 0, 0)
        ps_row.setSpacing(6)
        self.lbl_scat_presets = QLabel("Presets:")
        ps_row.addWidget(self.lbl_scat_presets)
        self.cb_scat_preset = QComboBox()
        self.cb_scat_preset.addItems([
            "— select preset —",
            "Age vs Systolic (color by gender)",
            "Age vs HeartRate (color by gender)",
        ])
        self.cb_scat_preset.setFixedHeight(FIELD_H)
        btn_clear_scat = self._style_button(QToolButton(self)); btn_clear_scat.setText("Clear preset")
        btn_clear_scat.clicked.connect(lambda: self._clear_preset("scat"))
        self.cb_scat_preset.currentIndexChanged.connect(lambda _: self._on_preset_change("scat"))
        ps_row.addWidget(self.cb_scat_preset); ps_row.addWidget(btn_clear_scat); ps_row.addStretch(1)
        self.preset_stack.addWidget(preset_scat)

        # --- Presety: Boxplot ---
        preset_box = QWidget()
        pb_row = QHBoxLayout(preset_box)
        pb_row.setContentsMargins(0, 0, 0, 0)
        pb_row.setSpacing(6)
        self.lbl_box_presets = QLabel("Presets:")
        pb_row.addWidget(self.lbl_box_presets)
        self.cb_box_preset = QComboBox()
        self.cb_box_preset.addItems([
            "— select preset —",
            "Systolic by Gender",
            "Diastolic by Gender",
            "HeartRate by Gender",
            "Age by Gender",
        ])
        self.cb_box_preset.setFixedHeight(FIELD_H)
        btn_clear_box = self._style_button(QToolButton(self)); btn_clear_box.setText("Clear preset")
        btn_clear_box.clicked.connect(lambda: self._clear_preset("box"))
        self.cb_box_preset.currentIndexChanged.connect(lambda _: self._on_preset_change("box"))
        pb_row.addWidget(self.cb_box_preset); pb_row.addWidget(btn_clear_box); pb_row.addStretch(1)
        self.preset_stack.addWidget(preset_box)

        controls.addWidget(self.preset_stack)

        # ===== Stacked panel z formularzami CUSTOM =====
        self.custom_stack = QStackedWidget()
        self.custom_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.custom_stack.setContentsMargins(0, 0, 0, 0)

        # --- Histogram (custom) ---
        hist_page = QWidget()
        hist_page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        hist_form = QFormLayout(hist_page)
        hist_form.setContentsMargins(0, 0, 0, 0)
        hist_form.setHorizontalSpacing(10)
        hist_form.setVerticalSpacing(6)
        self.cb_hist_metric = QComboBox(); self.cb_hist_metric.addItems(["Age","HeartRate","Systolic","Diastolic"])
        self.sp_hist_bins = QSpinBox(); self.sp_hist_bins.setRange(2, 100); self.sp_hist_bins.setValue(10)
        hist_form.addRow("Metric:", self.cb_hist_metric)
        hist_form.addRow("Bins:", self.sp_hist_bins)
        self.custom_stack.addWidget(hist_page)

        # --- Scatter (custom) ---
        scatter_page = QWidget()
        scatter_page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        scat_form = QFormLayout(scatter_page)
        scat_form.setContentsMargins(0, 0, 0, 0)
        scat_form.setHorizontalSpacing(10)
        scat_form.setVerticalSpacing(6)
        self.cb_scatter_x = QComboBox(); self.cb_scatter_x.addItems(["Age","HeartRate","Systolic","Diastolic"])
        self.cb_scatter_y = QComboBox(); self.cb_scatter_y.addItems(["HeartRate","Systolic","Diastolic","Age"])
        self.chk_scatter_color_gender = QCheckBox("Color by Gender")
        scat_form.addRow("X axis:", self.cb_scatter_x)
        scat_form.addRow("Y axis:", self.cb_scatter_y)
        scat_form.addRow("", self.chk_scatter_color_gender)
        self.custom_stack.addWidget(scatter_page)

        # --- Boxplot (custom) ---
        box_page = QWidget()
        box_page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        box_form = QFormLayout(box_page)
        box_form.setContentsMargins(0, 0, 0, 0)
        box_form.setHorizontalSpacing(10)
        box_form.setVerticalSpacing(6)
        self.cb_box_metric = QComboBox(); self.cb_box_metric.addItems(["HeartRate","Systolic","Diastolic","Age"])
        self.chk_box_means = QCheckBox("Show means"); self.chk_box_means.setChecked(True)
        box_form.addRow("Metric:", self.cb_box_metric)
        box_form.addRow("", self.chk_box_means)
        self.custom_stack.addWidget(box_page)

        controls.addWidget(self.custom_stack)

        # Przyciski rysowania i zapisu
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self.btn_draw_plot = self._style_button(QPushButton("Draw"))
        self.btn_save_plot = self._style_button(QPushButton("Save plot…"))
        self.btn_draw_plot.clicked.connect(self.draw_current_plot)
        self.btn_save_plot.clicked.connect(self.save_current_plot)
        btn_row.addWidget(self.btn_draw_plot); btn_row.addWidget(self.btn_save_plot); btn_row.addStretch(1)
        controls.addLayout(btn_row)

        # PRAWY panel (canvas)
        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(controls_box, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.canvas, 1)

        # --- stan dla histogramu (preset dekad) ---
        self._hist_use_decade_bins = False

        # >>> Ujednolicenie szerokości pól i wyrównanie „Clear preset” do „Boxplot”
        self._align_and_unify_control_widths(segment_row)

        # Domyślny wybór
        set_choice("hist")

    def _align_and_unify_control_widths(self, segment_row: QHBoxLayout):
        labels = [self.lbl_hist_presets, self.lbl_scat_presets, self.lbl_box_presets]
        max_lbl_w = max(l.sizeHint().width() for l in labels)
        for l in labels:
            l.setFixedWidth(max_lbl_w)

        segment_spacing = segment_row.spacing() if segment_row.spacing() >= 0 else 6
        preset_row_spacing = 6

        input_w = 2 * BUTTON_W + 2 * segment_spacing - preset_row_spacing - max_lbl_w
        input_w = max(input_w, 160)
        # input_w = 280  # <- opcjonalnie stała szerokość

        inputs = [
            self.cb_hist_preset, self.cb_scat_preset, self.cb_box_preset,
            self.cb_hist_metric, self.sp_hist_bins,
            self.cb_scatter_x, self.cb_scatter_y,
            self.cb_box_metric,
        ]
        for w in inputs:
            w.setFixedWidth(input_w)
            if isinstance(w, QComboBox):
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

    # ====== Obsługa presetów ======
    def _clear_preset_only_ui(self):
        self.custom_stack.setVisible(True)

    def _clear_preset(self, which: str):
        if which == "hist":
            self.cb_hist_preset.blockSignals(True); self.cb_hist_preset.setCurrentIndex(0); self.cb_hist_preset.blockSignals(False)
            self._hist_use_decade_bins = False
        elif which == "scat":
            self.cb_scat_preset.blockSignals(True); self.cb_scat_preset.setCurrentIndex(0); self.cb_scat_preset.blockSignals(False)
        elif which == "box":
            self.cb_box_preset.blockSignals(True); self.cb_box_preset.setCurrentIndex(0); self.cb_box_preset.blockSignals(False)
        self.custom_stack.setVisible(True)

    def _on_preset_change(self, which: str):
        if which == "hist":
            idx = self.cb_hist_preset.currentIndex()
            if idx <= 0:
                self._hist_use_decade_bins = False
                self.custom_stack.setVisible(True)
                return
            name = self.cb_hist_preset.currentText()
            self._apply_hist_preset(name)
            self.custom_stack.setVisible(False)
        elif which == "scat":
            idx = self.cb_scat_preset.currentIndex()
            if idx <= 0:
                self.custom_stack.setVisible(True)
                return
            name = self.cb_scat_preset.currentText()
            self._apply_scat_preset(name)
            self.custom_stack.setVisible(False)
        elif which == "box":
            idx = self.cb_box_preset.currentIndex()
            if idx <= 0:
                self.custom_stack.setVisible(True)
                return
            name = self.cb_box_preset.currentText()
            self._apply_box_preset(name)
            self.custom_stack.setVisible(False)

    def _apply_hist_preset(self, name: str):
        if name == "Age — decades dynamic":
            self.cb_hist_metric.setCurrentText("Age")
            self._hist_use_decade_bins = True
        elif name == "Age — 4 equal-width bins":
            self.cb_hist_metric.setCurrentText("Age")
            self.sp_hist_bins.setValue(4)
            self._hist_use_decade_bins = False
        elif name == "Systolic — 10 bins":
            self.cb_hist_metric.setCurrentText("Systolic")
            self.sp_hist_bins.setValue(10)
            self._hist_use_decade_bins = False
        elif name == "Diastolic — 10 bins":
            self.cb_hist_metric.setCurrentText("Diastolic")
            self.sp_hist_bins.setValue(10)
            self._hist_use_decade_bins = False
        elif name == "HeartRate — 10 bins":
            self.cb_hist_metric.setCurrentText("HeartRate")
            self.sp_hist_bins.setValue(10)
            self._hist_use_decade_bins = False

    def _apply_scat_preset(self, name: str):
        if name == "Age vs Systolic (color by gender)":
            self.cb_scatter_x.setCurrentText("Age")
            self.cb_scatter_y.setCurrentText("Systolic")
            self.chk_scatter_color_gender.setChecked(True)
        elif name == "Age vs HeartRate (color by gender)":
            self.cb_scatter_x.setCurrentText("Age")
            self.cb_scatter_y.setCurrentText("HeartRate")
            self.chk_scatter_color_gender.setChecked(True)

    def _apply_box_preset(self, name: str):
        if name == "Systolic by Gender":
            self.cb_box_metric.setCurrentText("Systolic")
            self.chk_box_means.setChecked(True)
        elif name == "Diastolic by Gender":
            self.cb_box_metric.setCurrentText("Diastolic")
            self.chk_box_means.setChecked(True)
        elif name == "HeartRate by Gender":
            self.cb_box_metric.setCurrentText("HeartRate")
            self.chk_box_means.setChecked(True)
        elif name == "Age by Gender":
            self.cb_box_metric.setCurrentText("Age")
            self.chk_box_means.setChecked(True)

    def _current_plot_type(self):
        return getattr(self, "_plot_choice", "hist")

    def save_current_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", filter="PNG (*.png)")
        if not path: return
        try:
            self.canvas.figure.savefig(path, bbox_inches="tight", dpi=150)
            QMessageBox.information(self, "Save plot", f"Saved: {path}")
            self._log(f"> Plot saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save plot", str(e))

    # === Etykiety z jednostkami ===
    def _metric_label(self, name: str) -> str:
        units = {
            "Age": "years",
            "Systolic": "mmHg",
            "Diastolic": "mmHg",
            "HeartRate": "bpm"
        }
        u = units.get(name, "")
        return f"{name} [{u}]" if u else name

    def draw_current_plot(self):
        if self.df_view is None or self.df_view.empty:
            QMessageBox.information(self, "Plot", "Brak danych po filtrach."); return
        pt = self._current_plot_type()
        if pt == "hist":   self._draw_histogram()
        elif pt == "scat": self._draw_scatter()
        elif pt == "box":  self._draw_boxplot_gender()
        else: QMessageBox.information(self, "Plot", "Nieobsługiwany typ wykresu.")

    def _draw_histogram(self):
        metric = self.cb_hist_metric.currentText()
        series = pd.to_numeric(self.df_view[metric], errors="coerce").dropna()
        if series.empty:
            QMessageBox.information(self, "Histogram", f"Brak danych w kolumnie {metric}."); return
        self.canvas.clear(); ax = self.canvas.ax
        if metric == "Age" and getattr(self, "_hist_use_decade_bins", False):
            max_age = int(math.ceil(series.max()))
            top = max(20, int(math.ceil(max_age / 20.0) * 20))
            bins = list(range(0, top + 1, 20))
            ax.hist(series, bins=bins, edgecolor="black")
            ax.set_xticks(bins); ax.set_xlim(bins[0], bins[-1])
            ax.set_xlabel(self._metric_label("Age")); ax.set_ylabel("Count [patients]"); ax.set_title("Age histogram (decades)")
        else:
            bins = int(self.sp_hist_bins.value())
            ax.hist(series, bins=bins, edgecolor="black")
            ax.set_xlabel(self._metric_label(metric)); ax.set_ylabel("Count [patients]"); ax.set_title(f"Histogram: {metric} (bins={bins})")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
        self.canvas.draw(); self._log(f"> Plot: histogram {metric}")

    def _draw_scatter(self):
        x = self.cb_scatter_x.currentText(); y = self.cb_scatter_y.currentText()
        if x == y: QMessageBox.information(self, "Scatter", "Wybierz różne kolumny dla osi X i Y."); return
        df = self.df_view.copy()
        df[x] = pd.to_numeric(df[x], errors="coerce"); df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[x, y])
        if df.empty: QMessageBox.information(self, "Scatter", "Brak danych do wykresu."); return
        self.canvas.clear(); ax = self.canvas.ax
        if self.chk_scatter_color_gender.isChecked() and "Gender" in df.columns:
            for g, sub in df.groupby("Gender"):
                ax.scatter(sub[x], sub[y], label=str(g), alpha=0.7)
            ax.legend(title="Gender")
        else:
            ax.scatter(df[x], df[y], alpha=0.7)
        ax.set_xlabel(self._metric_label(x)); ax.set_ylabel(self._metric_label(y)); ax.set_title(f"{x} vs {y}")
        ax.grid(True, linestyle=":", linewidth=0.8)
        self.canvas.draw(); self._log(f"> Plot: scatter {x} vs {y}")

    def _draw_boxplot_gender(self):
        metric = self.cb_box_metric.currentText()
        if "Gender" not in self.df_view.columns:
            QMessageBox.information(self, "Boxplot", "Brak kolumny Gender."); return
        df = self.df_view.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        f_vals = df.loc[df["Gender"] == "F", metric].dropna()
        m_vals = df.loc[df["Gender"] == "M", metric].dropna()
        if f_vals.empty and m_vals.empty:
            QMessageBox.information(self, "Boxplot", "Brak danych do wykresu."); return
        self.canvas.clear(); ax = self.canvas.ax
        data, labels = [], []
        if not f_vals.empty: data.append(f_vals.values); labels.append("Female")
        if not m_vals.empty: data.append(m_vals.values); labels.append("Male")
        ax.boxplot(data, labels=labels, showmeans=self.chk_box_means.isChecked())
        ax.set_ylabel(self._metric_label(metric)); ax.set_title(f"{metric} by Gender")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
        self.canvas.draw(); self._log(f"> Plot: boxplot {metric} by gender")

    # ---------- Zakładka EXPORT ----------
    def _build_tab_export(self):
        v = QVBoxLayout(self.tab_export)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        info = QLabel("Eksport danych lub raportu PDF.")
        btn_csv = self._style_button(QPushButton("Export CSV")); btn_csv.clicked.connect(self.export_csv)
        btn_pdf = self._style_button(QPushButton("Export PDF")); btn_pdf.clicked.connect(self.export_pdf_report)

        row = QHBoxLayout()
        row.addStretch(1); row.addWidget(btn_csv); row.addWidget(btn_pdf); row.addStretch(1)

        v.addWidget(info, alignment=Qt.AlignmentFlag.AlignHCenter)
        v.addLayout(row)
        v.addStretch(1)

    # ---------- Browsing / Loading ----------
    def on_browse_any(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data source",
            filter="CSV (*.csv);;SQLite (*.db *.sqlite *.sqlite3);;All files (*.*)"
        )
        if path:
            self.le_path.setText(path)

    def on_load_any(self):
        path = self.le_path.text().strip()
        if not path:
            QMessageBox.information(self, "Load", "Wybierz plik CSV lub SQLite.")
            return

        try:
            lower = path.lower()
            if lower.endswith(".csv"):
                self.df = load_data(path)
            elif lower.endswith(".db") or lower.endswith(".sqlite") or lower.endswith(".sqlite3"):
                table, ok = QInputDialog.getText(self, "SQLite table", "Table name:")
                if not ok or not table.strip():
                    return
                self.df = load_sqlite(path, table.strip())
            else:
                try:
                    self.df = load_data(path)
                except Exception:
                    table, ok = QInputDialog.getText(self, "SQLite table", "Table name:")
                    if not ok or not table.strip():
                        return
                    self.df = load_sqlite(path, table.strip())

            self.df = add_bp_numbers(self.df)
            self.df_view = self.df.copy()
            self.model.set_df(self.df_view)
            QMessageBox.information(self, "Loaded", f"Rows: {len(self.df)}")
            self._log(f"> Data loaded: {os.path.basename(path)} (rows: {len(self.df)})")

        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self._log(f"! Load error: {e}")

    # ---------- Filters ----------
    def apply_filter(self):
        if self.df is None:
            return

        gender_ui = self.cb_gender.currentText()
        if gender_ui == "All patients":
            gender = None
        elif gender_ui == "Female":
            gender = "F"
        else:
            gender = "M"

        sym_sel = self.cb_symptoms.currentText()
        if sym_sel == "All patients":
            only_missing = None
        elif sym_sel == "With symptoms":
            only_missing = False
        else:
            only_missing = True

        self.df_view = filter_patients(
            self.df,
            age_min=self._get_int(self.le_age_min),
            age_max=self._get_int(self.le_age_max),
            gender=gender,
            systolic_min=self._get_int(self.le_sys_min),
            systolic_max=self._get_int(self.le_sys_max),
            diastolic_min=self._get_int(self.le_dia_min),
            diastolic_max=self._get_int(self.le_dia_max),
            hr_min=self._get_int(self.le_hr_min),
            hr_max=self._get_int(self.le_hr_max),
            only_missing_symptom=only_missing,
        )
        self.model.set_df(self.df_view)
        self._log("> Filters applied.")

    def reset_filters(self):
        if self.df is None:
            return
        for le in [
            self.le_age_min, self.le_age_max,
            self.le_sys_min, self.le_sys_max,
            self.le_dia_min, self.le_dia_max,
            self.le_hr_min, self.le_hr_max,
        ]:
            le.clear()
        self.cb_gender.setCurrentText("All patients")
        self.cb_symptoms.setCurrentText("All patients")
        self.df_view = self.df.copy()
        self.model.set_df(self.df_view)
        self._log("> Filters reset.")

    # ---------- Export ----------
    def export_csv(self):
        if self.df_view is None or self.df_view.empty:
            QMessageBox.warning(self, "Export", "Brak danych.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV (*.csv)")
        if not path:
            return
        self.df_view.to_csv(path, index=False)
        QMessageBox.information(self, "Export", f"Zapisano: {path}")
        self._log(f"> Exported CSV: {path}")

    def export_pdf_report(self):
        if self.df_view is None or self.df_view.empty:
            QMessageBox.warning(self, "Export", "Brak danych.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", filter="PDF (*.pdf)")
        if not path:
            return
        with PdfPages(path) as pdf:
            fig = Figure(figsize=(8.27, 11.69))  # A4
            ax = fig.add_subplot(111); ax.axis("off")
            lines = [f"Patient summary (rows: {len(self.df_view)})", ""]
            bs = basic_summary(self.df_view)
            lines += bs.round(2).to_string().splitlines()
            ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
            pdf.savefig(fig); fig.clear()
        QMessageBox.information(self, "Export", f"PDF saved: {path}")
        self._log(f"> Exported PDF: {path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
