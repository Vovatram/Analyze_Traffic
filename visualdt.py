"""
Combined Track Data Viewer
===========================
Merges the table-heatmap viewer (3243.py) and the interactive chart
viewer (ret.py) into a single window with:

  - A dropdown to switch between multiple data files/tables
  - A "Table" tab (color-coded heatmap grid, 3 color-scale modes)
  - A "Chart" tab (line graph + save-as-image button)
  - A "Series" tab and a "Rows" tab that are SHARED: by default they
    control the main table/chart (columns / dates), but as soon as you
    pick a source on the Distribution tab, they switch to show that
    file's lanes / times instead, and control the histogram
  - A "Distribution" tab: just pick a source file (Distance/Speed);
    which times/lanes go into the histogram is chosen back on the
    Rows and Series tabs
  (Series/Rows filters apply to the table, chart, and distribution alike,
  and hidden series/rows are remembered when you switch files)

CONFIGURE THESE TWO LINES FOR YOUR SETUP:
"""

from pathlib import Path


def _discover_road_files(directory):
    """
    Road-tab files look like a Python dict literal: their first
    non-blank line starts with "{". Scan every .txt file in the
    folder and pick out the ones that match, so you don't have to
    list them by hand.
    """
    found = []
    if not directory.exists():
        return found
    for path in sorted(directory.glob("*.txt")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if line.startswith("{"):
                            found.append(path.stem)
                        break
        except OSError:
            continue
    return found


# 1) Folder that contains your .txt data files
DATA_DIR = Path(__file__).resolve().parent / "all_tracks"

# 2) File names WITHOUT the ".txt" extension, e.g. ["average_distancedataset", "other_file"]
FILE_NAMES = ['swiches_cardataset', 'unic_carsdataset', 'intensity_carsdataset', 'average_speed_carsdataset', 'med_speed_carsdataset', 'V85_speed_carsdataset', 'number_distancedataset', 'average_distancedataset', 'min_distancedataset']


# 3) Files used ONLY by the Distribution tab. These have a different shape:
#    time <tab> id ("seria") <tab> many raw, unlabeled sample values.
#    Change DIST_DATA_DIR if these live in a different folder than DATA_DIR.
DIST_DATA_DIR = DATA_DIR
DIST_FILE_NAMES = ["cars_distancedataset", "cars_speeddataset"]

# 4) Files used ONLY by the Road tab. These are Python-dict-literal files:
#    { timestamp: [ {car_id, point_x, point_y, obj_width, obj_length, lane,
#    [obj_speed]}, ... ], ... }. If a reading has no "obj_speed", speed is
#    calculated from how far that car_id moved since its previous reading.
#    Auto-detected below from every .txt file in ROAD_DATA_DIR that looks
#    like a dict literal — edit the line below only if you'd rather list
#    them yourself.
ROAD_DATA_DIR = DATA_DIR
ROAD_FILE_NAMES = _discover_road_files(ROAD_DATA_DIR)

# ----------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox
import math
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

TIME_WIDTH = 155
NAME_WIDTH = 190
CELL_WIDTH = 90
HEADER_HEIGHT = 34
ROW_HEIGHT = 27

ROAD_LANE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46b3a1", "#f032e6", "#9a9a00", "#fabebe", "#008080",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t", na_values=["-", ""], keep_default_na=True)
    df.columns = [str(c).strip() for c in df.columns]
    if "time" not in df.columns:
        raise ValueError(f"'time' column was not found. Columns: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["time"])

    # "time" and, if present, "name" are identifier columns and stay as text.
    # Everything else is a numeric series.
    for column in df.columns:
        if column not in ("time", "name"):
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    df = df.sort_values("time").reset_index(drop=True)
    return df


def load_raw_samples(filepath):
    """
    Loader for the Distribution-only files: no header row, and each line
    has a DIFFERENT number of fields (rows are ragged — a reading can have
    anywhere from a handful to thousands of raw samples).
    Column 0 = time, column 1 = an id ("seria" — the lane a car travels
    in), the rest are unlabeled raw sample values, auto-named col1,
    col2, col3, ... and padded with NaN so every row lines up in one
    DataFrame.
    """
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            rows.append(line.split("\t"))

    max_len = max(len(r) for r in rows)
    rows = [r + [None] * (max_len - len(r)) for r in rows]

    n_extra = max_len - 2
    df = pd.DataFrame(rows, columns=["time", "seria"] + [f"col{i}" for i in range(1, n_extra + 1)])

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["seria"] = pd.to_numeric(df["seria"], errors="coerce")

    for c in df.columns:
        if c not in ("time", "seria"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["time", "seria"]).reset_index(drop=True)
    return df


def load_road_scene(filepath):
    """
    Loader for Road-tab files: a Python-dict literal keyed by timestamp.
    Each timestamp's value is either:
      - a plain list of car dicts: [ {car_id, point_x, point_y, obj_width,
        obj_length, lane, [obj_speed]}, ... ]
      - or {"cars": [...], "distance": [{"car_ids": [id_a, id_b], "length": m}, ...]}
        where car_ids are actual car_id VALUES (matched against each
        car's "car_id" field), not positions in the "cars" list.
    Returns (sorted_timestamps, frames) where frames[timestamp] is always
    normalized to {"cars": [...], "distances": [...]}. Every car dict is
    guaranteed a "speed_kmh" key: taken from "obj_speed" if present and
    not NaN, otherwise calculated from how far that car_id moved since
    its previous reading (None if this is that car's first reading, so
    there's nothing to compare against).
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    # Restricted eval: only a "nan" name is exposed, no builtins at all,
    # since these files are plain Python literals (dict/list/float/str).
    raw = eval(content, {"__builtins__": {}}, {"nan": float("nan")})

    frames = {}
    for t, value in raw.items():
        if isinstance(value, dict):
            frames[t] = {"cars": value.get("cars", []), "distances": value.get("distance", [])}
        else:
            frames[t] = {"cars": value, "distances": []}

    # Group each car's readings chronologically so we can compute speed
    # from position deltas where "obj_speed" isn't supplied.
    by_car = {}
    for t, frame in frames.items():
        for c in frame["cars"]:
            by_car.setdefault(c.get("car_id"), []).append((t, c))

    for car_id, entries in by_car.items():
        entries.sort(key=lambda e: e[0])
        prev = None
        for t, c in entries:
            given = c.get("obj_speed")
            if given is not None and given == given:  # not NaN
                c["speed_kmh"] = float(given)
            elif prev is not None:
                t0, c0 = prev
                dt = t - t0
                x, y = c.get("point_x"), c.get("point_y")
                x0, y0 = c0.get("point_x"), c0.get("point_y")
                if dt > 0 and None not in (x, y, x0, y0) and not any(math.isnan(v) for v in (x, y, x0, y0)):
                    dist = math.hypot(x - x0, y - y0)
                    c["speed_kmh"] = (dist / dt) * 3.6
                else:
                    c["speed_kmh"] = None
            else:
                c["speed_kmh"] = None
            prev = (t, c)

    timestamps = sorted(frames.keys())
    return timestamps, frames


# ============================================================
# COLOR HELPERS (from 3243.py)
# ============================================================

def mix(a, b, t):
    return "#{:02x}{:02x}{:02x}".format(
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def cell_color(value, minimum, maximum, reverse=False):
    if pd.isna(value) or pd.isna(minimum) or pd.isna(maximum):
        return "#ffffff"
    if maximum == minimum:
        return "#fff2cc"
    t = max(0.0, min(1.0, (float(value) - float(minimum)) / (float(maximum) - float(minimum))))
    if reverse:
        t = 1.0 - t
    green, yellow, red = (112, 173, 71), (255, 235, 132), (248, 105, 107)
    if t <= 0.5:
        return mix(green, yellow, t * 2)
    return mix(yellow, red, (t - 0.5) * 2)


# ============================================================
# MAIN APP
# ============================================================

class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.reverse = tk.BooleanVar(value=False)
        self.color_mode = tk.StringVar(value="column")  # "table" | "column" | "row"
        self.current_file = tk.StringVar(value=FILE_NAMES[0] if FILE_NAMES else "")

        self.df_full = None   # everything loaded from disk
        self.df = None        # after series + row filtering
        self.series_vars = {} # series name -> BooleanVar (for the currently loaded file)
        self.hidden_series = set()  # series unchecked by the user, persists across files
        self.row_vars = {}    # row timestamp -> BooleanVar (for the currently loaded file)
        self.hidden_rows = set()  # row timestamps unchecked by the user, persists across files

        # Which dataset the shared Rows/Series tabs currently show: "table" or "dist"
        self.active_context = "table"

        # Distribution tab: its own separate file(s), rows and columns
        self.dist_source = tk.StringVar(value=DIST_FILE_NAMES[0] if DIST_FILE_NAMES else "")
        self.dist_df_full = None
        self.dist_row_vars = {}   # time -> BooleanVar
        self.dist_hidden_rows = set()
        self.dist_col_vars = {}   # lane number -> BooleanVar
        self.dist_hidden_cols = set()

        # Road tab: python-dict-literal scene files
        self.road_source = tk.StringVar(value=ROAD_FILE_NAMES[0] if ROAD_FILE_NAMES else "")
        self.road_timestamps = []
        self.road_frames = {}
        self.road_frame_index = 0
        self.road_playing = False
        self._suppress_road_slider_cb = False
        self.road_world_bounds = (0, 1, 0, 1)  # xmin, xmax, ymin, ymax
        self.road_lane_colors = {}
        self.road_lane_y = {}  # lane -> median point_y, for the guide lines

        root.title("Track Data Viewer")
        root.geometry("1300x800")
        root.minsize(900, 500)

        self._build_top_bar()
        self._build_notebook()

        if self.current_file.get():
            self.load_file(self.current_file.get())

    # ---------------- top bar ----------------

    def _build_top_bar(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="File:").pack(side="left")
        combo = ttk.Combobox(
            top, textvariable=self.current_file, values=FILE_NAMES,
            state="readonly", width=32,
        )
        combo.pack(side="left", padx=6)
        combo.bind("<<ComboboxSelected>>", lambda e: self.load_file(self.current_file.get()))

        ttk.Button(top, text="Reload", command=lambda: self.load_file(self.current_file.get())).pack(side="left", padx=6)

        self.rows_label = ttk.Label(top, text="Rows: 0")
        self.rows_label.pack(side="right")

    # ---------------- notebook / tabs ----------------

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text="Table")
        self._build_table_tab(self.table_tab)

        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="Chart")
        self._build_chart_tab(self.chart_tab)

        self.series_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.series_tab, text="Series")
        self._build_series_tab(self.series_tab)

        self.rows_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.rows_tab, text="Rows")
        self._build_rows_tab(self.rows_tab)

        self.dist_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dist_tab, text="Distribution")
        self._build_distribution_tab(self.dist_tab)

        self.road_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.road_tab, text="Road")
        self._build_road_tab(self.road_tab)

    # ---------------- table tab ----------------

    def _build_table_tab(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 7))

        ttk.Checkbutton(controls, text="Reverse colors", variable=self.reverse,
                        command=self.draw_table).pack(side="left")

        ttk.Label(controls, text="Color scale:").pack(side="left", padx=(20, 4))
        for value, label in (("table", "Whole table"), ("column", "Per column"), ("row", "Per row")):
            ttk.Radiobutton(controls, text=label, value=value, variable=self.color_mode,
                            command=self.draw_table).pack(side="left", padx=2)

        ttk.Label(controls, text="Low").pack(side="left", padx=(15, 4))
        self.legend = tk.Canvas(controls, width=180, height=16, highlightthickness=0)
        self.legend.pack(side="left")
        ttk.Label(controls, text="High").pack(side="left", padx=4)

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)

        xbar = ttk.Scrollbar(frame, orient="horizontal")
        xbar.pack(side="bottom", fill="x")
        ybar = ttk.Scrollbar(frame, orient="vertical")
        ybar.pack(side="right", fill="y")

        self.canvas = tk.Canvas(
            frame, bg="white", highlightthickness=1, highlightbackground="#c8c8c8",
            xscrollcommand=xbar.set, yscrollcommand=ybar.set,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        xbar.config(command=self.canvas.xview)
        ybar.config(command=self.canvas.yview)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120) * 3, "units"))

    def draw_legend(self):
        self.legend.delete("all")
        for x in range(180):
            self.legend.create_line(x, 0, x, 16, fill=cell_color(x / 179, 0, 1, self.reverse.get()))

    def draw_table(self):
        self.canvas.delete("all")
        self.draw_legend()
        if self.df is None:
            return

        df = self.df
        columns = list(df.columns)
        id_columns = [c for c in ("time", "name") if c in columns]
        value_columns = [c for c in columns if c not in id_columns]

        col_width = {"time": TIME_WIDTH, "name": NAME_WIDTH}
        total_w = sum(col_width.get(c, CELL_WIDTH) for c in columns)
        total_h = HEADER_HEIGHT + ROW_HEIGHT * len(df)
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))

        positions, x = {}, 0
        for c in columns:
            w = col_width.get(c, CELL_WIDTH)
            positions[c] = (x, x + w)
            x += w

        mode = self.color_mode.get()

        table_range = None
        column_ranges = None
        if mode == "table":
            all_vals = pd.concat([df[c] for c in value_columns], ignore_index=True).dropna() if value_columns else pd.Series(dtype=float)
            table_range = (all_vals.min(), all_vals.max()) if len(all_vals) else (None, None)
        elif mode == "column":
            column_ranges = {}
            for c in value_columns:
                vals = df[c].dropna()
                column_ranges[c] = (vals.min(), vals.max()) if len(vals) else (None, None)
        # mode == "row": range is computed per row, inside the loop below

        for c in columns:
            x1, x2 = positions[c]
            self.canvas.create_rectangle(x1, 0, x2, HEADER_HEIGHT, fill="#f2f2f2", outline="#d9d9d9")
            self.canvas.create_text((x1 + x2) / 2, HEADER_HEIGHT / 2, text=c, font=("Segoe UI", 10))

        for i, (_, row) in enumerate(df.iterrows()):
            y1 = HEADER_HEIGHT + i * ROW_HEIGHT
            y2 = y1 + ROW_HEIGHT

            row_range = None
            if mode == "row":
                row_vals = row[value_columns].dropna()
                row_range = (row_vals.min(), row_vals.max()) if len(row_vals) else (None, None)

            for c in columns:
                x1, x2 = positions[c]
                value = row[c]
                if c in id_columns:
                    bg = "#ffffff"
                elif mode == "table":
                    bg = cell_color(value, *table_range, self.reverse.get())
                elif mode == "row":
                    bg = cell_color(value, *row_range, self.reverse.get())
                else:  # column
                    bg = cell_color(value, *column_ranges[c], self.reverse.get())
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg, outline="#e0e0e0")
                if c == "time":
                    text = "" if pd.isna(value) else value.strftime("%d.%m.%Y %H:%M")
                elif c == "name":
                    text = "" if pd.isna(value) else str(value)
                else:
                    text = "-" if pd.isna(value) else f"{float(value):.2f}".replace(".", ",")
                self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, font=("Segoe UI", 9))

    # ---------------- chart tab ----------------

    def _build_chart_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        btns = ttk.Frame(parent)
        btns.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btns, text="Save graph", command=self.save_graph).pack(side="right")

    def draw_chart(self):
        self.ax.clear()
        if self.df is not None and len(self.df):
            df = self.df
            series = [c for c in df.columns if c not in ("time", "name")]
            for s in series:
                self.ax.plot(df["time"], df[s], marker="o", markersize=3,
                            linewidth=1.5, label=f"Series {s}")

            self.ax.set_title(f"Average distance — {self.current_file.get()}")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Value")
            self.ax.grid(True, alpha=0.25)
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for label in self.ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
            if self.ax.lines:
                self.ax.legend(loc="upper left")

        self.fig.tight_layout()
        self.chart_canvas.draw_idle()

    def save_graph(self):
        if not self.current_file.get():
            return
        output = (DATA_DIR / self.current_file.get()).with_suffix(".png")
        self.fig.savefig(output, dpi=200, bbox_inches="tight")
        messagebox.showinfo("Saved", f"Graph saved to:\n{output}")

    # ---------------- series tab ----------------

    def _build_series_tab(self, parent):
        self.series_desc_label = ttk.Label(parent, text="")
        self.series_desc_label.pack(anchor="w", padx=8, pady=(8, 4))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", padx=8)
        ttk.Button(btns, text="Select all", command=lambda: self.set_all_series_current(True)).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Deselect all", command=lambda: self.set_all_series_current(False)).pack(side="left")
        ttk.Button(btns, text="Apply", command=self.apply_current).pack(side="right")

        wrap = ttk.Frame(parent)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        series_canvas = tk.Canvas(wrap, highlightthickness=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=series_canvas.yview)
        self.series_list_frame = ttk.Frame(series_canvas)
        self.series_list_frame.bind(
            "<Configure>", lambda e: series_canvas.configure(scrollregion=series_canvas.bbox("all"))
        )
        series_canvas.create_window((0, 0), window=self.series_list_frame, anchor="nw")
        series_canvas.configure(yscrollcommand=scroll.set)
        series_canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def rebuild_series_checks(self):
        for child in self.series_list_frame.winfo_children():
            child.destroy()
        self.series_vars = {}
        if self.df_full is None:
            return
        columns = [c for c in self.df_full.columns if c not in ("time", "name")]
        for s in columns:
            # Same series hidden in a previous table? Keep it hidden here too.
            initial = s not in self.hidden_series
            var = tk.BooleanVar(value=initial)
            var.trace_add("write", lambda *args, s=s, var=var: self._on_series_toggle(s, var))
            self.series_vars[s] = var
            ttk.Checkbutton(self.series_list_frame, text=f"Series {s}", variable=var).pack(anchor="w")

    def _on_series_toggle(self, s, var):
        if var.get():
            self.hidden_series.discard(s)
        else:
            self.hidden_series.add(s)

    def set_all_series(self, value):
        for var in self.series_vars.values():
            var.set(value)  # triggers the trace above, which updates hidden_series too

    def apply_filters(self):
        if self.df_full is None:
            return
        active_series = {s for s, v in self.series_vars.items() if v.get()}
        active_rows = {t for t, v in self.row_vars.items() if v.get()}

        id_columns = [c for c in ("time", "name") if c in self.df_full.columns]
        keep_cols = id_columns + [c for c in self.df_full.columns if c not in id_columns and c in active_series]
        df = self.df_full[keep_cols]
        if active_rows:
            df = df[df["time"].isin(active_rows)]
        else:
            df = df.iloc[0:0]
        self.df = df.reset_index(drop=True)

        self.rows_label.config(text=f"Rows: {len(self.df)}")
        self.draw_table()
        self.draw_chart()

    # ---------------- rows tab ----------------

    def _build_rows_tab(self, parent):
        self.rows_desc_label = ttk.Label(parent, text="")
        self.rows_desc_label.pack(anchor="w", padx=8, pady=(8, 4))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", padx=8)
        ttk.Button(btns, text="Select all", command=lambda: self.set_all_rows_current(True)).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Deselect all", command=lambda: self.set_all_rows_current(False)).pack(side="left")
        ttk.Button(btns, text="Apply", command=self.apply_current).pack(side="right")

        wrap = ttk.Frame(parent)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        rows_canvas = tk.Canvas(wrap, highlightthickness=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=rows_canvas.yview)
        self.rows_list_frame = ttk.Frame(rows_canvas)
        self.rows_list_frame.bind(
            "<Configure>", lambda e: rows_canvas.configure(scrollregion=rows_canvas.bbox("all"))
        )
        rows_canvas.create_window((0, 0), window=self.rows_list_frame, anchor="nw")
        rows_canvas.configure(yscrollcommand=scroll.set)
        rows_canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def rebuild_row_checks(self):
        for child in self.rows_list_frame.winfo_children():
            child.destroy()
        self.row_vars = {}
        if self.df_full is None:
            return
        for t in self.df_full["time"]:
            # Same exact timestamp hidden in a previous table? Keep it hidden here too.
            initial = t not in self.hidden_rows
            var = tk.BooleanVar(value=initial)
            var.trace_add("write", lambda *args, t=t, var=var: self._on_row_toggle(t, var))
            self.row_vars[t] = var
            ttk.Checkbutton(self.rows_list_frame, text=t.strftime("%d.%m.%Y %H:%M"),
                            variable=var).pack(anchor="w")

    def _on_row_toggle(self, t, var):
        if var.get():
            self.hidden_rows.discard(t)
        else:
            self.hidden_rows.add(t)

    def set_all_rows(self, value):
        for var in self.row_vars.values():
            var.set(value)  # triggers the trace above, which updates hidden_rows too

    # ---------------- distribution tab ----------------
    # Uses its own separate files (DIST_FILE_NAMES) with a different shape:
    # time, a lane id ("seria"), then many unnamed raw sample columns.
    # Which times/lanes feed the histogram is chosen in the shared
    # Rows and Series tabs (they switch to show this file's data once
    # a Distribution source is picked).

    def _build_distribution_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Source:").pack(side="left")
        self.dist_combo = ttk.Combobox(top, textvariable=self.dist_source, values=DIST_FILE_NAMES,
                                       state="readonly", width=24)
        self.dist_combo.pack(side="left", padx=6)
        self.dist_combo.bind("<<ComboboxSelected>>", lambda e: self.load_dist_source(self.dist_source.get()))

        ttk.Label(parent, text="Pick the times and lanes to include in the Rows and Series tabs.",
                  foreground="#555555").pack(anchor="w", padx=8, pady=(0, 4))

        self.dist_fig = Figure(figsize=(10, 6))
        self.dist_ax = self.dist_fig.add_subplot(111)
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=parent)
        self.dist_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        if DIST_FILE_NAMES:
            self.load_dist_source(DIST_FILE_NAMES[0])

    def load_dist_source(self, name):
        if not name:
            return
        path = DIST_DATA_DIR / f"{name}.txt"
        try:
            self.dist_df_full = load_raw_samples(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load {path}:\n{e}")
            return

        self.active_context = "dist"
        self.refresh_shared_pickers()
        self.draw_distribution()

    def rebuild_dist_row_checks(self):
        for child in self.rows_list_frame.winfo_children():
            child.destroy()
        self.dist_row_vars = {}
        if self.dist_df_full is None:
            return
        for t in sorted(self.dist_df_full["time"].unique()):
            t = pd.Timestamp(t)
            initial = t not in self.dist_hidden_rows
            var = tk.BooleanVar(value=initial)
            var.trace_add("write", lambda *args, t=t, var=var: self._on_dist_row_toggle(t, var))
            self.dist_row_vars[t] = var
            ttk.Checkbutton(self.rows_list_frame, text=t.strftime("%d.%m.%Y %H:%M"),
                            variable=var).pack(anchor="w")

    def _on_dist_row_toggle(self, t, var):
        if var.get():
            self.dist_hidden_rows.discard(t)
        else:
            self.dist_hidden_rows.add(t)

    def set_all_dist_rows(self, value):
        for var in self.dist_row_vars.values():
            var.set(value)

    def rebuild_dist_col_checks(self):
        for child in self.series_list_frame.winfo_children():
            child.destroy()
        self.dist_col_vars = {}
        if self.dist_df_full is None:
            return
        lanes = sorted(self.dist_df_full["seria"].dropna().unique())
        for lane in lanes:
            lane = int(lane)
            initial = lane not in self.dist_hidden_cols
            var = tk.BooleanVar(value=initial)
            var.trace_add("write", lambda *args, lane=lane, var=var: self._on_dist_col_toggle(lane, var))
            self.dist_col_vars[lane] = var
            ttk.Checkbutton(self.series_list_frame, text=f"Lane {lane}", variable=var).pack(anchor="w")

    def _on_dist_col_toggle(self, c, var):
        if var.get():
            self.dist_hidden_cols.discard(c)
        else:
            self.dist_hidden_cols.add(c)

    def set_all_dist_cols(self, value):
        for var in self.dist_col_vars.values():
            var.set(value)

    # ---------------- shared Rows/Series tab dispatch ----------------
    # The Rows and Series tabs are shared: they show either the main
    # table's times/series or the current Distribution file's
    # times/lanes, depending on self.active_context.

    def refresh_shared_pickers(self):
        if self.active_context == "table":
            self.rebuild_row_checks()
            self.rebuild_series_checks()
            self.rows_desc_label.config(
                text="Uncheck a row to hide that exact reading from the table and chart:")
            self.series_desc_label.config(
                text="Uncheck a series to hide it from the table and chart:")
        else:
            self.rebuild_dist_row_checks()
            self.rebuild_dist_col_checks()
            self.rows_desc_label.config(
                text=f"Uncheck a time to leave it out of the {self.dist_source.get()} distribution:")
            self.series_desc_label.config(
                text=f"Uncheck a lane to leave it out of the {self.dist_source.get()} distribution:")

    def set_all_rows_current(self, value):
        if self.active_context == "table":
            self.set_all_rows(value)
        else:
            self.set_all_dist_rows(value)

    def set_all_series_current(self, value):
        if self.active_context == "table":
            self.set_all_series(value)
        else:
            self.set_all_dist_cols(value)

    def apply_current(self):
        if self.active_context == "table":
            self.apply_filters()
        else:
            self.draw_distribution()

    def draw_distribution(self):
        self.dist_ax.clear()

        if self.dist_df_full is not None:
            active_times = {t for t, v in self.dist_row_vars.items() if v.get()}
            active_lanes = {lane for lane, v in self.dist_col_vars.items() if v.get()}

            if active_times and active_lanes:
                mask = self.dist_df_full["time"].isin(active_times) & self.dist_df_full["seria"].isin(active_lanes)
                sample_cols = [c for c in self.dist_df_full.columns if c not in ("time", "seria")]
                subset = self.dist_df_full.loc[mask, sample_cols]
                vals = pd.Series(subset.values.flatten()).dropna().astype(float)
            else:
                vals = pd.Series([], dtype=float)

            if len(vals) >= 2:
                mean, std = vals.mean(), vals.std()
                bins = min(40, max(5, len(vals) // 20))
                self.dist_ax.hist(vals, bins=bins, density=True, alpha=0.55,
                                  color="#4a90d9", edgecolor="#2c5f8a", label="Data")

                if std > 0:
                    x = np.linspace(vals.min(), vals.max(), 200)
                    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
                    self.dist_ax.plot(x, y, color="#d9534f", linewidth=2, label="Normal curve fit")

                self.dist_ax.axvline(mean, color="#333333", linestyle="--", linewidth=1,
                                     label=f"Mean: {mean:.2f}")
                self.dist_ax.set_title(f"Distribution — {self.dist_source.get()}")
                self.dist_ax.set_xlabel("Value")
                self.dist_ax.set_ylabel("Density")
                self.dist_ax.legend(loc="upper right")
            else:
                self.dist_ax.set_title("Check some rows and lanes to see a distribution")

        self.dist_fig.tight_layout()
        self.dist_canvas.draw_idle()

    # ---------------- road tab ----------------
    # Shows one "snapshot" of the road at a time from a python-dict-literal
    # scene file: each car is drawn as a rectangle (color = lane), with its
    # speed in km/h inside it. Step through with the slider/buttons, or
    # press Play to watch it advance automatically.

    ROAD_CANVAS_W = 1000
    ROAD_CANVAS_H = 380
    ROAD_MARGIN = 40
    ROAD_PLAY_DELAY_MS = 150

    def _build_road_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Source:").pack(side="left")
        self.road_combo = ttk.Combobox(top, textvariable=self.road_source, values=ROAD_FILE_NAMES,
                                       state="readonly", width=24)
        self.road_combo.pack(side="left", padx=6)
        self.road_combo.bind("<<ComboboxSelected>>", lambda e: self.load_road_source(self.road_source.get()))

        ttk.Label(parent, text="Speed shown in km/h — calculated from movement when a reading has no obj_speed.",
                  foreground="#555555").pack(anchor="w", padx=8, pady=(0, 4))

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.road_canvas = tk.Canvas(body, bg="white", width=self.ROAD_CANVAS_W, height=self.ROAD_CANVAS_H,
                                     highlightthickness=1, highlightbackground="#c8c8c8")
        self.road_canvas.pack(side="left", fill="both", expand=True)

        side = ttk.LabelFrame(body, text="Distances", width=190)
        side.pack(side="left", fill="y", padx=(6, 0))
        side.pack_propagate(False)
        self.road_dist_list = tk.Listbox(side, font=("Segoe UI", 9))
        self.road_dist_list.pack(fill="both", expand=True, padx=4, pady=4)

        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=8, pady=(0, 8))

        self.road_play_btn = ttk.Button(controls, text="Play", command=self.toggle_road_play)
        self.road_play_btn.pack(side="left")
        ttk.Button(controls, text="\u25c0 Prev", command=lambda: self.set_road_frame(self.road_frame_index - 1)).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Next \u25b6", command=lambda: self.set_road_frame(self.road_frame_index + 1)).pack(side="left", padx=6)

        self.road_scale = ttk.Scale(controls, from_=0, to=0, orient="horizontal", command=self._on_road_slider)
        self.road_scale.pack(side="left", fill="x", expand=True, padx=6)

        self.road_frame_label = ttk.Label(controls, text="Frame 0/0", width=34, anchor="e")
        self.road_frame_label.pack(side="left", padx=(6, 0))

        if ROAD_FILE_NAMES:
            self.load_road_source(ROAD_FILE_NAMES[0])

    def load_road_source(self, name):
        if not name:
            return
        path = ROAD_DATA_DIR / f"{name}.txt"
        try:
            timestamps, frames = load_road_scene(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load {path}:\n{e}")
            return

        self.road_timestamps = timestamps
        self.road_frames = frames
        self.road_playing = False
        self.road_play_btn.config(text="Play")

        # Stable world bounds + lane colors/guide lines, computed once per
        # file so the view doesn't rescale or jump while stepping/playing.
        xs, ys = [], []
        lanes_seen = set()
        lane_ys = {}
        for frame in frames.values():
            for c in frame["cars"]:
                x, y, lane = c.get("point_x"), c.get("point_y"), c.get("lane")
                if x is not None and x == x:
                    xs.append(x)
                if y is not None and y == y:
                    ys.append(y)
                    if lane is not None and lane == lane:
                        lane_ys.setdefault(lane, []).append(y)
                if lane is not None and lane == lane:
                    lanes_seen.add(lane)

        xmin, xmax = (min(xs), max(xs)) if xs else (0, 1)
        ymin, ymax = (min(ys), max(ys)) if ys else (0, 1)
        if xmax - xmin < 1e-6:
            xmax = xmin + 1
        if ymax - ymin < 1e-6:
            ymax = ymin + 1
        pad_x = (xmax - xmin) * 0.03
        pad_y = (ymax - ymin) * 0.15
        self.road_world_bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

        sorted_lanes = sorted(lanes_seen)
        self.road_lane_colors = {
            lane: ROAD_LANE_COLORS[i % len(ROAD_LANE_COLORS)] for i, lane in enumerate(sorted_lanes)
        }
        self.road_lane_y = {lane: sum(vs) / len(vs) for lane, vs in lane_ys.items()}

        self.road_scale.configure(to=max(0, len(timestamps) - 1))
        self.set_road_frame(0)

    def _world_to_canvas(self, x, y):
        xmin, xmax, ymin, ymax = self.road_world_bounds
        w = self.ROAD_CANVAS_W - 2 * self.ROAD_MARGIN
        h = self.ROAD_CANVAS_H - 2 * self.ROAD_MARGIN
        sx = w / (xmax - xmin)
        sy = h / (ymax - ymin)
        px = self.ROAD_MARGIN + (x - xmin) * sx
        py = self.ROAD_MARGIN + (1 - (y - ymin) / (ymax - ymin)) * h
        return px, py, sx, sy

    def _on_road_slider(self, value):
        if self._suppress_road_slider_cb:
            return
        self.set_road_frame(int(round(float(value))), update_slider=False)

    def set_road_frame(self, index, update_slider=True):
        if not self.road_timestamps:
            return
        index = max(0, min(len(self.road_timestamps) - 1, index))
        self.road_frame_index = index

        if update_slider:
            self._suppress_road_slider_cb = True
            self.road_scale.set(index)
            self._suppress_road_slider_cb = False

        t = self.road_timestamps[index]
        readable = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.road_frame_label.config(text=f"Frame {index + 1}/{len(self.road_timestamps)} — {readable}")
        self.draw_road_frame()

    def toggle_road_play(self):
        self.road_playing = not self.road_playing
        self.road_play_btn.config(text="Pause" if self.road_playing else "Play")
        if self.road_playing:
            if self.road_frame_index >= len(self.road_timestamps) - 1:
                self.set_road_frame(0)
            self._road_play_step()

    def _road_play_step(self):
        if not self.road_playing:
            return
        if self.road_frame_index >= len(self.road_timestamps) - 1:
            self.road_playing = False
            self.road_play_btn.config(text="Play")
            return
        self.set_road_frame(self.road_frame_index + 1)
        self.root.after(self.ROAD_PLAY_DELAY_MS, self._road_play_step)

    def draw_road_frame(self):
        self.road_canvas.delete("all")
        self.road_dist_list.delete(0, "end")
        if not self.road_timestamps:
            return

        # faint guide line + label per lane, so lanes read as lanes
        for lane, y in self.road_lane_y.items():
            _, py, _, _ = self._world_to_canvas(0, y)
            color = self.road_lane_colors.get(lane, "#999999")
            self.road_canvas.create_line(self.ROAD_MARGIN, py, self.ROAD_CANVAS_W - self.ROAD_MARGIN, py,
                                         fill=color, dash=(3, 3))
            self.road_canvas.create_text(self.ROAD_MARGIN - 4, py, text=f"Lane {lane}",
                                         anchor="e", font=("Segoe UI", 8), fill=color)

        t = self.road_timestamps[self.road_frame_index]
        frame = self.road_frames.get(t, {"cars": [], "distances": []})
        cars = frame["cars"]

        positions = {}  # car_id -> (px, py)

        for idx, c in enumerate(cars):
            x, y = c.get("point_x"), c.get("point_y")
            if x is None or y is None or x != x or y != y:
                continue
            lane = c.get("lane")
            color = self.road_lane_colors.get(lane, "#666666")
            speed = c.get("speed_kmh")
            speed_text = f"{speed:.0f}" if speed is not None and speed == speed else "-"
            car_id = c.get("car_id")

            px, py, sx, sy = self._world_to_canvas(x, y)
            if car_id is not None:
                positions[car_id] = (px, py)
            length, width = c.get("obj_length"), c.get("obj_width")
            has_size = (length is not None and length == length and width is not None and width == width)

            if has_size:
                # Use one scale (sx) for both car dimensions so the rectangle
                # keeps its true real-world shape — the y-axis is deliberately
                # stretched (sy >> sx) just to make lane separation visible,
                # and would distort the car if used for its width too.
                hw = (length * sx) / 2  # obj_length runs along x (road direction)
                hh = (width * sx) / 2   # obj_width runs across the lane
                self.road_canvas.create_rectangle(px - hw, py - hh, px + hw, py + hh,
                                                  fill=color, outline="#222222")
            else:
                hw = hh = 5
                self.road_canvas.create_oval(px - hw, py - hh, px + hw, py + hh, fill=color, outline="#222222")

            if car_id is not None:
                self.road_canvas.create_text(px, py - hh - 8, text=f"#{car_id}",
                                             font=("Segoe UI", 8), fill="#222222")
            self.road_canvas.create_text(px, py, text=speed_text, font=("Segoe UI", 8, "bold"), fill="white")

        # distance lines between cars, and the matching side-panel list
        for d in frame["distances"]:
            ids = d.get("car_ids", [])
            length = d.get("length")
            if len(ids) != 2 or ids[0] not in positions or ids[1] not in positions:
                continue
            (x1, y1), (x2, y2) = positions[ids[0]], positions[ids[1]]
            self.road_canvas.create_line(x1, y1, x2, y2, fill="#555555", dash=(4, 2))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            length_text = f"{length:.1f}m" if length is not None and length == length else "-"
            self.road_canvas.create_text(mx, my - 6, text=length_text,
                                         font=("Segoe UI", 8), fill="#333333")

            self.road_dist_list.insert("end", f"{ids[0]} - {ids[1]} - {length_text}")

    # ---------------- file switching (main Table / Chart / Series / Rows tabs) ----------------

    def load_file(self, name):
        if not name:
            return
        path = DATA_DIR / f"{name}.txt"
        try:
            self.df_full = load_data(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load {path}:\n{e}")
            return

        self.active_context = "table"
        self.refresh_shared_pickers()
        self.apply_filters()


def main():
    if not FILE_NAMES:
        print("No file names configured. Edit FILE_NAMES at the top of this script.")
        return

    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    CombinedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
