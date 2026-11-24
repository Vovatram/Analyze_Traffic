# Semester 3 practice: traffic data analysis

Completed practice tasks using Python, pandas, and Jupyter Notebook. The notebook keeps the original Russian task statements and existing solutions, with clearer sections and portable file paths.

## Practice work

| Notebook | Tasks |
| --- | --- |
| [Practice 1: traffic analysis](notebooks/practice_01_traffic_analysis.ipynb) | Dataset inspection, outlier checks, traffic speed and V85 by lane, and additional vehicle distance calculations |

## Files

- `notebooks/`: organized practice notebooks.
- `all_tracks/`: supplied JSON recordings and previously generated text tables.
- `visualdt.py`: desktop viewer for the supplied tables, charts, distributions, and road tracks.
- `results/`: new exports created when running the notebook; excluded from Git.

## Run

Use Python 3.10 or newer. From the repository folder:

```sh
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependencies and open Jupyter:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `notebooks/practice_01_traffic_analysis.ipynb` and run its cells in order. The final cell processes all supplied JSON recordings and writes new tables to `results/`; this can take time. Notebook outputs are cleared so that saved results are not presented as results of the reorganized code.

To open the desktop viewer:

```sh
python visualdt.py
```

The viewer initially reads the existing files in `all_tracks/`. To inspect newly generated tables, set its `DATA_DIR` to the repository's `results/` folder. The viewer requires Tkinter, included with the standard Windows Python installer; some Linux installations require a separate system Tkinter package.

## Scope

The practice solution's numerical algorithms are preserved. Organizing the work does not establish that its formulas or results are correct. The original notebook, `Students_2026_01.ipynb`, is kept locally as an unchanged reference and is excluded from this repository's upload.
