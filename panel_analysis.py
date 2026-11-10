"""
panel_analysis.py
=================
Core processing for the solar-panel sweep measurements
(supercontinuum + AOTF + Keysight scope + 33611A generator).

This file has no user interface. app.py (the Streamlit app) imports it.
It can also be used on its own from Python, for example:

    import panel_analysis as pa
    runs = pa.load_zip(open("sweep_....zip", "rb").read(), "sweep_....zip")
    data = pa.merge_runs(runs)
    df = pa.add_derived(data)

Circuit:  generator (sine + DC) -> panel -> R_L -> ground
          ch1 = generator output (V1), ch2 = voltage on R_L (V2)
Transfer function:  H(w) = V2/V1 = R_L / (R_L + Z(w))
          =>  Z(w) = R_L * (V1/V2 - 1)   (phasors)
R_L here is the effective load: the resistor in parallel with the
scope input (1 Mohm), unless the user changes it.

Phase sign: the scope's ":MEASure:PHASe? CHANnel1,CHANnel2" returns
phi1 - phi2. This was verified from the data (with the opposite sign
the fitted capacitance comes out negative). So the physical phase of
H = V2/V1 is  phase_H = -phase_deg.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
R_LOAD_DEFAULT = 1000.0          # ohm
SCOPE_RIN_DEFAULT = 1.0e6        # ohm (scope input, DC coupling, 1 Mohm)
RELIABLE_CH2_MIN_V = 0.010       # ch2 amplitude below this = low reliability
FREQ_TOL = 0.02                  # |f_measured/f_set - 1| above this = bad
NEGLIGIBLE_SECTION_OHM = 50.0    # an R‖C section with R below this is "not needed"

NUMERIC_COLUMNS = [
    "wavelength_nm", "aotf_power_percent", "offset_set_V", "frequency_set_Hz",
    "ch1_amplitude_V", "ch1_vpp_V", "ch2_amplitude_V", "ch2_vpp_V",
    "ch1_dc_V", "ch2_dc_V", "ch1_frequency_Hz", "ch2_frequency_Hz",
    "phase_deg", "ch2_scale_V_per_div", "time_s",
]


# ----------------------------------------------------------------------
# 1. Loading ZIP files and parsing settings.txt
# ----------------------------------------------------------------------
@dataclass
class Run:
    name: str                      # folder / zip name
    settings_text: str             # raw settings.txt
    settings: dict                 # key -> value (strings)
    parsed: dict                   # numbers extracted from settings
    stop_message: str | None       # "RUN STOPPED: ..." or None
    df: pd.DataFrame               # raw sweep.csv
    start: datetime | None = None  # run start time
    notes: list = field(default_factory=list)


def parse_settings(text: str) -> tuple[dict, dict, str | None]:
    """Read settings.txt.

    Returns (settings, parsed, stop_message):
      settings  - every "key: value" line, as strings, in file order
      parsed    - numbers pulled out of the known lines
      stop      - the RUN STOPPED line, or None
    """
    settings: dict[str, str] = {}
    stop = None
    extra = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("RUN STOPPED"):
            stop = line
        if ":" in line:
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if key in settings:          # keep duplicates
                key = f"{key} ({len(settings)})"
            settings[key] = value
        else:
            extra += 1
            settings[f"line {extra}"] = line

    parsed: dict = {}
    s = {k.lower(): v for k, v in settings.items()}

    if "date" in s:
        try:
            parsed["start"] = datetime.strptime(s["date"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    m = re.search(r"([\d.]+)\s*-\s*([\d.]+)\s*nm.*?step\s*([\d.]+)",
                  s.get("wavelengths", ""))
    if m:
        parsed["wl_start"], parsed["wl_stop"], parsed["wl_step"] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)))

    m = re.search(r"^.*\bsine\b.*$", text, re.I | re.M)   # e.g. "sine 2 Vpp, DC offsets [...] V"
    sine = m.group(0) if m else s.get("sine", "")
    m = re.search(r"([\d.]+)\s*Vpp", sine)
    if m:
        parsed["sine_vpp"] = float(m.group(1))
    m = re.search(r"\[([^\]]*)\]", sine)
    if m:
        try:
            parsed["offsets_V"] = [float(x) for x in m.group(1).split(",") if x.strip()]
        except ValueError:
            pass

    m = re.search(r"averaging:\s*(\d+)", text)
    if m:
        parsed["averaging"] = int(m.group(1))
    m = re.search(r"(\d+)\s*cycles", text)
    if m:
        parsed["cycles_on_screen"] = int(m.group(1))

    m = re.search(r"(\d+)\s*points,\s*([\deE.+-]+)\s*-\s*([\deE.+-]+)\s*Hz",
                  s.get("frequencies", ""))
    if m:
        parsed["n_freq"] = int(m.group(1))
        parsed["f_min"] = float(m.group(2))
        parsed["f_max"] = float(m.group(3))

    m = re.search(r"Power\s*([\d.]+)\s*%", s.get("laser", ""))
    if m:
        parsed["laser_power_pct"] = float(m.group(1))
    m = re.search(r"Repetition rate\s*([\d.]+)\s*MHz", s.get("laser", ""))
    if m:
        parsed["rep_rate_MHz"] = float(m.group(1))

    m = re.search(r"([\d.]+)\s*kohm", s.get("circuit", ""), re.I)
    if m:
        parsed["R_load_ohm"] = float(m.group(1)) * 1e3

    if stop:
        m = re.search(r"at\s*([\d.]+)\s*nm", stop)
        if m:
            parsed["stopped_at_nm"] = float(m.group(1))
    return settings, parsed, stop


def _name_start(name: str) -> datetime | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", name)
    if m:
        try:
            return datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H-%M-%S")
        except ValueError:
            return None
    return None


def load_zip(data: bytes, zip_name: str = "upload.zip") -> list[Run]:
    """Read every sweep.csv (+ settings.txt next to it) inside a ZIP.

    Nothing is extracted to disk; the files are read in memory.
    A ZIP usually holds one run, but several folders are also handled.
    """
    runs: list[Run] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        csvs = [n for n in names if n.lower().endswith("sweep.csv")]
        if not csvs:
            csvs = [n for n in names if n.lower().endswith(".csv")]
        for csv_name in csvs:
            folder = csv_name.rsplit("/", 1)[0] if "/" in csv_name else ""
            settings_name = (folder + "/settings.txt") if folder else "settings.txt"
            if settings_name not in names:
                cands = [n for n in names if n.lower().endswith("settings.txt")]
                settings_name = cands[0] if len(cands) == 1 else None
            text = ""
            if settings_name:
                text = zf.read(settings_name).decode("utf-8", errors="replace")
            df = pd.read_csv(io.BytesIO(zf.read(csv_name)))
            df.columns = [c.strip() for c in df.columns]
            for c in NUMERIC_COLUMNS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "status" in df.columns:
                df["status"] = df["status"].fillna("").astype(str).str.strip()
            else:
                df["status"] = "ok"
            settings, parsed, stop = parse_settings(text)
            run_name = folder.split("/")[-1] if folder else zip_name.rsplit(".", 1)[0]
            start = parsed.get("start") or _name_start(run_name) or _name_start(zip_name)
            run = Run(run_name, text, settings, parsed, stop, df, start)
            if not text:
                run.notes.append("settings.txt not found in the ZIP")
            runs.append(run)
    if not runs:
        raise ValueError(f"No CSV file found in {zip_name}")
    return runs


# ----------------------------------------------------------------------
# 2. Merging runs
# ----------------------------------------------------------------------
def merge_runs(runs: list[Run]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine several runs into one table.

    For each wavelength the data come from ONE run:
      - among runs where that wavelength is complete, the LATEST run wins
        (so the morning re-run replaces the night run at 845-850 nm);
      - if no run has it complete, the run with the most points is used.
    Returns (data, source_table).
    """
    ordered = sorted(runs, key=lambda r: (r.start or datetime.min))
    frames = []
    for i, r in enumerate(ordered):
        d = r.df.copy()
        d["run"] = r.name
        d["run_order"] = i
        if r.start is not None and "time_s" in d:
            d["timestamp"] = pd.Timestamp(r.start) + pd.to_timedelta(d["time_s"], unit="s")
        else:
            d["timestamp"] = pd.NaT
        frames.append(d)
    all_data = pd.concat(frames, ignore_index=True)

    counts = (all_data.groupby(["wavelength_nm", "run_order", "run"])
              .size().rename("n_points").reset_index())
    full = all_data.groupby("run_order").apply(
        lambda g: g["offset_set_V"].nunique() * g["frequency_set_Hz"].round(6).nunique(),
        include_groups=False).rename("expected")
    counts = counts.merge(full, left_on="run_order", right_index=True)
    counts["complete"] = counts["n_points"] >= counts["expected"]

    chosen = []
    for wl, g in counts.groupby("wavelength_nm"):
        comp = g[g["complete"]]
        pick = comp.sort_values("run_order").iloc[-1] if len(comp) else \
            g.sort_values(["n_points", "run_order"]).iloc[-1]
        chosen.append(pick)
    chosen = pd.DataFrame(chosen)

    # source table: one row per wavelength, which run was used + alternatives
    src_rows = []
    for _, p in chosen.iterrows():
        others = counts[(counts.wavelength_nm == p.wavelength_nm) &
                        (counts.run_order != p.run_order)]
        src_rows.append({
            "wavelength_nm": p.wavelength_nm,
            "used_run": p.run,
            "points": int(p.n_points),
            "complete": bool(p.complete),
            "also_in": ", ".join(f"{o.run} ({int(o.n_points)} pts)" for o in others.itertuples()),
        })
    source = pd.DataFrame(src_rows).sort_values("wavelength_nm").reset_index(drop=True)

    key = set(zip(chosen.wavelength_nm, chosen.run_order))
    mask = [(w, o) in key for w, o in zip(all_data.wavelength_nm, all_data.run_order)]
    data = all_data[mask].reset_index(drop=True)
    return data, source


# ----------------------------------------------------------------------
# 3. Derived quantities
# ----------------------------------------------------------------------
def effective_load(r_load=R_LOAD_DEFAULT, scope_rin=SCOPE_RIN_DEFAULT) -> float:
    if not scope_rin or scope_rin <= 0:
        return float(r_load)
    return float(r_load * scope_rin / (r_load + scope_rin))


def add_derived(df: pd.DataFrame, r_load=R_LOAD_DEFAULT, scope_rin=SCOPE_RIN_DEFAULT,
                reliable_min_v=RELIABLE_CH2_MIN_V, physical_phase=True) -> pd.DataFrame:
    """Add gain, phase, current, H, Z and quality flags."""
    d = df.copy()
    rl = effective_load(r_load, scope_rin)
    d["gain"] = d["ch2_amplitude_V"] / d["ch1_amplitude_V"]
    d["gain_dB"] = 20 * np.log10(d["gain"])
    # phase of H = V2/V1 (physical convention) or raw scope reading
    d["phase_H_deg"] = -d["phase_deg"] if physical_phase else d["phase_deg"]
    d["I_dc_uA"] = d["ch2_dc_V"] / rl * 1e6
    d["V_panel_dc_V"] = d["ch1_dc_V"] - d["ch2_dc_V"]

    ph_true = np.deg2rad(-d["phase_deg"])     # always the physical sign for H and Z
    H = d["gain"].to_numpy() * np.exp(1j * ph_true.to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = rl * (1.0 / H - 1.0)
    d["H_re"], d["H_im"] = H.real, H.imag
    d["Z_re_ohm"], d["Z_im_ohm"] = Z.real, Z.imag
    d["Z_abs_ohm"] = np.abs(Z)
    d["Z_phase_deg"] = np.degrees(np.angle(Z))

    f = d["frequency_set_Hz"]
    d["f1_err"] = (d["ch1_frequency_Hz"] / f - 1).abs()
    d["f2_err"] = (d["ch2_frequency_Hz"] / f - 1).abs()
    meas_cols = ["ch1_amplitude_V", "ch2_amplitude_V", "phase_deg", "ch2_dc_V"]
    d["valid"] = (d["status"].str.lower() == "ok") & d[meas_cols].notna().all(axis=1) \
        & (d["f1_err"].fillna(1) <= FREQ_TOL) & (d["f2_err"].fillna(1) <= FREQ_TOL)
    d["reliable"] = d["valid"] & (d["ch2_amplitude_V"] >= reliable_min_v)
    d.attrs["R_load_eff"] = rl
    return d


# ----------------------------------------------------------------------
# 4. Quality / coverage report
# ----------------------------------------------------------------------
def coverage_report(runs: list[Run], d: pd.DataFrame, source: pd.DataFrame) -> dict:
    """Numbers for the "experiment & quality" tab."""
    planned = set()
    for r in runs:
        p = r.parsed
        if {"wl_start", "wl_stop", "wl_step"} <= p.keys():
            planned |= set(np.round(np.arange(p["wl_start"], p["wl_stop"] + p["wl_step"] / 2,
                                              p["wl_step"]), 3))
    measured = set(np.round(d["wavelength_nm"].unique(), 3))
    missing = sorted(planned - measured)
    incomplete = source[~source["complete"]]["wavelength_nm"].tolist()

    per_run = []
    for r in runs:
        g = r.df
        wls = g["wavelength_nm"].dropna()
        dur = g.groupby("wavelength_nm")["time_s"].agg(lambda t: (t.max() - t.min()) / 60)
        per_run.append({
            "run": r.name,
            "start": r.start.strftime("%Y-%m-%d %H:%M:%S") if r.start else "",
            "end (approx.)": (r.start + pd.to_timedelta(g["time_s"].max(), unit="s")).strftime(
                "%Y-%m-%d %H:%M:%S") if r.start is not None and g["time_s"].notna().any() else "",
            "wavelengths": f"{wls.min():g}-{wls.max():g} nm ({wls.nunique()})" if len(wls) else "-",
            "rows": len(g),
            "status": r.stop_message or "no stop message",
            "min/wavelength": f"{dur.median():.2f}" if len(dur) else "",
        })

    status_counts = d["status"].value_counts(dropna=False).to_dict()
    return {
        "per_run": pd.DataFrame(per_run),
        "planned_n": len(planned),
        "measured_n": len(measured),
        "missing": [float(x) for x in missing],
        "incomplete": [float(x) for x in incomplete],
        "rows": len(d),
        "status_counts": status_counts,
        "n_nan": int(d[["ch1_amplitude_V", "ch2_amplitude_V", "phase_deg"]].isna().any(axis=1).sum()),
        "n_freq_bad": int(((d["f1_err"] > FREQ_TOL) | (d["f2_err"] > FREQ_TOL)).sum()),
        "n_valid": int(d["valid"].sum()),
        "n_reliable": int(d["reliable"].sum()),
        "offsets": sorted(d["offset_set_V"].unique().tolist()),
        "n_freq": int(d["frequency_set_Hz"].round(6).nunique()),
        "f_range": (float(d["frequency_set_Hz"].min()), float(d["frequency_set_Hz"].max())),
        "wl_range": (float(d["wavelength_nm"].min()), float(d["wavelength_nm"].max())),
    }


# ----------------------------------------------------------------------
# 5. Phase differences between wavelengths (item 4)
# ----------------------------------------------------------------------
RANK_METRICS = {
    "range_reliable": "Range (max-min), reliable states only",
    "range": "Range (max-min), all states",
    "snr": "Significance: spread / noise",
    "trend": "Systematic trend (R² of smooth fit)",
}


def phase_spread_table(d: pd.DataFrame, value_col="phase_H_deg") -> pd.DataFrame:
    """One row per measurement state (offset, frequency).

    range      max - min of the phase over wavelengths
    std        standard deviation over wavelengths
    noise      point-to-point noise, std(diff)/sqrt(2). A smooth trend
               along the wavelength barely changes it, random scatter does.
    snr        std / noise. ~1 = the differences look like random scatter;
               clearly > 1 = a systematic change with wavelength.
    trend_r2   fraction of the variance explained by a smooth (cubic) curve
    slope      linear slope, degrees per 100 nm
    """
    rows = []
    g_all = d[d["valid"]]
    for (off, f), g in g_all.groupby(["offset_set_V", "frequency_set_Hz"]):
        g = g.sort_values("wavelength_nm")
        p = g[value_col].to_numpy(float)
        lam = g["wavelength_nm"].to_numpy(float)
        n = len(p)
        if n < 3:
            continue
        rng = float(p.max() - p.min())
        sd = float(p.std(ddof=1))
        noise = float(np.std(np.diff(p), ddof=1) / np.sqrt(2)) if n > 3 else np.nan
        x = (lam - lam.mean()) / 100.0
        deg = 3 if n >= 8 else 1
        coef = np.polyfit(x, p, deg)
        resid = p - np.polyval(coef, x)
        r2 = float(1 - resid.var() / p.var()) if p.var() > 0 else 0.0
        slope = float(np.polyfit(x, p, 1)[0])
        rows.append({
            "offset_V": off, "frequency_Hz": f, "n_wl": n,
            "mean_phase_deg": float(p.mean()),
            "range_deg": rng, "std_deg": sd, "noise_deg": noise,
            "snr": sd / noise if noise and noise > 0 else np.nan,
            "trend_r2": r2, "slope_deg_per_100nm": slope,
            "wl_at_max": float(lam[p.argmax()]), "wl_at_min": float(lam[p.argmin()]),
            "reliable_frac": float(g["reliable"].mean()),
            "ch2_amp_min_mV": float(g["ch2_amplitude_V"].min() * 1e3),
        })
    t = pd.DataFrame(rows)
    if len(t):
        t["all_reliable"] = t["reliable_frac"] >= 0.999
    return t


def rank_states(spread: pd.DataFrame, metric: str = "range_reliable") -> pd.DataFrame:
    t = spread.copy()
    if metric == "range_reliable":
        t = t[t["all_reliable"]].sort_values("range_deg", ascending=False)
    elif metric == "range":
        t = t.sort_values("range_deg", ascending=False)
    elif metric == "snr":
        t = t.sort_values("snr", ascending=False)
    elif metric == "trend":
        t = t[t["all_reliable"]].sort_values(["trend_r2", "range_deg"], ascending=False)
    return t.reset_index(drop=True)


def _freq_label(f: float) -> str:
    if f >= 1e6:
        return f"{f/1e6:.3g} MHz"
    if f >= 1e3:
        return f"{f/1e3:.3g} kHz"
    return f"{f:.3g} Hz"


def fmt_v(v: float) -> str:
    """'-9 V', '+4.5 V', '0 V'."""
    return "0 V" if v == 0 else f"{v:+g} V"


def _band(f: float) -> str:
    if f < 200:
        return "low"
    if f < 20e3:
        return "mid"
    return "high"


def spread_analysis_text(spread: pd.DataFrame, d: pd.DataFrame, metric: str, top_n: int = 5) -> str:
    """Hebrew text (markdown) describing the phase differences between wavelengths."""
    if spread.empty:
        return "אין מספיק נתונים לניתוח."
    ranked = rank_states(spread, metric)
    if ranked.empty:
        return "אין מצבי מדידה שעומדים בתנאי הדירוג שנבחר."
    top = ranked.head(top_n)
    n_wl = int(spread["n_wl"].max())
    lines = []
    lines.append(f"**נבדקו {len(spread)} מצבי מדידה** (צירופים של מתח DC ותדר), "
                 f"בכל אחד עד {n_wl} אורכי גל. שיטת הדירוג: **{RANK_METRICS.get(metric, metric)}**.")
    lines.append("")
    lines.append(f"**{len(top)} המצבים עם ההבדלים הגדולים ביותר:**")
    for i, r in enumerate(top.itertuples(), 1):
        verdict = "נראה כמו פיזור אקראי" if (np.isnan(r.snr) or r.snr < 1.3) else \
            ("שינוי שיטתי עם אורך הגל" if r.trend_r2 > 0.5 else "חלקית שיטתי")
        rel = "" if r.all_reliable else " ⚠️ כולל נקודות עם אות חלש בערוץ 2"
        lines.append(
            f"{i}. **{fmt_v(r.offset_V)}, {_freq_label(r.frequency_Hz)}**: טווח {r.range_deg:.2f}°, "
            f"סטיית תקן {r.std_deg:.2f}°, רעש נקודתי {r.noise_deg:.2f}° "
            f"(יחס {r.snr:.2f}), R² של מגמה {r.trend_r2:.2f}, "
            f"מקסימום ב-{r.wl_at_max:g} nm ומינימום ב-{r.wl_at_min:g} nm — {verdict}.{rel}")
    lines.append("")

    # overall picture
    rel = spread[spread["all_reliable"]]
    med_snr = float(np.nanmedian(spread["snr"]))
    med_rng = float(rel["range_deg"].median()) if len(rel) else float(spread["range_deg"].median())
    lines.append("**תמונה כללית:**")
    lines.append(f"- הטווח החציוני בין אורכי הגל (במצבים אמינים) הוא {med_rng:.2f}°, "
                 f"ויחס פיזור/רעש חציוני {med_snr:.2f}.")
    if med_snr < 1.3:
        lines.append("- ברוב המצבים ההבדלים בין אורכי הגל **בסדר גודל של הרעש הנקודתי** — כלומר "
                     "אין עדות ברורה לתלות של הפאזה באורך הגל. ההבדלים הגדולים נובעים בעיקר "
                     "מפיזור אקראי של נקודות בודדות.")
    else:
        lines.append("- בחלק ניכר מהמצבים הפיזור גדול מהרעש הנקודתי — יש תלות שיטתית באורך הגל.")

    sig = spread[(spread["snr"] >= 1.5) & (spread["trend_r2"] >= 0.5)]
    if len(sig):
        desc = ", ".join(f"{fmt_v(r.offset_V)}/{_freq_label(r.frequency_Hz)} "
                         f"({r.slope_deg_per_100nm:+.2f}°/100 nm)"
                         for r in sig.sort_values("snr", ascending=False).head(6).itertuples())
        lines.append(f"- מצבים שבהם יש מגמה עקבית ומעל הרעש (יחס ≥ 1.5, R² ≥ 0.5): {desc}.")
    else:
        lines.append("- לא נמצא מצב שבו יש מגמה עקבית עם אורך הגל שבולטת מעל הרעש "
                     "(יחס ≥ 1.5 ו-R² ≥ 0.5).")

    # where are the big ones
    bands = top["frequency_Hz"].map(_band).value_counts()
    if len(bands):
        b = bands.index[0]
        name = {"low": "בתדרים הנמוכים (מתחת ל-200 Hz)", "mid": "בתדרים הבינוניים (200 Hz–20 kHz)",
                "high": "בתדרים הגבוהים (מעל 20 kHz)"}[b]
        lines.append(f"- רוב המצבים בראש הדירוג נמצאים {name}.")
    unrel = spread[~spread["all_reliable"]]
    if len(unrel):
        lines.append(f"- {len(unrel)} מצבים כוללים נקודות שבהן אמפליטודת ערוץ 2 קטנה מסף האמינות "
                     f"(בעיקר תדרים נמוכים במתח 0 ובמתח אחורי). שם נמצא שהאמפליטודה תלויה "
                     f"בסקאלת הסקופ, ולכן הפאזה פחות אמינה.")

    lines.append("")
    lines.append("**הסתייגויות לפרשנות:**")
    if d["run"].nunique() == 1:
        lines.append("- בריצה אחת אורכי הגל נמדדו לפי הסדר, ולכן **אורך גל וזמן מתואמים לחלוטין**. "
                     "מגמה עם אורך הגל יכולה לנבוע גם מסחיפה בזמן (טמפרטורה, לייזר).")
    else:
        lines.append("- אורכי הגל נמדדו לפי הסדר בכל ריצה, כך שאורך גל וזמן מתואמים. "
                     "קפיצה בגבול בין ריצות יכולה לנבוע מהבדל בין הריצות ולא מאורך הגל.")
    lines.append("- עוצמת ה-AOTF כוילה כך שזרם הפוטו יהיה זהה (1.2 µA) בכל אורכי הגל. "
                 "לכן מצופה שההשפעה החשמלית של האור תהיה דומה בכל אורכי הגל; "
                 "הבדל אמיתי בפאזה יעיד על השפעה שתלויה באורך הגל מעבר לזרם הפוטו.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 6. Derivative of phase with respect to wavelength (item 5)
# ----------------------------------------------------------------------
def local_derivative(x: np.ndarray, y: np.ndarray, window_nm: float) -> np.ndarray:
    """dy/dx by a local quadratic fit in a window of +-window_nm/2 around each point.

    Works also when the wavelength spacing is not uniform (gaps).
    window_nm <= 0 gives a plain numerical gradient (no smoothing).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    out = np.full_like(y, np.nan)
    if len(x) < 2:
        return out
    if window_nm <= 0:
        out = np.gradient(y, x)
    else:
        half = window_nm / 2.0
        for i, x0 in enumerate(x):
            m = np.abs(x - x0) <= half + 1e-9
            if m.sum() >= 3:
                c = np.polyfit(x[m] - x0, y[m], 2 if m.sum() >= 4 else 1)
                out[i] = c[-2]          # derivative at x0
            elif m.sum() == 2:
                xx, yy = x[m], y[m]
                out[i] = (yy[1] - yy[0]) / (xx[1] - xx[0])
    res = np.empty_like(out)
    res[order] = out
    return res


# ----------------------------------------------------------------------
# 7. Impedance models and fitting (item 6)
# ----------------------------------------------------------------------
# Each model: parameter names, units, which are positive (fitted in log),
# Z(params, w), and bounds for linear parameters.
def _z_rc(p, w):
    Rp, C = p
    return 1.0 / (1.0 / Rp + 1j * w * C)


def _z_rs_rc(p, w):
    Rp, C, Rs = p
    return Rs + 1.0 / (1.0 / Rp + 1j * w * C)


def _z_rs_cpe(p, w):
    Rp, Q, n, Rs = p
    return Rs + 1.0 / (1.0 / Rp + Q * (1j * w) ** n)


def _z_rs_2rc(p, w):
    R1, C1, R2, C2, Rs = p
    return Rs + 1.0 / (1.0 / R1 + 1j * w * C1) + 1.0 / (1.0 / R2 + 1j * w * C2)


def _z_rs_ls_rc(p, w):
    Rp, C, Rs, Ls = p
    return Rs + 1j * w * Ls + 1.0 / (1.0 / Rp + 1j * w * C)


def _z_rs_ls_2rc(p, w):
    R1, C1, R2, C2, Rs, Ls = p
    return Rs + 1j * w * Ls + 1.0 / (1.0 / R1 + 1j * w * C1) + 1.0 / (1.0 / R2 + 1j * w * C2)


MODELS = {
    "Rs + R1‖C1 + R2‖C2": {
        "params": ["R1", "C1", "R2", "C2", "Rs"],
        "units": ["ohm", "F", "ohm", "F", "ohm"],
        "log": [True, True, True, True, True],
        "func": _z_rs_2rc,
        "desc": "Two parallel R‖C sections in series + series resistance. "
                "Best for reverse bias and 0 V (two time constants).",
    },
    "Rs + Ls + R1‖C1 + R2‖C2": {
        "params": ["R1", "C1", "R2", "C2", "Rs", "Ls"],
        "units": ["ohm", "F", "ohm", "F", "ohm", "H"],
        "log": [True, True, True, True, True, True],
        "func": _z_rs_ls_2rc,
        "desc": "Same as the two-section model + series inductance of the wiring "
                "(visible above ~300 kHz).",
    },
    "Rs + R‖CPE": {
        "params": ["Rp", "Q", "n", "Rs"],
        "units": ["ohm", "F·s^(n-1)", "", "ohm"],
        "log": [True, True, False, True],
        "func": _z_rs_cpe,
        "desc": "R in parallel with a constant-phase element (n=1 is an ideal capacitor) + Rs.",
    },
    "Rs + R‖C": {
        "params": ["Rp", "C", "Rs"],
        "units": ["ohm", "F", "ohm"],
        "log": [True, True, True],
        "func": _z_rs_rc,
        "desc": "Standard single-diode small-signal model: R‖C + series resistance.",
    },
    "Rs + Ls + R‖C": {
        "params": ["Rp", "C", "Rs", "Ls"],
        "units": ["ohm", "F", "ohm", "H"],
        "log": [True, True, True, True],
        "func": _z_rs_ls_rc,
        "desc": "R‖C + series resistance + series inductance of the wiring.",
    },
    "R‖C": {
        "params": ["Rp", "C"],
        "units": ["ohm", "F"],
        "log": [True, True],
        "func": _z_rc,
        "desc": "Simplest model: resistance in parallel with capacitance.",
    },
}


def _initial_guess(model: str, w: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Rough starting values from the measured Z."""
    Y = 1.0 / Z
    with np.errstate(invalid="ignore", divide="ignore"):
        c_est = np.nanmedian(np.where(Y.imag > 0, Y.imag / w, np.nan))
    if not np.isfinite(c_est) or c_est <= 0:
        c_est = 1e-8
    rp = 1.0 / max(Y.real[np.argmin(w)], 1e-9)
    rp = float(np.clip(rp, 10, 1e9))
    rs = float(np.clip(Z.real[np.argmax(w)], 1.0, 1e3))
    if model == "R‖C":
        return np.array([rp, c_est])
    if model == "Rs + R‖C":
        return np.array([rp, c_est, rs])
    if model == "Rs + R‖CPE":
        return np.array([rp, c_est, 0.95, rs])
    if model == "Rs + Ls + R1‖C1 + R2‖C2":
        return np.array([rp, c_est, rp / 20, c_est * 2, rs, 1e-6])
    if model == "Rs + Ls + R‖C":
        return np.array([rp, c_est, rs, 1e-6])
    return np.array([rp, c_est, rp / 20, c_est * 2, rs])


def _to_q(model, p):
    spec = MODELS[model]
    return np.array([np.log(v) if lg else v for v, lg in zip(p, spec["log"])], float)


def _from_q(model, q):
    spec = MODELS[model]
    return np.array([np.exp(v) if lg else v for v, lg in zip(q, spec["log"])], float)


def model_H(model: str, params, f_hz, r_load_eff: float) -> np.ndarray:
    w = 2 * np.pi * np.asarray(f_hz, float)
    Z = MODELS[model]["func"](params, w)
    return r_load_eff / (r_load_eff + Z)


def model_Z(model: str, params, f_hz) -> np.ndarray:
    w = 2 * np.pi * np.asarray(f_hz, float)
    return MODELS[model]["func"](params, w)


def fit_state(g: pd.DataFrame, model: str, r_load_eff: float, reliable_only=True,
              start: np.ndarray | None = None, n_starts: int = 6, seed: int = 0) -> dict:
    """Fit one measurement state (one wavelength, one DC offset) over frequency.

    Least squares on log(H_model/H_measured): the real part is the relative
    gain error, the imaginary part is the phase error (radians). Both are
    weighted equally (1% gain error ~ 0.57 deg phase error).
    """
    g = g.sort_values("frequency_set_Hz")
    g_all = g
    use = g["reliable"] if reliable_only else g["valid"]
    g = g[use]
    spec = MODELS[model]
    k = len(spec["params"])
    result = {"model": model, "n_points": len(g), "success": False}
    if len(g) < k + 2:
        return result
    f = g["frequency_set_Hz"].to_numpy(float)
    w = 2 * np.pi * f
    H = g["H_re"].to_numpy() + 1j * g["H_im"].to_numpy()
    Zm = g["Z_re_ohm"].to_numpy() + 1j * g["Z_im_ohm"].to_numpy()

    def resid(q):
        p = _from_q(model, q)
        with np.errstate(all="ignore"):
            Hm = r_load_eff / (r_load_eff + spec["func"](p, w))
            r = np.log(Hm / H)
        r = np.r_[r.real, r.imag]
        return np.where(np.isfinite(r), r, 1e3)

    lb = np.array([np.log(1e-3) if lg else 0.3 for lg in spec["log"]])
    ub = np.array([np.log(1e12) if lg else 1.0 for lg in spec["log"]])
    # capacitances / Q: tighter range
    for i, u in enumerate(spec["units"]):
        if u.startswith("F"):
            lb[i], ub[i] = np.log(1e-15), np.log(1e-2)
        if u == "H":
            lb[i], ub[i] = np.log(1e-10), np.log(1e-2)

    starts = []
    if start is not None:
        starts.append(_to_q(model, start))
    starts.append(_to_q(model, _initial_guess(model, w, Zm)))
    rng = np.random.default_rng(seed)
    base = starts[-1]
    for _ in range(max(0, n_starts - len(starts))):
        jitter = np.array([rng.normal(0, 1.5) if lg else rng.normal(0, 0.05)
                           for lg in spec["log"]])
        starts.append(base + jitter)

    best = None
    for q0 in starts:
        q0 = np.clip(q0, lb + 1e-6, ub - 1e-6)
        try:
            r = least_squares(resid, q0, bounds=(lb, ub), max_nfev=400 * k)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return result
    p = _from_q(model, best.x)
    report = p.copy()
    if model.endswith("R1‖C1 + R2‖C2"):
        if min(p[0], p[2]) < NEGLIGIBLE_SECTION_OHM:
            # One section is negligible (e.g. forward bias): it only trades a few ohms
            # with Rs. Refit with a single R‖C section and report the 2nd as "not used".
            single = "Rs + Ls + R‖C" if model.startswith("Rs + Ls") else "Rs + R‖C"
            keep = 0 if p[0] >= p[2] else 2
            st = np.r_[p[keep], p[keep + 1], max(p[4], 1.0), p[5:]]
            sub = fit_state(g_all, single, r_load_eff, reliable_only, start=st,
                            n_starts=n_starts, seed=seed)
            if "params" in sub:
                q = sub["params"]
                p = np.r_[q[0], q[1], 1e-12, 1e-12, q[2:]]
                report = np.r_[q[0], q[1], np.nan, np.nan, q[2:]]
                result["sections_used"] = 1
        else:
            if p[0] * p[1] < p[2] * p[3]:
                p = np.r_[p[2], p[3], p[0], p[1], p[4:]]   # section 1 = slower time constant
            report = p.copy()
            result["sections_used"] = 2
    r = resid(_to_q(model, p))
    m = len(f)
    result.update({
        "success": bool(best.success),
        "params": p,
        "params_report": report,
        "rms_gain_pct": float(np.sqrt(np.mean((np.exp(r[:m]) - 1) ** 2)) * 100),
        "rms_phase_deg": float(np.degrees(np.sqrt(np.mean(r[m:] ** 2)))),
        "max_phase_err_deg": float(np.degrees(np.max(np.abs(r[m:])))),
    })
    return result


def fit_all(d: pd.DataFrame, model: str, reliable_only=True, progress=None,
            r_load_eff: float | None = None) -> pd.DataFrame:
    """Fit every (wavelength, offset). Uses the previous wavelength's result as a
    starting point, which keeps neighbouring fits consistent and fast."""
    rl = r_load_eff or d.attrs.get("R_load_eff", effective_load())
    spec = MODELS[model]
    rows = []
    offsets = sorted(d["offset_set_V"].unique())
    wls = sorted(d["wavelength_nm"].unique())
    total = len(offsets) * len(wls)
    done = 0
    for off in offsets:
        prev = None
        for wl in wls:
            g = d[(d["offset_set_V"] == off) & (d["wavelength_nm"] == wl)]
            res = fit_state(g, model, rl, reliable_only, start=prev,
                            n_starts=6 if prev is None else 3, seed=int(wl))
            row = {"wavelength_nm": wl, "offset_V": off, "model": model,
                   "n_points": res["n_points"], "success": res["success"]}
            if "params" in res:
                prev = res["params"]
                for name, v in zip(spec["params"], res["params_report"]):
                    row[name] = v
                row["rms_gain_pct"] = res["rms_gain_pct"]
                row["rms_phase_deg"] = res["rms_phase_deg"]
                row["max_phase_err_deg"] = res["max_phase_err_deg"]
            rows.append(row)
            done += 1
            if progress:
                progress(done / total)
    return pd.DataFrame(rows)


def compare_models(d: pd.DataFrame, wls, offsets, reliable_only=True,
                   r_load_eff: float | None = None) -> pd.DataFrame:
    """Quick table: fit quality of every model for a few states."""
    rl = r_load_eff or d.attrs.get("R_load_eff", effective_load())
    rows = []
    for off in offsets:
        for wl in wls:
            g = d[(d["offset_set_V"] == off) & (d["wavelength_nm"] == wl)]
            for m in MODELS:
                r = fit_state(g, m, rl, reliable_only)
                if "params" in r:
                    rows.append({"wavelength_nm": wl, "offset_V": off, "model": m,
                                 "rms_gain_pct": r["rms_gain_pct"],
                                 "rms_phase_deg": r["rms_phase_deg"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 8. Plots (matplotlib; labels in English because matplotlib does not
#    lay out Hebrew right-to-left)
# ----------------------------------------------------------------------
WL_CMAP = plt.get_cmap("turbo")
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def wl_color(wl, wl_min=570, wl_max=1000):
    t = 0.0 if wl_max == wl_min else (wl - wl_min) / (wl_max - wl_min)
    return WL_CMAP(0.08 + 0.84 * float(np.clip(t, 0, 1)))


def _series_style(i_wl, n_wl, wl, i_off, wl_range, color_by="wavelength", n_off=1):
    if color_by == "offset" or n_wl == 1:
        color = plt.get_cmap("tab10")(i_off % 10)
        ls = LINESTYLES[i_wl % len(LINESTYLES)]
        mk = MARKERS[i_wl % len(MARKERS)]
    else:
        color = wl_color(wl, *wl_range)
        ls = LINESTYLES[i_off % len(LINESTYLES)]
        mk = MARKERS[i_off % len(MARKERS)]
    return color, ls, mk


FREQ_QUANTITIES = {
    "phase": ("phase_H_deg", "Phase of V2/V1 [deg]"),
    "gain": ("gain", "Gain |V2/V1|"),
    "gain_dB": ("gain_dB", "Gain [dB]"),
    "dc_out": ("ch2_dc_V", "Output DC (ch2) [V]"),
    "I_dc": ("I_dc_uA", "DC current [µA]"),
}


def plot_vs_frequency(d: pd.DataFrame, wls, offsets, quantities=("phase", "gain", "dc_out"),
                      log_gain=True, show_unreliable=True, phase_label=None):
    """Item 2: stacked plots vs frequency (log x) for every chosen wavelength x offset.
    Hollow markers = low-reliability points."""
    wls = sorted(wls)
    offsets = sorted(offsets)
    n = len(quantities)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.1 * n + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    wl_range = (float(d["wavelength_nm"].min()), float(d["wavelength_nm"].max()))
    handles = []
    for i_off, off in enumerate(offsets):
        for i_wl, wl in enumerate(wls):
            g = d[(d["wavelength_nm"] == wl) & (d["offset_set_V"] == off) & d["valid"]]
            g = g.sort_values("frequency_set_Hz")
            if g.empty:
                continue
            color, ls, mk = _series_style(i_wl, len(wls), wl, i_off, wl_range, n_off=len(offsets))
            label = f"{wl:g} nm, {fmt_v(off)}"
            for ax, q in zip(axes, quantities):
                col, _ = FREQ_QUANTITIES[q]
                ax.plot(g["frequency_set_Hz"], g[col], ls=ls, color=color, lw=1.3)
                r = g[g["reliable"]]
                ax.plot(r["frequency_set_Hz"], r[col], mk, color=color, ms=3.5, ls="none")
                if show_unreliable:
                    u = g[~g["reliable"]]
                    ax.plot(u["frequency_set_Hz"], u[col], mk, mfc="white", color=color,
                            ms=4, ls="none")
            handles.append(Line2D([], [], color=color, ls=ls, marker=mk, ms=4, label=label))
    for ax, q in zip(axes, quantities):
        col, lab = FREQ_QUANTITIES[q]
        if q == "phase" and phase_label:
            lab = phase_label
        ax.set_ylabel(lab)
        ax.set_xscale("log")
        if q == "gain" and log_gain:
            ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
    axes[-1].set_xlabel("Frequency [Hz]")
    if handles:
        ncol = 1 if len(handles) <= 12 else 2
        axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       fontsize=8, ncol=ncol, frameon=False)
    if show_unreliable:
        axes[-1].text(1.01, 0.0, "hollow markers =\nlow ch2 signal", transform=axes[-1].transAxes,
                      fontsize=7.5, color="0.4", va="bottom")
    fig.tight_layout()
    return fig


def nearest_freq(d: pd.DataFrame, f: float) -> float:
    fs = np.sort(d["frequency_set_Hz"].unique())
    return float(fs[np.argmin(np.abs(np.log(fs) - np.log(f)))])


def plot_phase_vs_wavelength(d: pd.DataFrame, freq: float, offsets, window_nm: float = 25.0,
                             show_raw_derivative=True, phase_label=None, highlight=None):
    """Items 3 + 5: phase vs wavelength at one frequency, and dphase/dlambda below."""
    offsets = sorted(offsets)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.25, 1]})
    for i, off in enumerate(offsets):
        g = d[(np.isclose(d["frequency_set_Hz"], freq)) & (d["offset_set_V"] == off) & d["valid"]]
        g = g.sort_values("wavelength_nm")
        if g.empty:
            continue
        color = plt.get_cmap("tab10")(i % 10)
        lam, p = g["wavelength_nm"].to_numpy(), g["phase_H_deg"].to_numpy()
        a1.plot(lam, p, "-", color=color, lw=1, alpha=0.8)
        rel = g["reliable"].to_numpy()
        a1.plot(lam[rel], p[rel], "o", color=color, ms=4, label=fmt_v(off))
        if (~rel).any():
            a1.plot(lam[~rel], p[~rel], "o", mfc="white", color=color, ms=4.5)
        # runs boundary
        if g["run"].nunique() > 1:
            for b in g.groupby("run")["wavelength_nm"].min().sort_values().iloc[1:]:
                a1.axvline(b - 2.5, color="0.6", ls=":", lw=1)
                a2.axvline(b - 2.5, color="0.6", ls=":", lw=1)
        if show_raw_derivative:
            a2.plot(lam, local_derivative(lam, p, 0), "-", color=color, lw=0.7, alpha=0.35)
        if window_nm > 0:
            a2.plot(lam, local_derivative(lam, p, window_nm), "-", color=color, lw=2,
                    label=f"{fmt_v(off)} (smoothed, {window_nm:g} nm)")
    a1.set_ylabel(phase_label or "Phase of V2/V1 [deg]")
    title = f"Phase vs wavelength at {_freq_label(freq)}"
    if highlight:
        title += f"   ({highlight})"
    a1.set_title(title, fontsize=11)
    a1.grid(alpha=0.35)
    a1.legend(fontsize=8, frameon=False, loc="best")
    a2.axhline(0, color="0.5", lw=0.8)
    a2.set_ylabel("dPhase/dλ [deg/nm]")
    a2.set_xlabel("Wavelength [nm]")
    a2.grid(alpha=0.35)
    if show_raw_derivative:
        a2.text(0.01, 0.97, "thin line = point-to-point derivative (no smoothing)",
                transform=a2.transAxes, fontsize=7.5, color="0.4", va="top")
    a2.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_spread_map(spread: pd.DataFrame, value="range_deg", title=None):
    """Item 4: heatmap of the between-wavelength phase spread, offset x frequency."""
    piv = spread.pivot(index="offset_V", columns="frequency_Hz", values=value).sort_index()
    rel = spread.pivot(index="offset_V", columns="frequency_Hz", values="all_reliable") \
        .reindex(index=piv.index, columns=piv.columns)
    f = piv.columns.to_numpy(float)
    edges = np.r_[f[0] / np.sqrt(f[1] / f[0]), np.sqrt(f[:-1] * f[1:]), f[-1] * np.sqrt(f[-1] / f[-2])]
    y = np.arange(len(piv.index) + 1) - 0.5
    fig, ax = plt.subplots(figsize=(10, 2.6 + 0.35 * len(piv.index)))
    cmap = "magma_r" if value != "snr" else "RdBu_r"
    kw = {}
    if value == "snr":
        vmax = max(2.0, float(np.nanmax(piv.values)))
        from matplotlib.colors import TwoSlopeNorm
        kw["norm"] = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)
    mesh = ax.pcolormesh(edges, y, piv.values, cmap=cmap, shading="flat", **kw)
    # mark states with low-signal points
    for iy, off in enumerate(piv.index):
        for ix in range(len(f)):
            if rel.iloc[iy, ix] is False or rel.iloc[iy, ix] == 0:
                ax.plot(f[ix], iy, "x", color="0.55", ms=3.5, mew=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([fmt_v(o) for o in piv.index])
    ax.set_xlabel("Frequency [Hz]")
    labels = {"range_deg": "Phase range over wavelengths [deg]",
              "std_deg": "Phase std over wavelengths [deg]",
              "snr": "Spread / point noise", "trend_r2": "Trend R²"}
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label(labels.get(value, value))
    ax.set_title(title or "Phase differences between wavelengths  (x = includes low-signal points)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_fit(d: pd.DataFrame, fits: dict, wl: float, offsets, model: str,
             r_load_eff: float | None = None):
    """Item 6: measured vs model for chosen wavelength and offsets.
    Top: |H| and phase of H.  Bottom: |Z| and phase of Z."""
    rl = r_load_eff or d.attrs.get("R_load_eff", effective_load())
    offsets = sorted(offsets)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for i, off in enumerate(offsets):
        g = d[(d["wavelength_nm"] == wl) & (d["offset_set_V"] == off) & d["valid"]]
        g = g.sort_values("frequency_set_Hz")
        if g.empty:
            continue
        color = plt.get_cmap("tab10")(i % 10)
        f = g["frequency_set_Hz"].to_numpy()
        rel = g["reliable"].to_numpy()
        H = g["H_re"].to_numpy() + 1j * g["H_im"].to_numpy()
        Z = g["Z_re_ohm"].to_numpy() + 1j * g["Z_im_ohm"].to_numpy()
        for ax, y in zip(axs.ravel(), [np.abs(H), np.degrees(np.angle(H)),
                                       np.abs(Z), np.degrees(np.angle(Z))]):
            ax.plot(f[rel], y[rel], "o", color=color, ms=4)
            ax.plot(f[~rel], y[~rel], "o", mfc="white", color=color, ms=4.5)
        res = fits.get((wl, off))
        lab = f"{fmt_v(off)}"
        if res and "params" in res:
            ff = np.logspace(np.log10(f.min()), np.log10(f.max()), 300)
            Hm = model_H(model, res["params"], ff, rl)
            Zm = model_Z(model, res["params"], ff)
            for ax, y in zip(axs.ravel(), [np.abs(Hm), np.degrees(np.angle(Hm)),
                                           np.abs(Zm), np.degrees(np.angle(Zm))]):
                ax.plot(ff, y, "-", color=color, lw=1.6)
            lab += f"  (rms {res['rms_gain_pct']:.1f}%, {res['rms_phase_deg']:.2f}°)"
        axs[0, 0].plot([], [], "o-", color=color, label=lab)
    axs[0, 0].set_ylabel("|H| = |V2/V1|")
    axs[0, 0].set_yscale("log")
    axs[0, 1].set_ylabel("Phase of H [deg]")
    axs[1, 0].set_ylabel("|Z| [ohm]")
    axs[1, 0].set_yscale("log")
    axs[1, 1].set_ylabel("Phase of Z [deg]")
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    for ax in axs[1]:
        ax.set_xlabel("Frequency [Hz]")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{wl:g} nm — points: measured (hollow = low signal), lines: model {model}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_fit_quality(fit_table: pd.DataFrame):
    """Item 6 overview: rms errors per wavelength for every offset."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for i, (off, g) in enumerate(fit_table.groupby("offset_V")):
        g = g.sort_values("wavelength_nm")
        c = plt.get_cmap("tab10")(i % 10)
        a1.plot(g["wavelength_nm"], g["rms_gain_pct"], "o-", ms=3, color=c, label=fmt_v(off))
        a2.plot(g["wavelength_nm"], g["rms_phase_deg"], "o-", ms=3, color=c)
    a1.set_ylabel("RMS gain error [%]")
    a2.set_ylabel("RMS phase error [deg]")
    for ax in (a1, a2):
        ax.set_xlabel("Wavelength [nm]")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    a1.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Model fit quality (lower = better)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_params_vs_wavelength(fit_table: pd.DataFrame, model: str):
    spec = MODELS[model]
    names = spec["params"]
    n = len(names)
    ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), sharex=True, squeeze=False)
    for k, (name, unit, lg) in enumerate(zip(names, spec["units"], spec["log"])):
        ax = axs.ravel()[k]
        for i, (off, g) in enumerate(fit_table.groupby("offset_V")):
            g = g.sort_values("wavelength_nm")
            c = plt.get_cmap("tab10")(i % 10)
            scale, u = 1.0, unit
            if unit == "F":
                scale, u = 1e9, "nF"
            elif unit == "ohm" and name == "Rs":
                scale, u = 1.0, "Ω"
            elif unit == "ohm":
                scale, u = 1e-3, "kΩ"
            elif unit == "H":
                scale, u = 1e6, "µH"
            ax.plot(g["wavelength_nm"], g[name] * scale, "o-", ms=2.5, lw=1, color=c,
                    label=fmt_v(off))
        ax.set_ylabel(f"{name} [{u}]" if u else name)
        vals = fit_table[name].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if lg and len(vals) and vals.max() / vals.min() > 20:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("Wavelength [nm]")
    axs.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Fitted parameters vs wavelength — {model}   (gaps = section not needed)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_to_png(fig, dpi=150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()
