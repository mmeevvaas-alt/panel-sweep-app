"""
app.py — Streamlit app for the solar-panel sweep ZIPs.

Run (from this folder):
    python -m streamlit run app.py
A browser tab opens. Upload one or more sweep ZIP files in the sidebar.
"""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import panel_analysis as pa

st.set_page_config(page_title="Panel sweep analysis", layout="wide")

# Hebrew text right-to-left (tables and plots stay left-to-right)
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"], [data-testid="stAlert"],
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {
        direction: rtl; text-align: right;
    }
    [data-testid="stMarkdownContainer"] code { direction: ltr; unicode-bidi: embed; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached computations
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all(files: tuple):
    """files = ((name, bytes), ...)"""
    runs, errors = [], []
    for name, data in files:
        try:
            runs.extend(pa.load_zip(data, name))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    return runs, errors


@st.cache_data(show_spinner=False)
def prepare(files: tuple, r_load: float, scope_rin: float, thr_mv: float, physical: bool):
    runs, errors = load_all(files)
    if not runs:
        return runs, errors, None, None, None, None
    data, source = pa.merge_runs(runs)
    d = pa.add_derived(data, r_load, scope_rin, thr_mv / 1000.0, physical)
    report = pa.coverage_report(runs, d, source)
    spread = pa.phase_spread_table(d)
    return runs, errors, d, source, report, spread


@st.cache_data(show_spinner=False)
def fit_all_cached(_d, key: str, model: str, reliable_only: bool, rl: float):
    # _d is not hashed (leading underscore); "key" identifies the data instead
    return pa.fit_all(_d, model, reliable_only, r_load_eff=rl)


def show_fig(fig, name: str, key: str):
    st.pyplot(fig, clear_figure=False)
    st.download_button("⬇️ הורדת הגרף (PNG)", pa.fig_to_png(fig), file_name=name,
                       mime="image/png", key=key)
    plt.close(fig)


def flab(f: float) -> str:
    return pa._freq_label(f)


def vlab(v: float) -> str:
    return pa.fmt_v(v)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("קבצים")
uploads = st.sidebar.file_uploader(
    "העלי קובץ ZIP אחד או יותר של סריקות (sweep_….zip)",
    type=["zip"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.subheader("הגדרות חישוב")
r_load = st.sidebar.number_input("נגד העומס R_L [Ω]", value=1000.0, min_value=1.0, step=100.0)
use_rin = st.sidebar.checkbox("לכלול את כניסת הסקופ (1 MΩ) במקביל לנגד", value=True)
scope_rin = 1e6 if use_rin else 0.0
thr_mv = st.sidebar.number_input(
    "סף אמינות: אמפליטודה מינימלית בערוץ 2 [mV]", value=10.0, min_value=0.0, step=1.0,
    help="נקודות עם אות חלש יותר מסומנות כ'לא אמינות' (עיגול חלול בגרפים). "
         "בבדיקת הנתונים נמצא שמתחת לכ-10 mV האמפליטודה תלויה בסקאלת הסקופ.")
phase_mode = st.sidebar.radio(
    "מוסכמת פאזה בגרפים",
    ["פאזה פיזיקלית של V2/V1  (= ‎−phase_deg)", "הערך הגולמי של הסקופ (phase_deg)"],
    help="הסקופ מחזיר φ1−φ2. בפאזה הפיזיקלית ערך חיובי = ערוץ 2 (הזרם) מקדים את ערוץ 1. "
         "החישוב של Z משתמש תמיד במוסכמה הפיזיקלית.")
physical = phase_mode.startswith("פאזה פיזיקלית")
phase_label = "Phase of V2/V1 [deg]" if physical else "Scope phase_deg (ch1 vs ch2) [deg]"

st.sidebar.markdown("---")
st.sidebar.subheader("דירוג מצבי מדידה")
metric = st.sidebar.selectbox(
    "איך לדרג 'הבדלים גדולים בין אורכי גל'",
    list(pa.RANK_METRICS.keys()),
    format_func=lambda k: {
        "range_reliable": "טווח (מקס'−מינ'), רק מצבים אמינים",
        "range": "טווח (מקס'−מינ'), כל המצבים",
        "snr": "מובהקות: פיזור ביחס לרעש",
        "trend": "מגמה שיטתית (R²)",
    }[k],
    help="הדירוג קובע גם את ברירת המחדל בגרף הפאזה מול אורך גל.")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("ניתוח סריקות פאנל סולארי")

if not uploads:
    st.info("העלי בצד ימין/שמאל (סרגל הצד) את קובצי ה-ZIP של הסריקה. אפשר להעלות כמה קבצים יחד "
            "(למשל ריצת הלילה וריצת הבוקר) — הם יאוחדו אוטומטית. "
            "באורכי גל שמופיעים ביותר מריצה אחת נלקחים הנתונים מהריצה **המאוחרת** שבה אורך הגל שלם.")
    st.stop()

files = tuple(sorted(((u.name, u.getvalue()) for u in uploads), key=lambda x: x[0]))
data_key = hashlib.md5(b"".join(n.encode() + hashlib.md5(b).digest() for n, b in files)).hexdigest()
with st.spinner("קוראת ומעבדת את הקבצים…"):
    runs, errors, d, source, report, spread = prepare(files, r_load, scope_rin, thr_mv, physical)
for e in errors:
    st.error(f"שגיאה בקריאת קובץ: {e}")
if d is None:
    st.stop()
rl_eff = pa.effective_load(r_load, scope_rin)

wls_all = sorted(d["wavelength_nm"].unique().tolist())
offs_all = sorted(d["offset_set_V"].unique().tolist())
freqs_all = sorted(d["frequency_set_Hz"].unique().tolist())
ranked = pa.rank_states(spread, metric)
if ranked.empty:
    ranked = pa.rank_states(spread, "range")
top = ranked.iloc[0]

tabs = st.tabs(["1. פרטי הניסוי ואיכות",
                "2. מול תדר",
                "3+5. פאזה מול אורך גל",
                "4. הבדלים בין אורכי גל",
                "6. אימפדנס Z(ω)",
                "הורדות"])

# ---------------------------------------------------------------- tab 1
with tabs[0]:
    st.header("פרטי הניסוי")
    for r in runs:
        with st.expander(f"ריצה: {r.name}", expanded=True):
            if r.stop_message:
                st.warning(f"הריצה נעצרה: `{r.stop_message}`")
            else:
                st.success("אין הודעת עצירה ב-settings.txt (הריצה הסתיימה כמתוכנן).")
            for n in r.notes:
                st.warning(n)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**settings.txt**")
                st.dataframe(pd.DataFrame({"פרמטר": list(r.settings.keys()),
                                           "ערך": list(r.settings.values())}),
                             hide_index=True, width="stretch")
            with c2:
                st.markdown("**ערכים שחולצו**")
                st.dataframe(pd.DataFrame({"שדה": list(r.parsed.keys()),
                                           "ערך": [str(v) for v in r.parsed.values()]}),
                             hide_index=True, width="stretch")

    st.header("כיסוי ואיכות")
    st.dataframe(report["per_run"], hide_index=True, width="stretch")
    status_txt = ", ".join(f"{k or '(ריק)'}: {v}" for k, v in report["status_counts"].items())
    c = st.columns(4)
    c[0].metric("אורכי גל שנמדדו", f"{report['measured_n']} / {report['planned_n'] or '?'}")
    c[1].metric("שורות בנתונים המאוחדים", f"{report['rows']:,}")
    c[2].metric("נקודות תקינות", f"{report['n_valid']:,}")
    c[3].metric("נקודות אמינות (אות ≥ סף)", f"{report['n_reliable']:,}")
    st.markdown(
        f"- טווח אורכי גל: **{report['wl_range'][0]:g}–{report['wl_range'][1]:g} nm**; "
        f"מתחי DC: **{', '.join(vlab(o) for o in report['offsets'])}**; "
        f"**{report['n_freq']}** תדרים, {flab(report['f_range'][0])}–{flab(report['f_range'][1])}.\n"
        f"- סטטוס: {status_txt}. "
        f"ערכים חסרים: {report['n_nan']}. תדר נמדד שחורג ביותר מ-2%: {report['n_freq_bad']}.")
    if report["missing"]:
        miss = report["missing"]
        st.warning(f"אורכי גל מתוכננים שחסרים ({len(miss)}): "
                   f"{', '.join(f'{m:g}' for m in miss[:40])}{' …' if len(miss) > 40 else ''}")
    if report["incomplete"]:
        st.warning("אורכי גל חלקיים (לא כל המתחים/התדרים): "
                   + ", ".join(f"{m:g}" for m in report["incomplete"]))
    if len(runs) > 1:
        st.markdown("**מאיזו ריצה נלקח כל אורך גל**")
        st.dataframe(source, hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    st.header("פאזה, הגבר ומתח DC במוצא כפונקציה של התדר")
    c1, c2 = st.columns(2)
    sel_wl = c1.multiselect("אורכי גל [nm]", wls_all,
                            default=[wls_all[len(wls_all) // 2]], format_func=lambda x: f"{x:g}")
    sel_off = c2.multiselect("מתח DC של אות הכניסה", offs_all, default=offs_all,
                             format_func=vlab)
    c3, c4, c5 = st.columns(3)
    gain_db = c3.checkbox("הגבר ב-dB", value=False)
    show_unrel = c4.checkbox("להציג נקודות לא אמינות (חלולות)", value=True)
    dc_as_current = c5.checkbox("להציג DC כזרם [µA] במקום מתח", value=False)
    n_series = len(sel_wl) * len(sel_off)
    if n_series == 0:
        st.info("בחרי לפחות אורך גל אחד ומתח אחד.")
    else:
        if n_series > 30:
            st.warning(f"נבחרו {n_series} סדרות — הגרף יהיה עמוס.")
        q = ("phase", "gain_dB" if gain_db else "gain", "I_dc" if dc_as_current else "dc_out")
        fig = pa.plot_vs_frequency(d, sel_wl, sel_off, q, log_gain=not gain_db,
                                   show_unreliable=show_unrel, phase_label=phase_label)
        show_fig(fig, "vs_frequency.png", "dl_t2")
        st.caption("צבע = אורך גל, סוג קו/סמן = מתח DC (כשנבחר אורך גל אחד: צבע = מתח). "
                   "עיגול חלול = אמפליטודה בערוץ 2 מתחת לסף האמינות.")
        st.info("שימי לב: מתח ה-DC בערוץ 2 קופץ בכמה mV בכל פעם שסקאלת ערוץ 2 משתנה "
                "(בעיקר סביב 10 kHz). זו סטייה של הסקופ בין טווחים ולא תכונה של הפאנל — "
                "ערכי ה-DC המדויקים הם בתדרים הנמוכים (סקאלה קטנה).")

# ---------------------------------------------------------------- tab 3+5
with tabs[2]:
    st.header("פאזה כפונקציה של אורך הגל, והנגזרת שלה")
    st.markdown(f"ברירת המחדל: המצב עם ההבדלים הגדולים ביותר לפי הדירוג שנבחר בסרגל הצד — "
                f"**{vlab(top.offset_V)}, {flab(top.frequency_Hz)}** "
                f"(טווח {top.range_deg:.2f}°).")
    c1, c2, c3 = st.columns([2, 2, 1])
    f_sel = c1.select_slider("תדר", options=freqs_all, value=top.frequency_Hz, format_func=flab)
    off_sel = c2.multiselect("מתח DC", offs_all, default=[top.offset_V], format_func=vlab)
    win = c3.number_input("חלון החלקה לנגזרת [nm]", value=25.0, min_value=0.0, step=5.0,
                          help="הנגזרת מחושבת מהתאמה ריבועית מקומית בחלון הזה. 0 = בלי החלקה. "
                               "בלי החלקה הנגזרת מגבירה מאוד את הרעש.")
    if not off_sel:
        st.info("בחרי לפחות מתח אחד.")
    else:
        is_top = np.isclose(f_sel, top.frequency_Hz) and off_sel == [top.offset_V]
        fig = pa.plot_phase_vs_wavelength(d, f_sel, off_sel, win, True, phase_label,
                                          highlight="largest spread" if is_top else None)
        show_fig(fig, "phase_vs_wavelength.png", "dl_t3")
        row = spread[np.isclose(spread.frequency_Hz, f_sel) & spread.offset_V.isin(off_sel)]
        if len(row):
            show = row[["offset_V", "range_deg", "std_deg", "noise_deg", "snr", "trend_r2",
                        "slope_deg_per_100nm", "wl_at_max", "wl_at_min", "all_reliable"]].copy()
            show.columns = ["מתח [V]", "טווח [°]", "סטיית תקן [°]", "רעש נקודתי [°]", "יחס פיזור/רעש",
                            "R² מגמה", "שיפוע [°/100 nm]", "λ במקס'", "λ במינ'", "כל הנקודות אמינות"]
            st.dataframe(show.round(3), hide_index=True, width="stretch")
        st.caption("רעש נקודתי = std(הפרשים בין אורכי גל סמוכים)/√2. יחס פיזור/רעש ≈ 1 → "
                   "ההבדלים נראים כמו פיזור אקראי; יחס גבוה בבירור מ-1 → שינוי שיטתי עם אורך הגל.")

# ---------------------------------------------------------------- tab 4
with tabs[3]:
    st.header("ניתוח ההבדלים בפאזה בין אורכי הגל")
    st.markdown(pa.spread_analysis_text(spread, d, metric))
    st.subheader("מפת ההבדלים: מתח × תדר")
    map_val = st.radio("מה להציג במפה", ["range_deg", "std_deg", "snr", "trend_r2"], horizontal=True,
                       format_func=lambda k: {"range_deg": "טווח", "std_deg": "סטיית תקן",
                                              "snr": "פיזור/רעש", "trend_r2": "R² מגמה"}[k])
    show_fig(pa.plot_spread_map(spread, map_val), f"spread_map_{map_val}.png", "dl_t4")
    st.subheader("טבלת הדירוג")
    n_show = st.slider("כמה שורות להציג", 5, len(ranked), min(20, len(ranked)))
    tbl = ranked.head(n_show)[["offset_V", "frequency_Hz", "range_deg", "std_deg", "noise_deg",
                               "snr", "trend_r2", "slope_deg_per_100nm", "mean_phase_deg",
                               "wl_at_max", "wl_at_min", "ch2_amp_min_mV", "all_reliable"]].copy()
    tbl.insert(0, "#", range(1, len(tbl) + 1))
    st.dataframe(tbl.round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 6
with tabs[4]:
    st.header("האימפדנס השקול של הפאנל Z(ω)")
    st.markdown(
        "פונקציית התמסורת: **H(ω) = V2/V1 = R_L / (R_L + Z(ω))**, ולכן מכל נקודת מדידה "
        "**Z(ω) = R_L·(V1/V2 − 1)** (בפאזורים). "
        f"R_L האפקטיבי בחישוב: **{rl_eff:.1f} Ω**. "
        "הנקודות בגרפים הן Z שחושב ישירות מהמדידה; הקווים הם מודל של מעגל שקול "
        "שהותאם לכל אורך גל ולכל מתח בנפרד (על פני כל התדרים).")
    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("מודל המעגל השקול", list(pa.MODELS.keys()),
                         help="\n\n".join(f"{k}: {v['desc']}" for k, v in pa.MODELS.items()))
    rel_only = c2.checkbox("התאמה רק לנקודות אמינות", value=True)
    st.caption(pa.MODELS[model]["desc"])

    c1, c2 = st.columns(2)
    wl_fit = c1.selectbox("אורך גל [nm]", wls_all, index=len(wls_all) // 2,
                          format_func=lambda x: f"{x:g}")
    off_fit = c2.multiselect("מתחי DC", offs_all, default=offs_all, format_func=vlab, key="off_fit")
    fits = {}
    for off in off_fit:
        g = d[(d.wavelength_nm == wl_fit) & (d.offset_set_V == off)]
        fits[(wl_fit, off)] = pa.fit_state(g, model, rl_eff, rel_only)
    if off_fit:
        show_fig(pa.plot_fit(d, fits, wl_fit, off_fit, model, rl_eff),
                 f"Z_fit_{wl_fit:g}nm.png", "dl_t6a")
        rows = []
        for off in off_fit:
            r = fits[(wl_fit, off)]
            row = {"מתח": vlab(off), "נקודות": r["n_points"]}
            if "params" in r:
                for name, unit, v in zip(pa.MODELS[model]["params"], pa.MODELS[model]["units"],
                                         r["params_report"]):
                    row[f"{name} [{unit}]" if unit else name] = "לא נדרש" if np.isnan(v) else f"{v:.4g}"
                row["RMS הגבר [%]"] = round(r["rms_gain_pct"], 2)
                row["RMS פאזה [°]"] = round(r["rms_phase_deg"], 2)
                row["שגיאת פאזה מרבית [°]"] = round(r["max_phase_err_deg"], 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander("השוואה בין כל המודלים לאורך הגל הזה"):
            cmp = pa.compare_models(d, [wl_fit], off_fit, rel_only, rl_eff)
            if len(cmp):
                st.dataframe(cmp.pivot(index="model", columns="offset_V",
                                       values=["rms_gain_pct", "rms_phase_deg"]).round(2),
                             width="stretch")
                st.caption("rms_gain_pct = שגיאה יחסית ממוצעת בהגבר [%]; "
                           "rms_phase_deg = שגיאה ממוצעת בפאזה [°]. נמוך יותר = התאמה טובה יותר.")

    st.info("**מה לדעת על ההתאמה:**\n"
            "- במתח 0 ובמתח אחורי המודל עם שני מקטעי R‖C מתאים למדידות היטב (כ-1–2% בהגבר, "
            "פחות מ-1° בפאזה).\n"
            "- במתח קדמי (+4.5, +9 V) בתדרים נמוכים נמדד Z עם פאזה חיובית (אינדוקטיבית, עד כ-+8°) "
            "בזמן שהגודל כמעט קבוע. אף אחד מהמודלים לא מתאר את זה, ולכן שגיאת הפאזה שם 3–4°. "
            "ייתכן שזו תופעה של הצומת בממתח קדמי (מה שמכונה 'קיבול שלילי'), וייתכן שזו "
            "השפעה של אי-ליניאריות כשאמפליטודה של 1 V מוסיפה על מתח קדמי.\n"
            "- מעל ~300 kHz, |Z| קטן (~10–20 Ω לעומת R_L = 1 kΩ), כך שהפרש קטן בפאזה (0.5°) "
            "משנה את Z משמעותית. שם Z מוגבל בדיוק המדידה; העלייה בפאזת Z לכיוון חיובי מתאימה "
            "להשראות של החיווט (המודל עם Ls).")

    st.subheader("התאמה לכל אורכי הגל והמתחים")
    fit_key = f"{data_key}|{model}|{rel_only}|{rl_eff:.6f}"
    if st.button("▶️ הרצת התאמה לכל המצבים (כ-5–15 שניות)"):
        with st.spinner("מתאימה את המודל לכל אורך גל ומתח…"):
            st.session_state["fit_table"] = fit_all_cached(d, fit_key, model, rel_only, rl_eff)
            st.session_state["fit_key"] = fit_key
            st.session_state["fit_model"] = model
    if st.session_state.get("fit_key") == fit_key:
        ft = st.session_state["fit_table"]
        show_fig(pa.plot_fit_quality(ft), "fit_quality.png", "dl_t6b")
        show_fig(pa.plot_params_vs_wavelength(ft, model), "fit_params.png", "dl_t6c")
        summ = ft.groupby("offset_V")[["rms_gain_pct", "rms_phase_deg"]].median().round(2)
        summ.index = [vlab(o) for o in summ.index]
        summ.columns = ["חציון RMS הגבר [%]", "חציון RMS פאזה [°]"]
        st.dataframe(summ, width="stretch")
        st.download_button("⬇️ טבלת הפרמטרים לכל המצבים (CSV)", ft.to_csv(index=False).encode(),
                           file_name=f"Z_fit_{model.replace(' ', '').replace('‖', '_')}.csv",
                           mime="text/csv")
    elif "fit_key" in st.session_state:
        st.caption("ההגדרות השתנו מאז ההתאמה האחרונה — לחצי שוב על הכפתור.")

# ---------------------------------------------------------------- downloads
with tabs[5]:
    st.header("הורדת נתונים מעובדים")
    cols = ["run", "timestamp", "wavelength_nm", "aotf_power_percent", "offset_set_V",
            "frequency_set_Hz", "ch1_amplitude_V", "ch2_amplitude_V", "ch1_dc_V", "ch2_dc_V",
            "phase_deg", "phase_H_deg", "gain", "gain_dB", "I_dc_uA", "V_panel_dc_V",
            "Z_re_ohm", "Z_im_ohm", "Z_abs_ohm", "Z_phase_deg", "ch2_scale_V_per_div",
            "status", "valid", "reliable"]
    st.download_button("⬇️ כל הנתונים המאוחדים + עמודות מחושבות (CSV)",
                       d[[c for c in cols if c in d.columns]].to_csv(index=False).encode(),
                       file_name="merged_processed.csv", mime="text/csv")
    st.download_button("⬇️ טבלת ההבדלים בין אורכי הגל לכל מצב (CSV)",
                       spread.to_csv(index=False).encode(),
                       file_name="phase_spread_by_state.csv", mime="text/csv")
    st.markdown("**הסבר לעמודות המחושבות:** `phase_H_deg` = פאזה של V2/V1 לפי המוסכמה שנבחרה; "
                "`gain` = ch2_amplitude/ch1_amplitude; `I_dc_uA` = ch2_dc/R_L; "
                "`V_panel_dc_V` = ch1_dc − ch2_dc; `Z_*` = האימפדנס שחושב מכל נקודה; "
                "`valid` = סטטוס ok ותדר תקין; `reliable` = valid ואמפליטודת ערוץ 2 מעל הסף.")
