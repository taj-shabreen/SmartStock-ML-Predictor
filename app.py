# ============================================================
# app.py — PREMIUM STOCKSENSE UI
# Bug-fixed: all HTML pre-built as strings before st.markdown
# Aesthetic: Obsidian Fintech Terminal · Teal + Gold
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, io, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_fetcher import fetch_stock_data, preprocess_data, get_stock_info, get_currency_symbol
from train import (
    prepare_train_test_split,
    train_all_classification_models,
    train_all_regression_models,
    save_models,
    get_feature_importance,
)
from evaluation import (
    evaluate_classification_models,
    evaluate_regression_models,
    select_best_classification_model,
    select_best_regression_model,
    get_confusion_matrix,
    predict_next_day,
    summarize_results,
)

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="StockSense · ML Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS ────────────────────────────────────────
BG      = "#05070e"
SURFACE = "#0b0f1a"
CARD    = "#0f1624"
CARD2   = "#131d2e"
BORDER  = "#1a2740"
BORDER2 = "#22334f"
TEAL    = "#00e5b4"
TEAL2   = "#00bfa5"
GOLD    = "#f0b429"
GOLD2   = "#ffd166"
BLUE    = "#4f8ef7"
RED     = "#ff4d6d"
GREEN   = "#22d3a0"
PURPLE  = "#b48ef7"
PINK    = "#f472b6"
TEXT    = "#edf2f7"
TEXT2   = "#c8d8e8"
MUTED   = "#4a6380"
MUTED2  = "#7a9ab5"

# ─── MASTER CSS ───────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --card:{CARD}; --card2:{CARD2};
  --border:{BORDER}; --border2:{BORDER2};
  --teal:{TEAL}; --teal2:{TEAL2}; --gold:{GOLD}; --gold2:{GOLD2};
  --blue:{BLUE}; --red:{RED}; --green:{GREEN}; --purple:{PURPLE}; --pink:{PINK};
  --text:{TEXT}; --text2:{TEXT2}; --muted:{MUTED}; --muted2:{MUTED2};
}}

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}

html,body,.stApp{{
  background:var(--bg)!important;
  color:var(--text)!important;
  font-family:'Inter',sans-serif!important;
}}

#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stDecoration"]{{display:none;}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
}}
[data-testid="stSidebar"] *{{
  color:var(--text)!important;
  font-family:'Inter',sans-serif!important;
}}
[data-testid="stSidebar"] label{{
  color:var(--text2)!important;
  font-size:11px!important;
  font-weight:600!important;
  letter-spacing:1.5px!important;
  text-transform:uppercase!important;
}}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p{{color:var(--muted2)!important;}}

/* ── INPUTS ── */
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input{{
  background:rgba(0,229,180,0.04)!important;
  border:1.5px solid var(--border2)!important;
  border-radius:10px!important;
  color:var(--text)!important;
  font-family:'Space Mono',monospace!important;
  font-size:13px!important;
  padding:10px 14px!important;
}}
[data-testid="stTextInput"] input:focus{{
  border-color:var(--teal)!important;
  box-shadow:0 0 0 3px rgba(0,229,180,0.1)!important;
}}

/* ── SLIDER ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{{
  background:var(--teal)!important;
  border-color:var(--teal)!important;
  box-shadow:0 0 12px rgba(0,229,180,0.4)!important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"]{{
  background:var(--teal)!important;
}}

/* ── BUTTONS ── */
.stButton>button{{
  background:rgba(0,229,180,0.06)!important;
  border:1.5px solid var(--border2)!important;
  color:var(--text)!important;
  border-radius:10px!important;
  font-family:'Inter',sans-serif!important;
  font-size:13px!important;
  font-weight:600!important;
  padding:11px 18px!important;
  transition:all .2s ease!important;
  width:100%!important;
  letter-spacing:0.3px!important;
}}
.stButton>button:hover{{
  background:rgba(0,229,180,0.12)!important;
  border-color:var(--teal)!important;
  color:var(--teal)!important;
  transform:translateY(-1px)!important;
  box-shadow:0 4px 20px rgba(0,229,180,0.15)!important;
}}
.stButton>button:disabled{{
  background:rgba(10,15,25,0.6)!important;
  border-color:var(--border)!important;
  color:var(--muted)!important;
  opacity:0.5!important;
  cursor:not-allowed!important;
  transform:none!important;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{{
  background:transparent!important;
  border-bottom:1px solid var(--border)!important;
  gap:0!important; padding:0!important;
}}
.stTabs [data-baseweb="tab"]{{
  background:transparent!important;
  border:none!important;
  border-bottom:2px solid transparent!important;
  border-radius:0!important;
  color:var(--muted2)!important;
  font-family:'Inter',sans-serif!important;
  font-size:12.5px!important;
  font-weight:600!important;
  padding:13px 22px!important;
  transition:all .15s!important;
  letter-spacing:0.2px!important;
}}
.stTabs [data-baseweb="tab"]:hover{{color:var(--text2)!important;}}
.stTabs [aria-selected="true"]{{
  color:var(--teal)!important;
  border-bottom-color:var(--teal)!important;
  background:transparent!important;
}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:30px!important;}}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{{border:1px solid var(--border)!important;border-radius:14px!important;overflow:hidden!important;}}

/* ── ALERTS ── */
.stAlert{{border-radius:12px!important;font-family:'Inter',sans-serif!important;font-size:13px!important;}}
.stSuccess{{background:rgba(34,211,160,.07)!important;border-color:var(--green)!important;color:var(--text)!important;}}
.stWarning{{background:rgba(240,180,41,.07)!important;border-color:var(--gold)!important;color:var(--text)!important;}}
.stError{{background:rgba(255,77,109,.07)!important;border-color:var(--red)!important;color:var(--text)!important;}}
.stInfo{{background:rgba(79,142,247,.07)!important;border-color:var(--blue)!important;color:var(--text)!important;}}

/* ── EXPANDER ── */
.streamlit-expanderHeader{{
  background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:10px!important;font-family:'Inter',sans-serif!important;
  font-weight:600!important;font-size:12.5px!important;color:var(--text2)!important;
}}

/* ── DOWNLOAD ── */
[data-testid="stDownloadButton"]>button{{
  background:rgba(79,142,247,0.07)!important;
  border:1.5px solid var(--border2)!important;
  color:var(--text2)!important;border-radius:10px!important;
  font-size:12px!important;font-weight:600!important;
  font-family:'Inter',sans-serif!important;transition:all .2s!important;
}}
[data-testid="stDownloadButton"]>button:hover{{
  border-color:var(--blue)!important;color:var(--blue)!important;
  background:rgba(79,142,247,0.12)!important;
}}

hr{{border:none;border-top:1px solid var(--border)!important;margin:28px 0!important;}}

.block-container{{padding:0 2.5rem 4rem 2.5rem!important;max-width:1480px!important;}}

/* ════ SIDEBAR COMPONENTS ════ */
.sb-brand{{
  padding:28px 22px 22px;
  border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(0,229,180,0.04),transparent);
}}
.sb-logo-row{{display:flex;align-items:center;gap:12px;margin-bottom:8px;}}
.sb-logo{{
  width:36px;height:36px;
  background:linear-gradient(135deg,var(--teal),#0099ff);
  border-radius:10px;display:flex;align-items:center;
  justify-content:center;font-size:18px;flex-shrink:0;
  box-shadow:0 4px 15px rgba(0,229,180,0.3);
}}
.sb-name{{font-size:19px!important;font-weight:800!important;color:var(--text)!important;
  font-family:'Sora',sans-serif!important;letter-spacing:-0.5px;}}
.sb-sub{{font-size:9px!important;color:var(--muted2)!important;letter-spacing:3px;
  text-transform:uppercase;font-family:'Space Mono',monospace!important;}}
.sb-sec{{
  font-family:'Space Mono',monospace!important;font-size:8.5px!important;
  font-weight:700!important;letter-spacing:3px!important;text-transform:uppercase!important;
  color:var(--muted)!important;padding:18px 22px 7px!important;
}}
.sb-status{{
  display:flex;align-items:center;gap:9px;
  padding:10px 14px;
  background:rgba(0,229,180,0.06);
  border:1px solid rgba(0,229,180,0.2);
  border-radius:10px;margin:6px 14px;
}}
.sb-dot{{
  width:7px;height:7px;border-radius:50%;
  background:var(--teal);
  box-shadow:0 0 8px rgba(0,229,180,0.6);
  animation:pulse 2s ease-in-out infinite;flex-shrink:0;
}}
.sb-status span{{font-family:'Space Mono',monospace!important;font-size:11px!important;color:var(--teal)!important;}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:.5;transform:scale(0.85);}}}}

/* ════ TOP BAR ════ */
.topbar{{
  display:flex;align-items:center;justify-content:space-between;
  padding:26px 0 20px;border-bottom:1px solid var(--border);margin-bottom:32px;
}}
.tb-left{{}}
.tb-h1{{
  font-size:28px;font-weight:800;color:var(--text);
  font-family:'Sora',sans-serif;letter-spacing:-0.8px;
  line-height:1;margin-bottom:6px;
}}
.tb-sub{{
  font-size:10.5px;color:var(--muted2);
  font-family:'Space Mono',monospace;letter-spacing:1.5px;
}}
.tb-badges{{display:flex;gap:8px;align-items:center;}}
.tb-badge{{
  display:inline-flex;align-items:center;gap:7px;
  background:var(--card2);border:1px solid var(--border2);
  border-radius:20px;padding:6px 14px;
  font-size:11px;color:var(--muted2);font-family:'Space Mono',monospace;
}}
.tb-badge.live{{border-color:rgba(0,229,180,0.4);color:var(--teal);background:rgba(0,229,180,0.06);}}
.tb-live-dot{{
  width:6px;height:6px;border-radius:50%;
  background:var(--teal);display:inline-block;
  animation:pulse 2s infinite;
}}

/* ════ HERO CARD ════ */
.hero{{
  position:relative;overflow:hidden;
  background:linear-gradient(145deg,#0c1525,#0a1320 45%,#0f1a2e);
  border:1px solid var(--border2);border-radius:22px;
  margin-bottom:28px;
  box-shadow:0 8px 40px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04);
}}
.hero::before{{
  content:'';position:absolute;top:-120px;right:-80px;
  width:500px;height:500px;
  background:radial-gradient(ellipse,rgba(0,229,180,0.06),transparent 65%);
  pointer-events:none;
}}
.hero::after{{
  content:'';position:absolute;bottom:-80px;left:-40px;
  width:350px;height:350px;
  background:radial-gradient(ellipse,rgba(79,142,247,0.05),transparent 65%);
  pointer-events:none;
}}
.hero-grid{{
  display:grid;
  grid-template-columns:1fr 1px 1fr 1px 1fr;
  min-height:210px;position:relative;z-index:1;
}}
.hero-divider{{background:var(--border);align-self:stretch;}}
.hero-col{{padding:38px 40px;display:flex;flex-direction:column;justify-content:center;}}
.hero-col.ctr{{text-align:center;align-items:center;}}

.hero-eyebrow{{
  font-family:'Space Mono',monospace;font-size:9px;font-weight:700;
  letter-spacing:3.5px;text-transform:uppercase;color:var(--teal);
  margin-bottom:14px;opacity:0.85;
}}
.hero-dir{{
  font-size:80px;font-weight:800;line-height:0.85;
  letter-spacing:-4px;font-family:'Sora',sans-serif;
}}
.hero-dir.up{{color:var(--green);text-shadow:0 0 60px rgba(34,211,160,0.25);}}
.hero-dir.dn{{color:var(--red);text-shadow:0 0 60px rgba(255,77,109,0.25);}}
.hero-arrow{{font-size:28px;margin-top:10px;}}
.hero-sub{{
  font-size:11px;color:var(--muted2);margin-top:8px;
  font-family:'Space Mono',monospace;letter-spacing:0.5px;
}}

.price-label{{
  font-size:9px;font-family:'Space Mono',monospace;
  letter-spacing:2.5px;text-transform:uppercase;
  color:var(--muted2);margin-bottom:8px;
}}
.price-from{{
  font-family:'Space Mono',monospace;font-size:13px;
  color:var(--muted2);margin-bottom:5px;
}}
.price-big{{
  font-family:'Sora',sans-serif;font-size:46px;font-weight:800;
  color:var(--text);letter-spacing:-2px;line-height:1;
}}
.price-chg{{
  font-family:'Space Mono',monospace;font-size:15px;
  font-weight:700;margin-top:10px;
}}
.price-chg.pos{{color:var(--green);}} .price-chg.neg{{color:var(--red);}}

.conf-label{{
  font-size:9px;font-family:'Space Mono',monospace;
  letter-spacing:2.5px;text-transform:uppercase;
  color:var(--muted2);margin-bottom:8px;
}}
.conf-num{{
  font-family:'Sora',sans-serif;font-size:58px;font-weight:800;
  color:var(--gold);letter-spacing:-3px;line-height:1;
}}
.conf-pct{{font-size:24px;letter-spacing:0;}}
.conf-bar{{width:100%;height:5px;background:var(--border);border-radius:3px;margin-top:16px;overflow:hidden;}}
.conf-fill{{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--gold),#ff9500);}}
.conf-ticks{{
  display:flex;justify-content:space-between;margin-top:6px;
  font-family:'Space Mono',monospace;font-size:8.5px;color:var(--muted);
}}

.algo-strip{{
  border-top:1px solid var(--border);
  padding:16px 40px;
  display:flex;align-items:center;gap:28px;flex-wrap:wrap;
  background:rgba(0,229,180,0.02);
  position:relative;z-index:1;
}}
.as-label{{
  font-family:'Space Mono',monospace;font-size:8.5px;
  letter-spacing:2.5px;text-transform:uppercase;
  color:var(--muted);margin-bottom:4px;
}}
.as-val{{font-size:14px;font-weight:700;color:var(--teal);
  font-family:'Sora',sans-serif;letter-spacing:-0.2px;}}
.as-sep{{width:1px;height:30px;background:var(--border);flex-shrink:0;}}
.as-pill{{
  display:inline-flex;align-items:center;gap:7px;
  background:rgba(0,229,180,0.08);
  border:1px solid rgba(0,229,180,0.25);
  border-radius:20px;padding:5px 14px;
  font-family:'Space Mono',monospace;font-size:11px;color:var(--teal);
}}
.as-pill-dot{{
  width:5px;height:5px;border-radius:50%;
  background:var(--teal);animation:pulse 2s infinite;
}}

/* ════ SECTION HEADER ════ */
.sec-hd{{
  display:flex;align-items:center;gap:13px;
  margin:36px 0 20px;
}}
.sec-icon{{
  width:30px;height:30px;
  background:rgba(0,229,180,0.08);
  border:1px solid rgba(0,229,180,0.2);
  border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:14px;flex-shrink:0;
}}
.sec-txt{{}}
.sec-title{{font-size:16px;font-weight:700;color:var(--text);font-family:'Sora',sans-serif;letter-spacing:-0.2px;}}
.sec-sub{{font-size:11px;color:var(--muted2);font-family:'Space Mono',monospace;margin-top:2px;}}
.sec-hd::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}}

/* ════ METRIC CARDS ════ */
.mc{{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;padding:24px 22px 20px;
  position:relative;overflow:hidden;
  transition:border-color .2s,transform .2s,box-shadow .2s;
}}
.mc:hover{{border-color:var(--border2);transform:translateY(-3px);box-shadow:0 8px 30px rgba(0,0,0,0.3);}}
.mc::before{{content:'';position:absolute;top:0;left:0;right:0;height:2.5px;border-radius:16px 16px 0 0;}}
.mc.blue::before{{background:linear-gradient(90deg,var(--blue),#7eb3ff);}}
.mc.green::before{{background:linear-gradient(90deg,var(--green),#6ef5d0);}}
.mc.amber::before{{background:linear-gradient(90deg,var(--gold),var(--gold2));}}
.mc.purple::before{{background:linear-gradient(90deg,var(--purple),#d4b8ff);}}
.mc-icon{{font-size:20px;margin-bottom:12px;display:block;opacity:0.9;}}
.mc-val{{
  font-family:'Sora',sans-serif;font-size:34px;font-weight:800;
  letter-spacing:-1.5px;line-height:1;
}}
.mc.blue .mc-val{{color:var(--blue);}} .mc.green .mc-val{{color:var(--green);}}
.mc.amber .mc-val{{color:var(--gold);}} .mc.purple .mc-val{{color:var(--purple);}}
.mc-lbl{{
  font-size:9px;font-weight:700;letter-spacing:2.5px;
  text-transform:uppercase;color:var(--muted2);margin-top:8px;
  font-family:'Space Mono',monospace;
}}
.mc-bar{{height:3px;border-radius:2px;background:var(--border);margin-top:14px;overflow:hidden;}}
.mc-bar-fill{{height:100%;border-radius:2px;}}
.mc.blue .mc-bar-fill{{background:var(--blue);}} .mc.green .mc-bar-fill{{background:var(--green);}}
.mc.amber .mc-bar-fill{{background:var(--gold);}} .mc.purple .mc-bar-fill{{background:var(--purple);}}
.mc-desc{{font-size:11px;color:var(--muted2);margin-top:7px;font-family:'Space Mono',monospace;line-height:1.5;}}

/* ════ MODEL CARDS ════ */
.model-card{{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:26px;height:100%;
  transition:border-color .2s;
}}
.model-card:hover{{border-color:var(--border2);}}
.mc2-badge{{
  display:inline-flex;align-items:center;gap:7px;
  background:rgba(0,229,180,0.07);
  border:1px solid rgba(0,229,180,0.2);
  border-radius:7px;padding:4px 12px;
  font-family:'Space Mono',monospace;font-size:8.5px;
  letter-spacing:2.5px;text-transform:uppercase;
  color:var(--teal);margin-bottom:16px;
}}
.mc2-name{{
  font-size:20px;font-weight:700;color:var(--text);
  font-family:'Sora',sans-serif;letter-spacing:-0.3px;
  margin-bottom:18px;line-height:1.25;
}}
.mc2-row{{
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 0;border-bottom:1px solid var(--border);font-size:13px;
}}
.mc2-row:last-child{{border-bottom:none;}}
.mc2-k{{
  color:var(--muted2);font-family:'Space Mono',monospace;
  font-size:9.5px;letter-spacing:0.5px;text-transform:uppercase;
}}
.mc2-v{{font-weight:700;color:var(--text);font-family:'Space Mono',monospace;font-size:12.5px;}}

/* ════ REASONING BOX ════ */
.reasoning-box{{
  background:var(--card);
  border:1px solid var(--border);
  border-left:3px solid var(--teal);
  border-radius:0 14px 14px 0;
  padding:22px 24px;
  font-size:13px;color:var(--text2);line-height:1.85;
  font-family:'Inter',sans-serif;
}}

/* ════ CONFUSION MATRIX ════ */
.cm-item{{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:15px 18px;margin-bottom:11px;
}}
.cmi-label{{
  font-size:9px;color:var(--muted2);font-family:'Space Mono',monospace;
  letter-spacing:1px;margin-bottom:5px;text-transform:uppercase;
}}
.cmi-row{{display:flex;align-items:center;justify-content:space-between;}}
.cmi-desc{{font-size:13px;font-weight:600;color:var(--text2);}}
.cmi-val{{font-family:'Sora',sans-serif;font-size:30px;font-weight:800;letter-spacing:-1px;}}
.cmi-prog{{height:3px;border-radius:2px;background:var(--border);margin-top:9px;overflow:hidden;}}
.cmi-fill{{height:100%;border-radius:2px;}}
.acc-box{{
  background:linear-gradient(135deg,rgba(34,211,160,0.07),rgba(0,229,180,0.05));
  border:1px solid rgba(34,211,160,0.25);
  border-radius:16px;padding:26px;text-align:center;margin-top:18px;
}}
.ab-label{{
  font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;
  text-transform:uppercase;color:var(--green);margin-bottom:10px;
}}
.ab-val{{
  font-family:'Sora',sans-serif;font-size:46px;font-weight:800;
  color:var(--green);letter-spacing:-2px;line-height:1;
}}
.ab-sub{{font-size:11.5px;color:rgba(34,211,160,0.6);margin-top:7px;font-family:'Space Mono',monospace;}}

/* ════ EMPTY STATE ════ */
.empty-state{{text-align:center;padding:100px 20px;}}
.empty-state .es-icon{{font-size:64px;margin-bottom:24px;display:block;}}
.empty-state h2{{
  font-size:30px;font-weight:800;color:var(--text);
  font-family:'Sora',sans-serif;letter-spacing:-0.8px;margin-bottom:12px;
}}
.empty-state p{{
  font-size:14px;color:var(--muted2);line-height:1.75;
  max-width:440px;margin:0 auto;font-family:'Space Mono',monospace;
}}
.es-steps{{display:flex;gap:10px;justify-content:center;margin-top:32px;flex-wrap:wrap;}}
.es-step{{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:12px 20px;
  font-size:12.5px;font-weight:600;color:var(--text2);
  font-family:'Inter',sans-serif;
}}
.es-step span{{
  display:block;font-family:'Space Mono',monospace;
  font-size:9px;color:var(--teal);margin-bottom:4px;letter-spacing:1px;
}}

/* ════ FOOTER ════ */
.footer{{
  border-top:1px solid var(--border);
  padding:20px 0;margin-top:50px;
  display:flex;align-items:center;justify-content:center;
}}
.ft-brand{{
  font-family:'Sora',sans-serif;font-size:13px;
  font-weight:700;color:var(--muted2);letter-spacing:-0.2px;
}}
.ft-brand span{{color:var(--teal);}}

/* ════ READY BANNER ════ */
.ready-banner{{
  margin:20px 0;padding:22px 30px;
  background:linear-gradient(135deg,rgba(0,229,180,0.07),rgba(79,142,247,0.05));
  border:1.5px solid rgba(0,229,180,0.3);border-radius:16px;
  display:flex;align-items:center;justify-content:space-between;gap:20px;
}}
.rb-text .rb-eyebrow{{
  font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;
  text-transform:uppercase;color:var(--teal);margin-bottom:7px;
}}
.rb-text .rb-title{{font-size:16px;font-weight:700;color:var(--text);margin-bottom:5px;font-family:'Sora',sans-serif;}}
.rb-text .rb-hint{{font-family:'Space Mono',monospace;font-size:11.5px;color:var(--muted2);}}
.rb-icon{{font-size:44px;opacity:0.5;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── CHART THEME ──────────────────────────────────────────
def chart_style():
    plt.rcParams.update({
        'figure.facecolor':   CARD,
        'axes.facecolor':     CARD,
        'axes.edgecolor':     BORDER,
        'axes.labelcolor':    MUTED2,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'text.color':         TEXT2,
        'xtick.color':        MUTED,
        'ytick.color':        MUTED,
        'grid.color':         BORDER,
        'grid.linewidth':     0.5,
        'grid.linestyle':     '--',
        'font.family':        'monospace',
        'font.size':          10,
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'axes.titlecolor':    TEXT,
        'figure.dpi':         115,
    })


# ─── HELPERS ──────────────────────────────────────────────
def md_bold(text):
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#edf2f7;">\1</strong>', text)
    return text.replace('\n', '<br>')

def H(tag, cls, content, style=""):
    s = f' style="{style}"' if style else ""
    return f'<{tag} class="{cls}"{s}>{content}</{tag}>'


# ─── SESSION STATE ────────────────────────────────────────
for k in ['data_loaded','models_trained','raw_df','processed_df',
          'feature_cols','X','y_clf','y_reg','stock_info',
          'clf_results','reg_results','clf_metrics','reg_metrics',
          'best_clf','best_reg','clf_scaler','reg_scaler',
          'X_test_clf','y_test_clf','X_test_reg','y_test_reg',
          'ticker','saved_paths','prediction']:
    if k not in st.session_state: st.session_state[k] = None
for k in ('data_loaded','models_trained'):
    if not st.session_state[k]: st.session_state[k] = False


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-logo-row">
        <div class="sb-logo">◈</div>
        <div>
          <div class="sb-name">StockSense</div>
        </div>
      </div>
      <div class="sb-sub">ML Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Instrument</div>', unsafe_allow_html=True)
    ticker_input = st.text_input("", value="AAPL",
                                  placeholder="AAPL · TSLA · INFY.NS",
                                  label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Date Range</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("From", value=pd.to_datetime("2020-01-01"))
    with c2: end_date   = st.date_input("To",   value=pd.to_datetime("2024-12-31"))

    st.markdown('<div class="sb-sec">Test Split</div>', unsafe_allow_html=True)
    test_pct = st.slider("", 10, 40, 20, 5, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Actions</div>', unsafe_allow_html=True)
    fetch_btn = st.button("⬇  Fetch Market Data",  use_container_width=True)
    train_btn = st.button("⚡  Train All 17 Models", use_container_width=True,
                          disabled=not st.session_state.data_loaded)
    save_btn  = st.button("💾  Save Models",         use_container_width=True,
                          disabled=not st.session_state.models_trained)

    if st.session_state.models_trained:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        tkr = st.session_state.ticker or ""
        st.markdown(f"""
        <div class="sb-status">
          <div class="sb-dot"></div>
          <span>Ready · {tkr}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Models</div>', unsafe_allow_html=True)
    with st.expander("9 Classifiers"):
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:10.5px;
        color:#7a9ab5;line-height:2.2;">Logistic Regression<br>Decision Tree<br>Random Forest<br>
        K-Nearest Neighbors<br>Support Vector Machine<br>Naive Bayes<br>AdaBoost<br>
        Gradient Boosting<br>XGBoost</div>""", unsafe_allow_html=True)
    with st.expander("8 Regressors"):
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:10.5px;
        color:#7a9ab5;line-height:2.2;">Linear Regression<br>Decision Tree Reg.<br>
        Random Forest Reg.<br>KNN Regressor<br>Support Vector Reg.<br>AdaBoost Reg.<br>
        Gradient Boosting Reg.<br>XGBoost Regressor</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  FETCH
# ════════════════════════════════════════════════════════════
if fetch_btn:
    ticker = ticker_input.strip().upper()
    st.session_state.ticker = ticker
    st.session_state.models_trained = False
    st.session_state.prediction = None
    with st.spinner(f"Fetching market data for {ticker}…"):
        try:
            raw_df  = fetch_stock_data(ticker, str(start_date), str(end_date))
            info    = get_stock_info(ticker)
            proc_df, feat_cols, X, y_clf, y_reg = preprocess_data(raw_df)
            st.session_state.update({
                'raw_df':raw_df,'processed_df':proc_df,'feature_cols':feat_cols,
                'X':X,'y_clf':y_clf,'y_reg':y_reg,'stock_info':info,'data_loaded':True,
            })
            st.success(f"✓  {len(raw_df):,} trading days loaded for **{ticker}**")
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.data_loaded = False


# ════════════════════════════════════════════════════════════
#  TRAIN
# ════════════════════════════════════════════════════════════
if train_btn and st.session_state.data_loaded:
    with st.spinner("Training 17 ML models — approx 60–90 seconds…"):
        try:
            X  = st.session_state.X
            ts = test_pct / 100.0
            Xtr_c,Xte_c,ytr_c,yte_c,clf_sc = prepare_train_test_split(X, st.session_state.y_clf, ts)
            Xtr_r,Xte_r,ytr_r,yte_r,reg_sc = prepare_train_test_split(X, st.session_state.y_reg, ts)
            clf_res = train_all_classification_models(Xtr_c,Xte_c,ytr_c,yte_c,st.session_state.feature_cols)
            reg_res = train_all_regression_models(Xtr_r,Xte_r,ytr_r,yte_r,st.session_state.feature_cols)
            clf_met  = evaluate_classification_models(clf_res, yte_c)
            reg_met  = evaluate_regression_models(reg_res, yte_r)
            best_clf = select_best_classification_model(clf_met, clf_res)
            best_reg = select_best_regression_model(reg_met, reg_res)
            pred = predict_next_day(best_clf['model'],best_reg['model'],clf_sc,reg_sc,
                                    X.iloc[-1].values,best_clf['name'],best_reg['name'])
            st.session_state.update({
                'clf_results':clf_res,'reg_results':reg_res,
                'clf_metrics':clf_met,'reg_metrics':reg_met,
                'best_clf':best_clf,'best_reg':best_reg,
                'clf_scaler':clf_sc,'reg_scaler':reg_sc,
                'X_test_clf':Xte_c,'y_test_clf':yte_c,
                'X_test_reg':Xte_r,'y_test_reg':yte_r,
                'prediction':pred,'models_trained':True,
            })
            st.success("✓  All 17 models trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback; st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════
if save_btn and st.session_state.models_trained:
    with st.spinner("Saving models to disk…"):
        try:
            p = save_models(
                st.session_state.clf_results, st.session_state.reg_results,
                st.session_state.clf_scaler, st.session_state.reg_scaler,
                st.session_state.ticker)
            st.session_state.saved_paths = p
            st.sidebar.success(f"✓  {len(p)} model files saved")
        except Exception as e:
            st.sidebar.error(str(e))


# ════════════════════════════════════════════════════════════
#  TOP BAR
# ════════════════════════════════════════════════════════════
info     = st.session_state.stock_info or {}
is_live  = st.session_state.models_trained
sector   = info.get('sector','ML System')
cur_code = info.get('currency', '')
cur_sym  = get_currency_symbol(cur_code) if cur_code else ''
cur_badge = f"{cur_sym} {cur_code}" if cur_code else ''

live_dot = '<span class="tb-live-dot"></span>' if is_live else ''
live_cls = 'live' if is_live else ''
status   = 'Models Trained' if is_live else 'Awaiting Training'

topbar_html = f"""
<div class="topbar">
  <div class="tb-left">
    <div class="tb-h1">Stock Market Prediction</div>
    <div class="tb-sub">SUPERVISED ML &nbsp;·&nbsp; 17 ALGORITHMS &nbsp;·&nbsp; AUTO MODEL SELECTION</div>
  </div>
  <div class="tb-badges">
    <span class="tb-badge">{sector}</span>
    {'<span class="tb-badge">' + cur_badge + '</span>' if cur_badge else ''}
    <span class="tb-badge {live_cls}">{live_dot}{status}</span>
  </div>
</div>
"""
st.markdown(topbar_html, unsafe_allow_html=True)


# ─── EMPTY STATE ──────────────────────────────────────────
if not st.session_state.data_loaded:
    st.markdown("""
    <div class="empty-state">
      <span class="es-icon">◈</span>
      <h2>Ready to Analyze</h2>
      <p>Enter a stock ticker, pick your date range, fetch data, then train all 17 ML models.</p>
      <div class="es-steps">
        <div class="es-step"><span>01</span>Enter Ticker</div>
        <div class="es-step"><span>02</span>Set Date Range</div>
        <div class="es-step"><span>03</span>Fetch Data</div>
        <div class="es-step"><span>04</span>Train Models</div>
        <div class="es-step"><span>05</span>View Results</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── DATA LOADED — NOT YET TRAINED ───────────────────────
if not st.session_state.models_trained:
    raw_df  = st.session_state.raw_df
    proc_df = st.session_state.processed_df
    ticker  = st.session_state.ticker
    n_days  = len(raw_df)
    n_feat  = len(st.session_state.feature_cols)

    header_html = f"""
    <div style="text-align:center;padding:28px 0 22px;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3.5px;
                  text-transform:uppercase;color:{TEAL};margin-bottom:10px;">DATA LOADED</div>
      <div style="font-size:42px;font-weight:800;color:{TEXT};letter-spacing:-2px;
                  margin-bottom:8px;font-family:'Sora',sans-serif;">{ticker}</div>
      <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED2};">
        {n_days:,} trading days &nbsp;·&nbsp; {n_feat} engineered features
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    chart_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5),
                                   gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(proc_df.index, proc_df['Close'], color=TEAL, lw=1.8, label='Close', zorder=3)
    if 'SMA_20' in proc_df.columns:
        ax1.plot(proc_df.index, proc_df['SMA_20'], color=GOLD,   lw=1, ls='--', alpha=.8, label='SMA 20')
    if 'SMA_50' in proc_df.columns:
        ax1.plot(proc_df.index, proc_df['SMA_50'], color=PURPLE, lw=1, ls='--', alpha=.8, label='SMA 50')
    if 'BB_Upper' in proc_df.columns:
        ax1.fill_between(proc_df.index, proc_df['BB_Lower'], proc_df['BB_Upper'], alpha=.07, color=TEAL)
    ax1.set_ylabel('Price'); ax1.legend(fontsize=9, framealpha=.2, facecolor=CARD)
    ax1.grid(True, alpha=.2); ax1.tick_params(labelbottom=False)
    ax1.set_title(f'{ticker}  ·  Price History & Bollinger Bands', pad=10)
    vc = [GREEN if proc_df['Close'].iloc[i] >= proc_df['Open'].iloc[i]
          else RED for i in range(len(proc_df))]
    ax2.bar(proc_df.index, proc_df['Volume'], color=vc, alpha=.6, width=.8)
    ax2.set_ylabel('Volume'); ax2.grid(True, alpha=.2)
    fig.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    ready_html = f"""
    <div class="ready-banner">
      <div class="rb-text">
        <div class="rb-eyebrow">✓ Data Ready</div>
        <div class="rb-title">{n_days:,} days loaded &nbsp;·&nbsp; {n_feat} features engineered</div>
        <div class="rb-hint">Click <strong style="color:{TEAL};">⚡ Train All 17 Models</strong> in the sidebar to begin (~60–90 seconds)</div>
      </div>
      <div class="rb-icon">⚡</div>
    </div>
    """
    st.markdown(ready_html, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════
#  FULL RESULTS
# ════════════════════════════════════════════════════════════
ticker     = st.session_state.ticker
pred       = st.session_state.prediction
best_clf   = st.session_state.best_clf
best_reg   = st.session_state.best_reg
clf_met    = st.session_state.clf_metrics
reg_met    = st.session_state.reg_metrics
clf_res    = st.session_state.clf_results
reg_res    = st.session_state.reg_results
raw_df     = st.session_state.raw_df
proc_df    = st.session_state.processed_df
y_test_clf = st.session_state.y_test_clf
y_test_reg = st.session_state.y_test_reg

current_price = float(raw_df['Close'].iloc[-1])
pred_price    = pred['predicted_price']
is_up         = pred['direction_value'] == 1
price_change  = pred_price - current_price
pct_change    = price_change / current_price * 100
conf          = pred.get('confidence') or 0.0
dir_word      = "UP" if is_up else "DOWN"
dir_css       = "up" if is_up else "dn"
dir_col       = GREEN if is_up else RED
chg_css       = "pos" if price_change >= 0 else "neg"
chg_sign      = "+" if price_change >= 0 else ""
arrow         = "▲" if is_up else "▼"

# ─── CURRENCY SYMBOL (dynamic based on actual stock currency) ──
_info     = st.session_state.stock_info or {}
_currency = _info.get('currency', 'USD')
CUR       = get_currency_symbol(_currency)   # e.g. "$" for AAPL, "₹" for INFY.NS

acc  = best_clf['metrics']['Accuracy']
f1   = best_clf['metrics']['F1-Score']
prec = best_clf['metrics']['Precision']
rec  = best_clf['metrics']['Recall']
r2   = best_reg['metrics']['R² Score']
rmse = best_reg['metrics']['RMSE']
mae  = best_reg['metrics']['MAE']
mape = best_reg['metrics']['MAPE (%)']

# ─── HERO — pre-built string to avoid f-string HTML rendering bug ──
conf_fill_w  = min(conf, 100)
price_change_abs = abs(price_change)
pct_change_abs   = abs(pct_change)

hero_html = (
    '<div class="hero">'
    '  <div class="hero-grid">'

    # Column 1 — Direction
    '    <div class="hero-col ctr">'
    f'      <div class="hero-eyebrow">◈ {ticker} &nbsp;·&nbsp; NEXT SESSION FORECAST</div>'
    f'      <div class="hero-dir {dir_css}">{dir_word}</div>'
    f'      <div class="hero-arrow" style="color:{dir_col};">{arrow}</div>'
    '       <div class="hero-sub">Movement Direction</div>'
    '    </div>'

    '    <div class="hero-divider"></div>'

    # Column 2 — Price
    '    <div class="hero-col">'
    '      <div class="price-label">Predicted Closing Price</div>'
    f'     <div class="price-from">Current &nbsp;→&nbsp; {CUR}{current_price:.2f}</div>'
    f'     <div class="price-big">{CUR}{pred_price:.2f}</div>'
    f'     <div class="price-chg {chg_css}">{chg_sign}{CUR}{price_change:.2f} &nbsp;({chg_sign}{pct_change:.2f}%)</div>'
    '    </div>'

    '    <div class="hero-divider"></div>'

    # Column 3 — Confidence
    '    <div class="hero-col">'
    '      <div class="conf-label">Model Confidence</div>'
    f'     <div class="conf-num">{conf:.1f}<span class="conf-pct">%</span></div>'
    f'     <div class="conf-bar"><div class="conf-fill" style="width:{conf_fill_w:.0f}%;"></div></div>'
    '      <div class="conf-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>'
    '    </div>'

    '  </div>'

    # Algo strip
    '  <div class="algo-strip">'
    f'    <div><div class="as-label">Best Classifier</div><div class="as-val">{best_clf["name"]}</div></div>'
    '     <div class="as-sep"></div>'
    f'    <div><div class="as-label">F1-Score</div><div class="as-val" style="color:{GOLD};">{f1:.4f}</div></div>'
    '     <div class="as-sep"></div>'
    f'    <div><div class="as-label">Best Regressor</div><div class="as-val">{best_reg["name"]}</div></div>'
    '     <div class="as-sep"></div>'
    f'    <div><div class="as-label">R² Score</div><div class="as-val" style="color:{PURPLE};">{r2:.4f}</div></div>'
    '     <div class="as-sep"></div>'
    '     <div class="as-pill"><div class="as-pill-dot"></div>17 Models Evaluated</div>'
    '  </div>'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────
tabs = st.tabs([
    "◈  Prediction & Metrics",
    "⬡  Classification",
    "⬡  Confusion Matrix",
    "⬡  Regression",
    "⬡  All Models",
    "⬡  Stock Charts",
])


# ════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION & METRICS
# ════════════════════════════════════════════════════════════
with tabs[0]:

    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">📊</div>
    <div class="sec-txt">
      <div class="sec-title">Classification Performance</div>
      <div class="sec-sub">Best model · Test set evaluation</div>
    </div></div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ('blue',   '◎', f'{acc*100:.1f}%',  'ACCURACY',  int(acc*100),  f'{acc*100:.1f}% of all predictions correct'),
        ('green',  '⬡', f'{f1*100:.1f}%',   'F1-SCORE',  int(f1*100),   'Harmonic mean of precision & recall'),
        ('amber',  '◇', f'{prec*100:.1f}%', 'PRECISION', int(prec*100), f'{prec*100:.1f}% of UP calls were correct'),
        ('purple', '△', f'{rec*100:.1f}%',  'RECALL',    int(rec*100),  f'Captured {rec*100:.1f}% of actual UP moves'),
    ]
    for col, (cls, icon, val, lbl, pct, desc) in zip(cols, cards):
        with col:
            card_html = (
                f'<div class="mc {cls}">'
                f'  <span class="mc-icon">{icon}</span>'
                f'  <div class="mc-val">{val}</div>'
                f'  <div class="mc-lbl">{lbl}</div>'
                f'  <div class="mc-bar"><div class="mc-bar-fill" style="width:{pct}%;"></div></div>'
                f'  <div class="mc-desc">{desc}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">🤖</div>
    <div class="sec-txt">
      <div class="sec-title">Selected Algorithms</div>
      <div class="sec-sub">Auto-selected · Highest F1 and R²</div>
    </div></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        clf_card = (
            '<div class="model-card">'
            '  <div class="mc2-badge">◎ CLASSIFICATION MODEL</div>'
            f' <div class="mc2-name">{best_clf["name"]}</div>'
            f' <div class="mc2-row"><span class="mc2-k">PREDICTION</span>'
            f'   <span class="mc2-v" style="color:{dir_col};">{arrow} {dir_word}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">CONFIDENCE</span>'
            f'   <span class="mc2-v" style="color:{GOLD};">{conf:.1f}%</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">ACCURACY</span>'
            f'   <span class="mc2-v">{acc:.4f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">F1-SCORE</span>'
            f'   <span class="mc2-v">{f1:.4f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">PRECISION</span>'
            f'   <span class="mc2-v">{prec:.4f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">RECALL</span>'
            f'   <span class="mc2-v">{rec:.4f}</span></div>'
            '</div>'
        )
        st.markdown(clf_card, unsafe_allow_html=True)

    with c2:
        reg_card = (
            '<div class="model-card">'
            '  <div class="mc2-badge">◇ REGRESSION MODEL</div>'
            f' <div class="mc2-name">{best_reg["name"]}</div>'
            f' <div class="mc2-row"><span class="mc2-k">PREDICTED PRICE</span>'
            f'   <span class="mc2-v" style="color:{TEAL};">{CUR}{pred_price:.2f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">EXPECTED CHANGE</span>'
            f'   <span class="mc2-v" style="color:{dir_col};">{chg_sign}{CUR}{price_change:.2f} ({chg_sign}{pct_change:.2f}%)</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">R² SCORE</span>'
            f'   <span class="mc2-v">{r2:.4f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">RMSE</span>'
            f'   <span class="mc2-v">{rmse:.4f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">MAE</span>'
            f'   <span class="mc2-v">{mae:.4f}</span></div>'
            f' <div class="mc2-row"><span class="mc2-k">MAPE</span>'
            f'   <span class="mc2-v">{mape:.2f}%</span></div>'
            '</div>'
        )
        st.markdown(reg_card, unsafe_allow_html=True)

    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">💡</div>
    <div class="sec-txt">
      <div class="sec-title">Why These Models Were Selected</div>
      <div class="sec-sub">Automatic selection rationale</div>
    </div></div>""", unsafe_allow_html=True)

    r1, r2_ = st.columns(2)
    with r1:
        st.markdown(f'<div class="reasoning-box">{md_bold(best_clf["reasoning"])}</div>',
                    unsafe_allow_html=True)
    with r2_:
        st.markdown(f'<div class="reasoning-box">{md_bold(best_reg["reasoning"])}</div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — CLASSIFICATION
# ════════════════════════════════════════════════════════════
with tabs[1]:
    chart_style()
    best_name = best_clf['name']
    models    = clf_met['Model'].tolist()
    x         = np.arange(len(models))
    w         = 0.2
    best_idx  = clf_met[clf_met['Model'] == best_name].index[0] - 1

    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">📊</div>
    <div class="sec-txt">
      <div class="sec-title">All Metrics — 9 Classification Models</div>
      <div class="sec-sub">Accuracy · Precision · Recall · F1-Score</div>
    </div></div>""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (color, metric) in enumerate([(BLUE,'Accuracy'),(GOLD,'Precision'),(PURPLE,'Recall'),(TEAL,'F1-Score')]):
        vals = clf_met[metric].values
        bars = ax.bar(x + i*w, vals, w, label=metric, color=color, alpha=0.85, zorder=3, linewidth=0)
        bars[best_idx].set_linewidth(2)
        bars[best_idx].set_edgecolor(GOLD2)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    ax.set_ylim(0, 1.22); ax.set_ylabel('Score', fontsize=10)
    ax.axhline(0.5, color=RED, ls='--', lw=0.8, alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.15, facecolor=CARD, edgecolor=BORDER, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y', zorder=0)
    ax.set_title(f'Classification Performance  ·  ★ Best: {best_name}', pad=10)
    ax.annotate('★ Best', xy=(best_idx + w*2.5 + w, clf_met.iloc[best_idx]['F1-Score'] + 0.05),
                ha='center', color=GOLD2, fontsize=9, fontweight='bold')
    fig.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="sec-hd" style="margin-top:20px;"><div class="sec-icon">⬡</div>
        <div class="sec-txt"><div class="sec-title">F1-Score Ranking</div></div></div>
        """, unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        clrs2 = [GOLD if m == best_name else BLUE for m in models]
        brs2  = ax2.barh(models, clf_met['F1-Score'].values, color=clrs2, alpha=0.85, height=0.55, linewidth=0)
        for bar, val in zip(brs2, clf_met['F1-Score'].values):
            ax2.text(bar.get_width()+.006, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9, color=TEXT)
        ax2.set_xlim(0, 1.12); ax2.invert_yaxis()
        ax2.grid(True, alpha=0.2, axis='x')
        ax2.set_title('F1-Score · Ranked', pad=8)
        ax2.legend(handles=[mpatches.Patch(color=GOLD, label='Best'), mpatches.Patch(color=BLUE, label='Others')],
                   fontsize=9, framealpha=.2, facecolor=CARD)
        fig2.patch.set_facecolor(CARD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with c2:
        st.markdown("""
        <div class="sec-hd" style="margin-top:20px;"><div class="sec-icon">◎</div>
        <div class="sec-txt"><div class="sec-title">Accuracy Ranking</div></div></div>
        """, unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        clrs3 = [GREEN if m == best_name else TEAL for m in models]
        brs3  = ax3.barh(models, clf_met['Accuracy'].values, color=clrs3, alpha=0.85, height=0.55, linewidth=0)
        for bar, val in zip(brs3, clf_met['Accuracy'].values):
            ax3.text(bar.get_width()+.006, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9, color=TEXT)
        ax3.set_xlim(0, 1.12); ax3.invert_yaxis()
        ax3.grid(True, alpha=0.2, axis='x')
        ax3.set_title('Accuracy · Ranked', pad=8)
        fig3.patch.set_facecolor(CARD)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">△</div>
    <div class="sec-txt">
      <div class="sec-title">Precision vs Recall</div>
      <div class="sec-sub">Per-model breakdown</div>
    </div></div>""", unsafe_allow_html=True)

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax_i, metric, color, title in [
        (axes4[0], 'Precision', GOLD,   'Precision per Model'),
        (axes4[1], 'Recall',    PURPLE, 'Recall per Model'),
    ]:
        vals  = clf_met[metric].values
        clrs4 = [PINK if m == best_name else color for m in models]
        brs4  = ax_i.barh(models, vals, color=clrs4, alpha=0.85, height=0.55, linewidth=0)
        for bar, val in zip(brs4, vals):
            ax_i.text(bar.get_width()+.005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9, color=TEXT)
        ax_i.set_xlim(0, 1.12); ax_i.invert_yaxis()
        ax_i.grid(True, alpha=0.2, axis='x')
        ax_i.set_title(title, pad=8)
    fig4.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    buf = io.StringIO(); clf_met.to_csv(buf, index=False)
    st.download_button("⬇  Download Classification Metrics CSV", buf.getvalue(), f"{ticker}_clf_metrics.csv", "text/csv")


# ════════════════════════════════════════════════════════════
#  TAB 3 — CONFUSION MATRIX
# ════════════════════════════════════════════════════════════
with tabs[2]:
    chart_style()

    st.markdown(f"""
    <div class="sec-hd"><div class="sec-icon">⬡</div>
    <div class="sec-txt">
      <div class="sec-title">Confusion Matrix</div>
      <div class="sec-sub">Best model: {best_clf['name']}</div>
    </div></div>""", unsafe_allow_html=True)

    cm     = get_confusion_matrix(clf_res, y_test_clf, best_clf['name'])
    tn, fp, fn, tp = cm.ravel()
    total  = cm.sum()
    cm_acc = (tn + tp) / total

    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5.5))
        cmap_cm = mcolors.LinearSegmentedColormap.from_list('ss', [CARD, '#003328', TEAL], N=256)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_cm,
                    xticklabels=['DOWN (0)', 'UP (1)'], yticklabels=['DOWN (0)', 'UP (1)'],
                    ax=ax_cm, linewidths=2, linecolor=BG,
                    annot_kws={'size':22,'weight':'bold','color':TEXT})
        ax_cm.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
        ax_cm.set_ylabel('True Label',      fontsize=11, labelpad=10)
        ax_cm.set_title(f'{best_clf["name"]}\nConfusion Matrix', pad=12)
        fig_cm.patch.set_facecolor(CARD)
        ax_cm.set_facecolor(CARD)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with c2:
        for label, val, color, short, desc in [
            ('True Negatives',  tn, BLUE,   'TN', 'Correctly predicted DOWN'),
            ('False Positives', fp, RED,    'FP', 'Predicted UP — actual was DOWN'),
            ('False Negatives', fn, GOLD,   'FN', 'Predicted DOWN — actual was UP'),
            ('True Positives',  tp, GREEN,  'TP', 'Correctly predicted UP'),
        ]:
            pct = val/total*100
            item_html = (
                '<div class="cm-item">'
                f'  <div class="cmi-label">{label} ({short})</div>'
                '  <div class="cmi-row">'
                f'    <div class="cmi-desc">{desc}</div>'
                f'    <div class="cmi-val" style="color:{color};">{val}</div>'
                '  </div>'
                f'  <div class="cmi-prog"><div class="cmi-fill" style="width:{pct:.0f}%;background:{color};"></div></div>'
                '</div>'
            )
            st.markdown(item_html, unsafe_allow_html=True)

        acc_html = (
            '<div class="acc-box">'
            '  <div class="ab-label">Overall Accuracy</div>'
            f' <div class="ab-val">{cm_acc:.2%}</div>'
            f' <div class="ab-sub">{tn+tp} correct out of {total} predictions</div>'
            '</div>'
        )
        st.markdown(acc_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-hd" style="margin-top:34px;"><div class="sec-icon">⬡</div>
    <div class="sec-txt">
      <div class="sec-title">All 9 Models — Confusion Matrices</div>
      <div class="sec-sub">3-column grid overview</div>
    </div></div>""", unsafe_allow_html=True)

    all_mods  = clf_met['Model'].tolist()
    ncols_g, nrows_g = 3, (len(all_mods) + 2) // 3
    fig_g, axes_g = plt.subplots(nrows_g, ncols_g, figsize=(14, nrows_g * 4))
    flat = axes_g.flatten() if hasattr(axes_g, 'flatten') else [axes_g]
    cmap_mini = mcolors.LinearSegmentedColormap.from_list('mini', [CARD, '#001f18', TEAL], N=256)

    for i, m in enumerate(all_mods):
        ax = flat[i]
        try:
            cm_i = get_confusion_matrix(clf_res, y_test_clf, m)
            sns.heatmap(cm_i, annot=True, fmt='d', cmap=cmap_mini,
                        xticklabels=['DN','UP'], yticklabels=['DN','UP'],
                        ax=ax, linewidths=1.5, linecolor=BG,
                        annot_kws={'size':13,'weight':'bold'})
            ai   = clf_met[clf_met['Model']==m]['Accuracy'].values[0]
            star = '  ★' if m == best_name else ''
            ax.set_title(f'{m}{star}\nAcc {ai:.3f}', fontsize=8.5,
                         color=GOLD2 if m == best_name else MUTED2, pad=5)
            ax.set_xlabel('Pred', fontsize=8); ax.set_ylabel('True', fontsize=8)
        except Exception:
            ax.axis('off')
    for j in range(i+1, len(flat)):
        flat[j].axis('off')

    fig_g.patch.set_facecolor(CARD)
    plt.suptitle('Confusion Matrices — All Classification Models', color=TEXT,
                 fontsize=11, y=1.01, fontfamily='monospace')
    plt.tight_layout()
    st.pyplot(fig_g, use_container_width=True)
    plt.close(fig_g)


# ════════════════════════════════════════════════════════════
#  TAB 4 — REGRESSION
# ════════════════════════════════════════════════════════════
with tabs[3]:
    chart_style()
    brnm  = best_reg['name']
    y_arr = np.array(y_test_reg)
    y_prd = reg_res[brnm]['y_pred']
    rmods = reg_met['Model'].tolist()
    bri   = reg_met[reg_met['Model']==brnm].index[0] - 1

    st.markdown(f"""
    <div class="sec-hd"><div class="sec-icon">◇</div>
    <div class="sec-txt">
      <div class="sec-title">Regression Analysis</div>
      <div class="sec-sub">Best model: {brnm} · R² = {r2:.4f}</div>
    </div></div>""", unsafe_allow_html=True)

    fig_r2, ax_r2 = plt.subplots(figsize=(13, 4))
    clrs_r = [GREEN if m == brnm else TEAL for m in rmods]
    brs_r  = ax_r2.barh(rmods, reg_met['R² Score'].values, color=clrs_r, alpha=0.85, height=0.55, linewidth=0)
    for bar, val in zip(brs_r, reg_met['R² Score'].values):
        ax_r2.text(bar.get_width()+.006, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9, color=TEXT)
    ax_r2.invert_yaxis(); ax_r2.set_xlim(-0.1, 1.15)
    ax_r2.axvline(0, color=RED, ls='--', lw=0.8, alpha=0.5)
    ax_r2.grid(True, alpha=0.2, axis='x')
    ax_r2.set_title(f'R² Score  ·  Higher is Better  ·  ★ Best: {brnm}', pad=8)
    fig_r2.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig_r2, use_container_width=True)
    plt.close(fig_r2)

    fig_e, (ae1, ae2) = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax_e, metric, color in [(ae1,'RMSE',RED),(ae2,'MAE',GOLD)]:
        vals_e = reg_met[metric].values
        clrs_e = [PINK if m == brnm else color for m in rmods]
        brs_e  = ax_e.barh(rmods, vals_e, color=clrs_e, alpha=0.85, height=0.55, linewidth=0)
        for bar, val in zip(brs_e, vals_e):
            ax_e.text(bar.get_width()+max(vals_e)*.01, bar.get_y()+bar.get_height()/2,
                      f'{val:.3f}', va='center', fontsize=9, color=TEXT)
        ax_e.invert_yaxis(); ax_e.grid(True, alpha=0.2, axis='x')
        ax_e.set_title(f'{metric}  ·  Lower is Better', pad=8)
    fig_e.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig_e, use_container_width=True)
    plt.close(fig_e)

    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">◇</div>
    <div class="sec-txt">
      <div class="sec-title">Actual vs Predicted Price</div>
      <div class="sec-sub">Time-series & scatter plot</div>
    </div></div>""", unsafe_allow_html=True)

    fig_ap, (al, ar) = plt.subplots(1, 2, figsize=(14, 5))
    al.plot(y_arr, color=TEAL,  lw=1.8, label='Actual Price',       zorder=3)
    al.plot(y_prd, color=GOLD,  lw=1.8, ls='--', label=f'Predicted ({brnm})', zorder=2, alpha=0.9)
    al.fill_between(range(len(y_arr)), y_arr, y_prd, alpha=0.07, color=GREEN)
    al.set_xlabel('Test Period Steps'); al.set_ylabel('Price')
    al.set_title('Price: Actual vs Predicted', pad=8)
    al.legend(fontsize=9, framealpha=.2, facecolor=CARD); al.grid(True, alpha=0.2)

    mn_ = min(y_arr.min(), y_prd.min()); mx_ = max(y_arr.max(), y_prd.max())
    ar.scatter(y_arr, y_prd, alpha=0.35, color=TEAL, s=16, zorder=2)
    ar.plot([mn_,mx_],[mn_,mx_], color=GOLD, lw=1.8, ls='--', label='Perfect Prediction')
    ar.set_xlabel('Actual Price'); ar.set_ylabel('Predicted Price')
    ar.set_title('Scatter: Actual vs Predicted', pad=8)
    ar.legend(fontsize=9, framealpha=.2, facecolor=CARD); ar.grid(True, alpha=0.2)

    fig_ap.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig_ap, use_container_width=True)
    plt.close(fig_ap)

    st.dataframe(
        reg_met.style
            .background_gradient(subset=['R² Score'], cmap='Greens')
            .background_gradient(subset=['RMSE','MAE'], cmap='Reds_r')
            .format({'MAE':'{:.4f}','MSE':'{:.4f}','RMSE':'{:.4f}',
                     'MAPE (%)':'{:.2f}%','R² Score':'{:.4f}','Training Time (s)':'{:.3f}s'}),
        use_container_width=True)

    buf2 = io.StringIO(); reg_met.to_csv(buf2, index=False)
    st.download_button("⬇  Download Regression Metrics CSV", buf2.getvalue(), f"{ticker}_reg_metrics.csv", "text/csv")


# ════════════════════════════════════════════════════════════
#  TAB 5 — ALL MODELS
# ════════════════════════════════════════════════════════════
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="sec-hd"><div class="sec-icon">◎</div>
        <div class="sec-txt">
          <div class="sec-title">Classification — All 9 Models</div>
          <div class="sec-sub">Ranked by F1-Score</div>
        </div></div>""", unsafe_allow_html=True)
        st.dataframe(
            clf_met.style
                .background_gradient(subset=['F1-Score','Accuracy'], cmap='Blues')
                .background_gradient(subset=['Precision','Recall'],  cmap='Greens')
                .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}',
                         'F1-Score':'{:.4f}','Training Time (s)':'{:.3f}s'}),
            use_container_width=True, height=370)
        st.success(f"🏆  Best: **{best_clf['name']}**  ·  F1 = {f1:.4f}  ·  Acc = {acc:.4f}")

    with c2:
        st.markdown("""
        <div class="sec-hd"><div class="sec-icon">◇</div>
        <div class="sec-txt">
          <div class="sec-title">Regression — All 8 Models</div>
          <div class="sec-sub">Ranked by R² Score</div>
        </div></div>""", unsafe_allow_html=True)
        st.dataframe(
            reg_met.style
                .background_gradient(subset=['R² Score'], cmap='Greens')
                .background_gradient(subset=['RMSE','MAE'], cmap='Reds_r')
                .format({'MAE':'{:.4f}','MSE':'{:.4f}','RMSE':'{:.4f}',
                         'MAPE (%)':'{:.2f}%','R² Score':'{:.4f}','Training Time (s)':'{:.3f}s'}),
            use_container_width=True, height=370)
        st.success(f"🏆  Best: **{best_reg['name']}**  ·  R² = {r2:.4f}")

    chart_style()
    st.markdown("""
    <div class="sec-hd" style="margin-top:14px;"><div class="sec-icon">⏱</div>
    <div class="sec-txt">
      <div class="sec-title">Training Time Comparison</div>
      <div class="sec-sub">Wall-clock seconds per model</div>
    </div></div>""", unsafe_allow_html=True)

    fig_t, (at1, at2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax_t, mdf, title, bst in [
        (at1, clf_met, 'Classification Models', best_clf['name']),
        (at2, reg_met, 'Regression Models',     best_reg['name']),
    ]:
        mods_t = mdf['Model'].tolist()
        times  = mdf['Training Time (s)'].values
        cgrad  = plt.cm.YlGnBu(np.linspace(0.3, 0.85, len(mods_t)))
        brs_t  = ax_t.barh(mods_t, times, color=cgrad, alpha=0.85, height=0.55, linewidth=0)
        for bar, tv in zip(brs_t, times):
            ax_t.text(bar.get_width()+max(times)*.01, bar.get_y()+bar.get_height()/2,
                      f'{tv:.3f}s', va='center', fontsize=9, color=TEXT)
        ax_t.invert_yaxis(); ax_t.grid(True, alpha=0.2, axis='x')
        ax_t.set_title(title, pad=8); ax_t.set_xlabel('Seconds')
    fig_t.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)

    st.markdown("""
    <div class="sec-hd"><div class="sec-icon">△</div>
    <div class="sec-txt">
      <div class="sec-title">Feature Importance — Top 15</div>
      <div class="sec-sub">Tree-based models only</div>
    </div></div>""", unsafe_allow_html=True)

    fi_clf = get_feature_importance(best_clf['model'], st.session_state.feature_cols)
    fi_reg = get_feature_importance(best_reg['model'], st.session_state.feature_cols)

    fc1, fc2 = st.columns(2)
    for col, fi_df, name in [(fc1, fi_clf, best_clf['name']), (fc2, fi_reg, best_reg['name'])]:
        with col:
            if fi_df is None:
                st.info(f"Feature importance not available for {name}.")
                continue
            top = fi_df.head(15)
            fig_fi, ax_fi = plt.subplots(figsize=(6.5, 5.5))
            cg = plt.cm.GnBu(np.linspace(0.35, 0.95, len(top)))[::-1]
            ax_fi.barh(top['Feature'], top['Importance'], color=cg, alpha=0.9, height=0.65, linewidth=0)
            ax_fi.invert_yaxis(); ax_fi.grid(True, alpha=0.2, axis='x')
            ax_fi.set_title(f'Top Features · {name}', fontsize=9, pad=8); ax_fi.tick_params(labelsize=8)
            fig_fi.patch.set_facecolor(CARD)
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

    st.markdown("---")
    summary_txt = summarize_results(clf_met, reg_met, best_clf, best_reg)
    st.download_button("⬇  Download Full Analysis Report",
                       summary_txt, f"{ticker}_full_report.txt", "text/plain",
                       use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 6 — STOCK CHARTS
# ════════════════════════════════════════════════════════════
with tabs[5]:
    chart_style()

    fig_p, (ap1, ap2) = plt.subplots(2, 1, figsize=(13, 8),
                                      gridspec_kw={'height_ratios':[3,1]})
    ap1.plot(proc_df.index, proc_df['Close'], color=TEAL, lw=1.8, label='Close', zorder=3)
    if 'SMA_20' in proc_df.columns:
        ap1.plot(proc_df.index, proc_df['SMA_20'], color=GOLD,   lw=1, ls='--', alpha=.8, label='SMA 20')
    if 'SMA_50' in proc_df.columns:
        ap1.plot(proc_df.index, proc_df['SMA_50'], color=PURPLE, lw=1, ls='--', alpha=.8, label='SMA 50')
    if 'BB_Upper' in proc_df.columns:
        ap1.fill_between(proc_df.index, proc_df['BB_Lower'], proc_df['BB_Upper'],
                         alpha=.07, color=TEAL, label='Bollinger Bands')
    ap1.set_ylabel('Price'); ap1.legend(fontsize=9, framealpha=.2, facecolor=CARD)
    ap1.grid(True, alpha=.2); ap1.tick_params(labelbottom=False)
    ap1.set_title(f'{ticker}  ·  Price & Bollinger Bands', pad=10)
    vc = [GREEN if proc_df['Close'].iloc[i] >= proc_df['Open'].iloc[i] else RED for i in range(len(proc_df))]
    ap2.bar(proc_df.index, proc_df['Volume'], color=vc, alpha=.6, width=.8)
    ap2.set_ylabel('Volume'); ap2.grid(True, alpha=.2)
    fig_p.patch.set_facecolor(CARD)
    plt.tight_layout()
    st.pyplot(fig_p, use_container_width=True)
    plt.close(fig_p)

    if 'RSI_14' in proc_df.columns and 'MACD' in proc_df.columns:
        fig_ti, axs = plt.subplots(3, 1, figsize=(13, 9),
                                   gridspec_kw={'height_ratios':[2,1,1]})
        axs[0].plot(proc_df.index, proc_df['Close'], color=TEAL, lw=1.4, label='Close')
        if 'EMA_12' in proc_df.columns:
            axs[0].plot(proc_df.index, proc_df['EMA_12'], color=GOLD,   lw=1, ls='--', alpha=.8, label='EMA 12')
        if 'EMA_26' in proc_df.columns:
            axs[0].plot(proc_df.index, proc_df['EMA_26'], color=PURPLE, lw=1, ls='--', alpha=.8, label='EMA 26')
        axs[0].set_ylabel('Price'); axs[0].legend(fontsize=9, framealpha=.2, facecolor=CARD)
        axs[0].grid(True, alpha=.2); axs[0].tick_params(labelbottom=False)
        axs[0].set_title(f'{ticker}  ·  EMA · RSI · MACD Dashboard', pad=10)

        axs[1].plot(proc_df.index, proc_df['RSI_14'], color=PINK, lw=1.2, label='RSI (14)')
        axs[1].axhline(70, color=RED,   lw=.8, ls='--', alpha=.6, label='Overbought 70')
        axs[1].axhline(30, color=GREEN, lw=.8, ls='--', alpha=.6, label='Oversold 30')
        axs[1].fill_between(proc_df.index, proc_df['RSI_14'], 70,
                             where=(proc_df['RSI_14']>=70), color=RED,   alpha=.1)
        axs[1].fill_between(proc_df.index, proc_df['RSI_14'], 30,
                             where=(proc_df['RSI_14']<=30), color=GREEN, alpha=.1)
        axs[1].set_ylim(0, 100); axs[1].set_ylabel('RSI')
        axs[1].legend(fontsize=8, framealpha=.2, facecolor=CARD)
        axs[1].grid(True, alpha=.2); axs[1].tick_params(labelbottom=False)

        axs[2].plot(proc_df.index, proc_df['MACD'], color=BLUE, lw=1.2, label='MACD')
        if 'MACD_Signal' in proc_df.columns:
            axs[2].plot(proc_df.index, proc_df['MACD_Signal'], color=GOLD, lw=1, ls='--', label='Signal')
        if 'MACD_Hist' in proc_df.columns:
            hc = [GREEN if v >= 0 else RED for v in proc_df['MACD_Hist']]
            axs[2].bar(proc_df.index, proc_df['MACD_Hist'], color=hc, alpha=.5, width=.8)
        axs[2].axhline(0, color=MUTED, lw=.5, alpha=.5)
        axs[2].set_ylabel('MACD'); axs[2].legend(fontsize=8, framealpha=.2, facecolor=CARD)
        axs[2].grid(True, alpha=.2)

        fig_ti.patch.set_facecolor(CARD)
        plt.tight_layout()
        st.pyplot(fig_ti, use_container_width=True)
        plt.close(fig_ti)


# ─── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="ft-brand">Stock<span>Sense</span> ML</div>
</div>
""", unsafe_allow_html=True)


#venv\Scripts\activate           # Windows

# streamlit run app.py