import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Range1d, Span, Slider, Toggle,
    CustomJS, Div, CustomJSTickFormatter
)
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
import streamlit.components.v1 as components


# =========================================================
# settings
# =========================================================
DISPLAY_FPS = 4
CENTER_TIME_DEFAULT = 5.0
Y_LIM = (-1000, 5000)
MAX_POINTS_DEFAULT = 100_000


# =========================================================
# utils
# =========================================================
def list_pkl_files(dir_path="tmp_raw_d"):
    p = Path(dir_path)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.pkl")])


@st.cache_data(show_spinner=False)
def load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def downsample(t, y, max_points: int):
    t = np.asarray(t)
    y = np.asarray(y)
    n = len(t)
    if max_points is None or max_points <= 0 or n <= max_points:
        return t, y
    stride = int(np.ceil(n / max_points))
    return t[::stride], y[::stride]


def make_bokeh_app(data, max_points: int):
    # ---- extract ----
    gut_t = np.asarray(data["gut_t"])

    gut_d = np.asarray(data["gut_d"])
    filtered_gut_d = np.asarray(data["filtered_gut_d"])

    gut2_d = np.asarray(data["gut2_d"])
    filtered_gut2_d = np.asarray(data["filtered_gut2_d"])

    gut3_d = np.asarray(data["gut3_d"])
    filtered_gut3_d = np.asarray(data["filtered_gut3_d"])

    t_end = float(gut_t[-1])

    # ---- downsample ----
    t1, y1 = downsample(gut_t, gut_d, max_points)
    t1f, y1f = downsample(gut_t, filtered_gut_d, max_points)

    t2, y2 = downsample(gut_t, gut2_d, max_points)
    t2f, y2f = downsample(gut_t, filtered_gut2_d, max_points)

    t3, y3 = downsample(gut_t, gut3_d, max_points)
    t3f, y3f = downsample(gut_t, filtered_gut3_d, max_points)

    # ---- sources ----
    src1 = ColumnDataSource(dict(t=t1, y=y1))
    src1f = ColumnDataSource(dict(t=t1f, y=y1f))

    src2 = ColumnDataSource(dict(t=t2, y=y2))
    src2f = ColumnDataSource(dict(t=t2f, y=y2f))

    src3 = ColumnDataSource(dict(t=t3, y=y3))
    src3f = ColumnDataSource(dict(t=t3f, y=y3f))

    sources = [(src1, src1f), (src2, src2f), (src3, src3f)]

    # ---- independent x ranges ----
    x_ranges = [Range1d(float(gut_t[0]), float(gut_t[-1])) for _ in range(3)]

    # initial: center keep width
    center0 = min(CENTER_TIME_DEFAULT, t_end)
    full_w = x_ranges[0].end - x_ranges[0].start
    for r in x_ranges:
        r.start = center0 - full_w / 2
        r.end = center0 + full_w / 2

    # ---- y range ----
    y_range = Range1d(*Y_LIM)

    # ---- tick formatter ----
    hms_formatter = CustomJSTickFormatter(code="""
        const value = tick;
        const hours = Math.floor(value / 3600);
        const minutes = Math.floor((value / 60) % 60);
        const seconds = Math.floor(value % 60);
        const mm = String(minutes).padStart(2, '0');
        const ss = String(seconds).padStart(2, '0');
        return `${hours}:${mm}:${ss}`;
    """)

    # ---- figures (IMPORTANT: each uses its own x_ranges[i]) ----
    figs = []
    for i in range(3):
        p = figure(
            height=220,
            sizing_mode="stretch_width",
            x_range=x_ranges[i],
            y_range=y_range,
            tools="xpan,xwheel_zoom,reset",
            active_drag="xpan",
            active_scroll="xwheel_zoom",
        )
        p.xaxis.formatter = hms_formatter
        p.yaxis.axis_label = "Amplitude"
        p.xaxis.axis_label = "Time (h:mm:ss)" if i == 2 else ""

        p.line("t", "y", source=sources[i][0], line_width=1, alpha=0.9)
        p.line("t", "y", source=sources[i][1], line_width=1, alpha=0.9)

        figs.append(p)

    # ---- red center lines ----
    spans = []
    for p in figs:
        sp = Span(location=center0, dimension="height", line_color="red", line_width=2)
        p.add_layout(sp)
        spans.append(sp)

    # ---- slider ----
    slider = Slider(
        title="Center time (s)",
        start=0.0, end=float(t_end),
        value=float(center0),
        step=1.0 / DISPLAY_FPS
    )

    # ---- Include toggles (3) ----
    receive = [
        Toggle(label="✓ Include ax1", active=True, button_type="success"),
        Toggle(label="✓ Include ax2", active=True, button_type="success"),
        Toggle(label="✓ Include ax3", active=True, button_type="success"),
    ]

    # ---- recursion guard ----
    state = ColumnDataSource(data=dict(updating=[0]))

    # =========================================================
    # CustomJS core
    #   - FULL copy only between Included axes
    #   - otherwise center-only (=> 赤線中心は揃う)
    # =========================================================
    core_js = """
    function centerOf(r){ return 0.5*(r.start + r.end); }
    function widthOf(r){ return (r.end - r.start); }
    function setCenterKeepWidth(r, center){
        const w = widthOf(r);
        r.start = center - w/2;
        r.end   = center + w/2;
    }
    function axisActive(i){
        return receive[i].active;
    }
    function updateSpansToCenters(){
        for(let i=0;i<3;i++){
            spans[i].location = centerOf(ranges[i]);
        }
    }

    function applySyncFrom(src){
        if(state.data.updating[0] === 1) return;
        state.data.updating[0] = 1;

        const rsrc = ranges[src];
        const src_center = centerOf(rsrc);

        // slider follows center always
        slider.value = src_center;

        const active_src = axisActive(src);

        for(let j=0;j<3;j++){
            if(j===src) continue;

            if(active_src && axisActive(j)){
                // Included同士 → FULL copy
                ranges[j].start = rsrc.start;
                ranges[j].end   = rsrc.end;
            }else{
                // それ以外 → center-only
                setCenterKeepWidth(ranges[j], src_center);
            }
        }

        updateSpansToCenters();
        state.data.updating[0] = 0;
    }
    """

    # range changed callbacks (xlim_changed 相当)
    for i in range(3):
        cb = CustomJS(
            args=dict(ranges=x_ranges, slider=slider, spans=spans, receive=receive, state=state),
            code=core_js + f"applySyncFrom({i});"
        )
        x_ranges[i].js_on_change("start", cb)
        x_ranges[i].js_on_change("end", cb)

    # slider callback: move ALL axes by center, keep width (include無視)
    slider_cb = CustomJS(
        args=dict(ranges=x_ranges, slider=slider, spans=spans, receive=receive, state=state),
        code=core_js + """
        if(state.data.updating[0] === 1) return;
        state.data.updating[0] = 1;

        const c = slider.value;
        for(let i=0;i<3;i++){
            setCenterKeepWidth(ranges[i], c);
        }
        updateSpansToCenters();

        state.data.updating[0] = 0;
        """
    )
    slider.js_on_change("value", slider_cb)

    # include toggle callback:
    # - label update
    # - baseline は「最初に active な軸」にする（ax1 off時も ax2/ax3 をFULL copyで揃える）
    receive_cb = CustomJS(
        args=dict(ranges=x_ranges, slider=slider, spans=spans, receive=receive, state=state),
        code=core_js + """
        for(let i=0;i<3;i++){
            receive[i].label = (receive[i].active ? "✓ " : "   ") + `Include ax${i+1}`;
            receive[i].button_type = receive[i].active ? "success" : "default";
        }

        let base = 0;
        if(receive[0].active) base = 0;
        else if(receive[1].active) base = 1;
        else if(receive[2].active) base = 2;

        applySyncFrom(base);
        """
    )
    for t in receive:
        t.js_on_change("active", receive_cb)

    header = Div(
        text="<b>Pan/Zoom any axis</b> → Included axes FULL-copy sync, otherwise center-only. (Red line = each axis center)",
        sizing_mode="stretch_width"
    )
    receive_row = row(*receive, sizing_mode="stretch_width")
    layout = column(
        header,
        receive_row,
        figs[0],
        figs[1],
        figs[2],
        slider,
        sizing_mode="stretch_width",
    )
    return layout


# =========================================================
# Streamlit main
# =========================================================
st.set_page_config(page_title="Viewer", layout="wide")
st.title("Viewer (Include-only sync)")

pkl_files = list_pkl_files("tmp_raw_d")
if not pkl_files:
    st.error("tmp_raw_d/ に .pkl が見つかりませんでした。リポジトリに tmp_raw_d と pkl を含めてください。")
    st.stop()

with st.sidebar:
    st.header("Data")
    pkl_path = st.selectbox("Pick a .pkl", pkl_files, index=0)

    st.header("Performance")
    max_points = st.number_input(
        "Max points per trace (downsample)",
        min_value=10_000,
        max_value=2_000_000,
        value=MAX_POINTS_DEFAULT,
        step=10_000
    )

data = load_pkl(pkl_path)

required = ["gut_t","gut_d","filtered_gut_d","gut2_d","filtered_gut2_d","gut3_d","filtered_gut3_d"]
missing = [k for k in required if k not in data]
if missing:
    st.error(f"pkl に必要キーがありません: {missing}")
    st.stop()

layout = make_bokeh_app(data, int(max_points))
html = file_html(layout, CDN, "viewer")
components.html(html, height=1100, scrolling=True)
