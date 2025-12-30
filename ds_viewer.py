import os
import pickle
# 分:秒形式に変換する関数
def format_func(value, tick_number):
    hours = int(value// 3600 )
    minutes = int((value // 60) % 60)
    seconds = int(value % 60)
    return f"{hours}:{minutes}:{seconds:02}"


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import FuncFormatter

# =========================================================
# load
# =========================================================
filename = os.path.splitext(os.listdir("tmp_raw_d/")[0])[0]
with open(f"tmp_raw_d/{filename}.pkl", "rb") as f:
    data = pickle.load(f)

gut_d = data["gut_d"]
filtered_gut_d = data["filtered_gut_d"]
gut2_d = data["gut2_d"]
filtered_gut2_d = data["filtered_gut2_d"]
gut3_d = data["gut3_d"]
filtered_gut3_d = data["filtered_gut3_d"]
gut_t = data['gut_t']

emg_ma_t = data["emg_ma_t"]
emg_ma = data["emg_ma"]

# =========================================================
# initial settings
# =========================================================
display_fps = 4
center_time = 5

# =========================================================
# plot
# =========================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
ax1, ax2, ax3 = axs
axes_list = [ax1, ax2, ax3]
names = ["ax1", "ax2", "ax3"]

ax1.plot(gut_t, gut_d, lw=0.3)
ax1.plot(gut_t, filtered_gut_d, lw=0.3)
# ax1.plot(emg_ma_t, emg_ma, lw=0.3)

ax2.plot(gut_t, gut2_d, lw=0.3)
ax2.plot(gut_t, filtered_gut2_d, lw=0.3)
# ax2.plot(emg_ma_t, emg_ma, lw=0.3)

ax3.plot(gut_t, gut3_d, lw=0.3)
ax3.plot(gut_t, filtered_gut3_d, lw=0.3)
# ax3.plot(emg_ma_t, emg_ma, lw=0.3)

for ax in axs:
    ax.set_ylim([-1000, 5000])

# =========================================================
# helpers
# =========================================================
def _xlim_width(ax):
    xl = ax.get_xlim()
    return xl[1] - xl[0]

def set_center_keep_width(ax, center):
    w = _xlim_width(ax)
    ax.set_xlim(center - w / 2, center + w / 2)

def center_of(ax):
    xl = ax.get_xlim()
    return 0.5 * (xl[0] + xl[1])

# =========================================================
# red lines: ALWAYS at each axis center
# =========================================================
redlines = [ax.axvline(center_time, color="r", lw=0.8) for ax in axs]

def update_redlines_to_center():
    for ax, rl in zip(axs, redlines):
        rl.set_xdata([center_of(ax)])
    fig.canvas.draw_idle()

# =========================================================
# slider (bottom): move ALL axes by center, keep width
# =========================================================
slider_ax = plt.axes([0.10, 0.02, 0.80, 0.03])
slider = Slider(
    slider_ax, "Center time (s)",
    0, float(gut_t[-1]),
    valinit=center_time,
    valstep=1 / display_fps
)

# =========================================================
# link rule you want:
#   Receive OFF -> that axis is completely excluded from linking
#   => any edge involving that axis is treated as OFF
# =========================================================
receive = {"ax1": True, "ax2": True, "ax3": True}

edge_labels = [
    "ax1→ax2", "ax1→ax3",
    "ax2→ax1", "ax2→ax3",
    "ax3→ax1", "ax3→ax2",
]
edge_state = {lab: True for lab in edge_labels}  # 好きに初期値OK（後でUIでON/OFF）
edge_to_pair = {
    "ax1→ax2": (0, 1),
    "ax1→ax3": (0, 2),
    "ax2→ax1": (1, 0),
    "ax2→ax3": (1, 2),
    "ax3→ax1": (2, 0),
    "ax3→ax2": (2, 1),
}

def _axis_active(i: int) -> bool:
    return receive[names[i]]

def current_sync_map():
    """
    有効な同期だけ返す。
    条件:
      1) edge_state が ON
      2) source軸もtarget軸も receive が ON（=その軸を含むリンクのみ）
    """
    sync_map = {0: set(), 1: set(), 2: set()}
    for lab, on in edge_state.items():
        if not on:
            continue
        s, d = edge_to_pair[lab]
        if (not _axis_active(s)) or (not _axis_active(d)):
            continue
        sync_map[s].add(d)
    return sync_map

# =========================================================
# sync behavior:
#   - linked dst: xlim fully copy (center+width match)
#   - not linked: center follow only, width unchanged
#   - redline always at center -> consistent
# =========================================================
is_updating = False

def apply_sync_from(source_ax):
    global is_updating
    if is_updating:
        return
    is_updating = True

    src_i = axes_list.index(source_ax)

    # source が inactive なら同期しない（見た目だけ整える）
    src_xlim = source_ax.get_xlim()
    src_center = 0.5 * (src_xlim[0] + src_xlim[1])

    slider.set_val(src_center)

    sync_map = current_sync_map()
    active_src = _axis_active(src_i)

    for j, ax in enumerate(axes_list):
        if j == src_i:
            continue

        if active_src and (j in sync_map[src_i]):
            ax.set_xlim(*src_xlim)              # FULL copy
        else:
            set_center_keep_width(ax, src_center)  # center only

    update_redlines_to_center()
    fig.canvas.draw_idle()
    is_updating = False

def on_xlim_changed(event_ax):
    if is_updating:
        return
    apply_sync_from(event_ax)

for ax in axes_list:
    ax.callbacks.connect("xlim_changed", on_xlim_changed)

def on_slider_change(val):
    global is_updating
    if is_updating:
        return
    is_updating = True

    for ax in axes_list:
        set_center_keep_width(ax, val)

    update_redlines_to_center()
    fig.canvas.draw_idle()
    is_updating = False

slider.on_changed(on_slider_change)

# =========================================================
# UI: Receive buttons (3) — persistent colors
# =========================================================
def btn_text(ax_name):
    return ("✓ " if receive[ax_name] else "   ") + f"Include {ax_name}"

ON_FACE  = "lightgreen"
OFF_FACE = "lightgray"

btns = {}

def refresh_receive_button_style(name_):
    b = btns[name_]
    fc = ON_FACE if receive[name_] else OFF_FACE
    b.color = fc
    b.hovercolor = fc
    b.ax.patch.set_facecolor(fc)
    b.label.set_text(btn_text(name_))
    b.ax.figure.canvas.draw_idle()

left0, bottom, width, height = 0.10, 0.92, 0.80, 0.06
gap = 0.01
btn_w = (width - gap * 2) / 3

for i, nm in enumerate(names):
    x = left0 + i * (btn_w + gap)
    a = plt.axes([x, bottom, btn_w, height])
    b = Button(a, btn_text(nm))
    btns[nm] = b
    refresh_receive_button_style(nm)

    def make_onclick(name_):
        def _onclick(_event):
            receive[name_] = not receive[name_]
            refresh_receive_button_style(name_)
            apply_sync_from(ax1)  # 整合（基準は好みで変更OK）
        return _onclick

    b.on_clicked(make_onclick(nm))

# =========================================================
# axis formatting
# =========================================================
ax1.xaxis.set_major_formatter(FuncFormatter(format_func))
ax2.xaxis.set_major_formatter(FuncFormatter(format_func))
ax3.xaxis.set_major_formatter(FuncFormatter(format_func))
ax3.set_xlabel("Time (s)")

# initial center alignment
is_updating = True
for ax in axes_list:
    set_center_keep_width(ax, center_time)
update_redlines_to_center()
slider.set_val(center_time)
is_updating = False

plt.tight_layout(rect=[0, 0.07, 1, 0.90])
plt.show()
