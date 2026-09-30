# -*- coding: utf-8 -*-
"""
人海中寻找真爱 —— 信号检测论（SDT）交互式实验
Python / tkinter 实现（仅用标准库，无需安装任何第三方包）

运行方式：
    python sdt_experiment.py

实验任务：
    屏幕闪现一屏 3x3 的人群（共 9 人），其中可能藏着一位「真爱」。
    所有人的人形、灰色、大小完全一致，唯一的区分特征是胸口心形的完整程度：
        完整的心（100%）  -> 真爱（信号 S），每屏最多 1 个
        3/4 颗心（75%）   -> 干扰项（右侧约 1/4 被裁掉）
        半颗心（50%）     -> 干扰项（右侧一半被裁掉）
    刺激消失后判断「刚才的人群里出现真爱了吗」：
        按 F = 有真爱，按 J = 没有（也可以点击按钮，请切换到英文输入法）
    全程无即时反馈，24 试次结束后统一报告 P(Hit)/P(FA)/d′/c/β 等结果。
"""

import math
import random
import time
import tkinter as tk
from tkinter import ttk

# =====================================================================
# 全局参数
# =====================================================================
CONFIG = {
    "trials": 24,          # 试次总数（>= 20）
    "cols": 3, "rows": 3,  # 人群网格 3 x 3 = 9 人
    "person_scale": 2.5,   # 人物缩放：基准人身高约 72 单位 -> 约 180px
    "fixation_ms": 400,    # 注视点时长（试次间隔）
    "canvas_w": 900,
    "canvas_h": 640,
    "bg": "#eef1f5",
    # 难度：duration = 人群呈现时长(ms)；n_quarter = 干扰项中「3/4 颗心」的数量
    "levels": {
        "easy":   {"label": "容易", "duration": 1500, "n_quarter": 4},
        "medium": {"label": "中等", "duration": 1000, "n_quarter": 4},
        "hard":   {"label": "困难", "duration": 500,  "n_quarter": 4},
    },
}

# =====================================================================
# 统计工具
# =====================================================================
def norm_cdf(x):
    """标准正态分布累积函数 Φ(x)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_norm(p):
    """标准正态分布的反函数 z(p)（Acklam 算法）"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def compute_sdt(trials):
    """SDT 指标计算（对数线性校正，避免 0/1 比例导致 z 值无穷大）"""
    n_s = sum(1 for t in trials if t["is_signal"])
    n_n = len(trials) - n_s
    hit = sum(1 for t in trials if t["is_signal"] and t["said_yes"])
    fa = sum(1 for t in trials if not t["is_signal"] and t["said_yes"])
    n_yes = hit + fa                                   # 回答「有真爱」的总次数
    p_h = (hit + 0.5) / (n_s + 1)
    p_f = (fa + 0.5) / (n_n + 1)
    z_h, z_f = inv_norm(p_h), inv_norm(p_f)
    return {
        "hit": hit, "miss": n_s - hit, "fa": fa, "cr": n_n - fa,
        "n_s": n_s, "n_n": n_n, "n_yes": n_yes,
        "yes_rate": n_yes / len(trials),
        "p_h": p_h, "p_f": p_f,
        "d_prime": z_h - z_f,
        "c": -(z_h + z_f) / 2.0,
        "beta": math.exp((z_f * z_f - z_h * z_h) / 2.0),
        "acc": (hit + (n_n - fa)) / len(trials),
    }


# =====================================================================
# 几何：心形与身体（与网页版完全相同的贝塞尔参数）
# =====================================================================
def _bezier(p0, c1, c2, p3, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        pts.append((mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p3[0],
                    mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p3[1]))
    return pts


def _quad(p0, c, p1, n=16):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        pts.append((mt**2 * p0[0] + 2 * mt * t * c[0] + t**2 * p1[0],
                    mt**2 * p0[1] + 2 * mt * t * c[1] + t**2 * p1[1]))
    return pts


def _heart_points(s):
    pts = _bezier((0, 0.95 * s), (-1.35 * s, -0.15 * s), (-0.75 * s, -1.15 * s), (0, -0.4 * s))
    pts += _bezier((0, -0.4 * s), (0.75 * s, -1.15 * s), (1.35 * s, -0.15 * s), (0, 0.95 * s))
    return pts


S = 8                                                        # 心的半宽（基准单位）
HEART_PTS = [(x, y - 2) for (x, y) in _heart_points(S)]      # 平移到胸口 (0, -2)
HEART_XEDGE = max(abs(p[0]) for p in HEART_PTS)              # 心形实际水平半宽 ≈ 0.803 * S
BODY_PTS = [(-15, -13), (15, -13)]
BODY_PTS += _quad((15, -13), (18, -6), (18, 28))
BODY_PTS += _quad((18, 28), (0, 36), (-18, 28))
BODY_PTS += _quad((-18, 28), (-18, -6), (-15, -13))


def hsl(h, s, l):
    """hsl(h∈[0,360), s,l∈[0,100]) -> '#rrggbb'"""
    h, s, l = h / 360.0, s / 100.0, l / 100.0

    def f(n):
        k = (n + h * 12) % 12
        a = s * min(l, 1 - l)
        v = l - a * max(-1.0, min(k - 3, 9 - k, 1))
        return round(v * 255)
    return f"#{f(0):02x}{f(8):02x}{f(4):02x}"


# =====================================================================
# 人群生成
# =====================================================================
def make_crowd(is_signal, level):
    """生成一屏 9 人：真爱 = 唯一完整的心；干扰项 = n_quarter 个 75% 心 + 其余 50% 心"""
    w, h = CONFIG["canvas_w"], CONFIG["canvas_h"]
    cols, rows = CONFIG["cols"], CONFIG["rows"]
    cw, ch = w / cols, h / rows
    slots = [{"x": cw * (c + 0.5) + random.uniform(-24, 24),
              "y": ch * (r + 0.5) + random.uniform(-10, 10)}
             for r in range(rows) for c in range(cols)]
    random.shuffle(slots)

    crowd, idx = [], 0

    def put(heart_level):
        nonlocal idx
        sl = slots[idx]
        idx += 1
        crowd.append({"x": sl["x"], "y": sl["y"], "heart_level": heart_level,
                      "seed": random.randint(0, 99999)})

    if is_signal:
        put(1.0)                                   # 唯一的真爱：完整的心
    n_rest = len(slots) - idx
    n_quarter = min(level["n_quarter"], n_rest)
    for i in range(n_rest):
        put(0.75 if i < n_quarter else 0.5)        # 位置已随 slots 打乱
    return crowd


def build_sequence():
    """生成 24 试次的平衡序列：恰好一半有信号，随机打乱"""
    seq = [True] * (CONFIG["trials"] // 2) + [False] * (CONFIG["trials"] // 2)
    random.shuffle(seq)
    return seq


# =====================================================================
# 主应用（GUI）
# =====================================================================
class ExperimentApp:
    def __init__(self, root):
        self.root = root
        self.difficulty = "medium"
        self.phase = "idle"
        self.trials = []
        self.trial_idx = 0
        self.cur_trial = None
        self.seq = []
        self._after_id = None

        top = tk.Frame(root)
        top.pack(fill="x", padx=16, pady=(12, 0))
        self.diff_buttons = {}
        for lv in ("easy", "medium", "hard"):
            btn = tk.Button(top, text=CONFIG["levels"][lv]["label"], width=8,
                            command=lambda v=lv: self.set_difficulty(v))
            btn.pack(side="left", padx=4)
            self.diff_buttons[lv] = btn
        self.progress = tk.Label(top, text="准备就绪", fg="#6b7280")
        self.progress.pack(side="right")

        note = ("实验过程中不提供即时正误反馈，每答完一题画面直接回到注视点；"
                "全部 24 试次结束后统一报告 P(Hit)、P(FA)、d′、c、β。"
                "难度越高呈现时间越短（容易 1.5s / 中等 1.0s / 困难 0.5s）。")
        tk.Label(root, text=note, fg="#6b7280", justify="left",
                 wraplength=CONFIG["canvas_w"]).pack(anchor="w", padx=16, pady=(6, 0))

        self.canvas = tk.Canvas(root, width=CONFIG["canvas_w"], height=CONFIG["canvas_h"],
                                bg=CONFIG["bg"], highlightthickness=1,
                                highlightbackground="#e5e7eb")
        self.canvas.pack(padx=16, pady=8)

        bottom = tk.Frame(root)
        bottom.pack(pady=(0, 4))
        self.start_btn = tk.Button(bottom, text="开始实验", font=("Microsoft YaHei", 12, "bold"),
                                   bg="#e11d63", fg="white", relief="flat",
                                   padx=26, pady=6, cursor="hand2", command=self.start_experiment)
        self.start_btn.pack()
        self.yes_btn = tk.Button(bottom, text="有真爱 [F]", font=("Microsoft YaHei", 12, "bold"),
                                 bg="white", fg="#e11d63", relief="solid", bd=2,
                                 padx=30, pady=6, cursor="hand2",
                                 command=lambda: self.record_response(True))
        self.no_btn = tk.Button(bottom, text="没有 [J]", font=("Microsoft YaHei", 12, "bold"),
                                bg="white", fg="#475569", relief="solid", bd=2,
                                padx=30, pady=6, cursor="hand2",
                                command=lambda: self.record_response(False))

        self.status = tk.Label(root, text="选择难度后点击「开始实验」。", fg="#6b7280")
        self.status.pack(pady=(0, 10))

        root.bind("<KeyPress>", self.on_key)
        self.draw_fixation()
        self._refresh_diff_buttons()

    # ---------------- 界面辅助 ----------------
    def _refresh_diff_buttons(self):
        for lv, btn in self.diff_buttons.items():
            active = lv == self.difficulty
            btn.configure(bg="#fce7f0" if active else "white",
                          fg="#e11d63" if active else "#6b7280",
                          font=("Microsoft YaHei", 10, "bold" if active else "normal"))

    def set_difficulty(self, lv):
        if self.phase not in ("idle", "done"):
            return
        self.difficulty = lv
        self._refresh_diff_buttons()

    def _schedule(self, ms, fn):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(ms, fn)

    # ---------------- 绘制 ----------------
    def _clear(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, CONFIG["canvas_w"], CONFIG["canvas_h"],
                                     fill=CONFIG["bg"], width=0)

    def draw_fixation(self):
        self._clear()
        self.canvas.create_text(CONFIG["canvas_w"] / 2, CONFIG["canvas_h"] / 2,
                                text="+", font=("Arial", 40, "bold"), fill="#334155")

    def draw_person(self, cx, cy, heart_level, seed):
        k = CONFIG["person_scale"]
        jitter = ((seed * 7919) % 30) / 10 - 1.5          # 每人 ±1.5% 明度抖动（所有人同为灰色）
        body = hsl(214 + (seed * 13) % 10, 14, 52 + jitter)
        heart = hsl(348, 52, 60)                          # 心形颜色对所有人一致，只区分形状

        def pts(base):
            return [coord for p in base for coord in (cx + p[0] * k, cy + p[1] * k)]

        # 身体（圆角梯形）、脖子、头
        self.canvas.create_polygon(pts(BODY_PTS), fill=body, width=0)
        self.canvas.create_rectangle(cx - 4 * k, cy - 18 * k, cx + 4 * k, cy - 12 * k,
                                     fill=body, width=0)
        self.canvas.create_oval(cx - 11 * k, cy - 36 * k, cx + 11 * k, cy - 14 * k,
                                fill=body, width=0)

        # 胸口的心：按 heart_level 从左往右保留相应比例（1=完整 / 0.75 / 0.5）
        if heart_level >= 1:
            hp = HEART_PTS
        else:
            cut = -HEART_XEDGE + 2 * HEART_XEDGE * heart_level   # 按实际宽度比例竖直裁切
            hp = [p for p in HEART_PTS if p[0] <= cut + 1e-6]
        self.canvas.create_polygon(pts(hp), fill=heart, width=0)

    def draw_crowd(self, crowd):
        self._clear()
        for p in crowd:
            self.draw_person(p["x"], p["y"], p["heart_level"], p["seed"])

    def draw_mask(self):
        """遮蔽画面：杂乱灰点，防止回视"""
        self._clear()
        for _ in range(340):
            r = random.uniform(1, 4.5)
            x, y = random.uniform(0, CONFIG["canvas_w"]), random.uniform(0, CONFIG["canvas_h"])
            self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                    fill=hsl(215, 10, random.uniform(60, 90)), width=0)

    # ---------------- 实验流程 ----------------
    def start_experiment(self):
        self.trials = []
        self.trial_idx = 0
        self.seq = build_sequence()                # 一次性生成 12 信号 / 12 噪声的平衡序列
        self.start_btn.configure(state="disabled", text="实验进行中…")
        self.phase = "fixation"
        self.next_trial()

    def next_trial(self):
        if self.trial_idx >= len(self.seq):
            self.finish_experiment()
            return
        self.cur_trial = {"is_signal": self.seq[self.trial_idx]}
        self.phase = "fixation"
        self.progress.configure(text=f"第 {self.trial_idx + 1} / {CONFIG['trials']} 试次")
        self.status.configure(text="")
        self.draw_fixation()
        self._schedule(CONFIG["fixation_ms"], self.show_stimulus)

    def show_stimulus(self):
        self.phase = "stimulus"
        lv = CONFIG["levels"][self.difficulty]
        self.cur_trial["crowd"] = make_crowd(self.cur_trial["is_signal"], lv)
        self.draw_crowd(self.cur_trial["crowd"])
        self._schedule(lv["duration"], self.ask_response)   # 刺激限时，超时自动遮蔽

    def ask_response(self):
        self.phase = "response"
        self.cur_trial["stim_off"] = time.perf_counter()
        self.draw_mask()
        cx, cy = CONFIG["canvas_w"] / 2, CONFIG["canvas_h"] / 2
        self.canvas.create_text(cx, cy - 30, text="刚才的人群里，出现真爱了吗？",
                                font=("Microsoft YaHei", 17, "bold"), fill="#334155")
        self.canvas.create_text(cx, cy + 12, text="按 [F] 有真爱    按 [J] 没有（请切换到英文输入法）",
                                font=("Microsoft YaHei", 11), fill="#64748b")
        self.canvas.create_text(cx, cy + 52, text="（作答后直接进入下一题，结果在实验结束时统一报告）",
                                font=("Microsoft YaHei", 10), fill="#94a3b8")
        self.yes_btn.pack(side="left", padx=8)
        self.no_btn.pack(side="left", padx=8)
        self.start_btn.pack_forget()
        self.status.configure(text="凭你的整体印象作答即可。")

    def record_response(self, said_yes):
        if self.phase != "response":
            return
        self.phase = "iti"
        self.cur_trial["said_yes"] = said_yes
        self.cur_trial["rt"] = round((time.perf_counter() - self.cur_trial["stim_off"]) * 1000)
        self.trials.append(self.cur_trial)
        self.yes_btn.pack_forget()
        self.no_btn.pack_forget()
        self.start_btn.pack()
        self.trial_idx += 1
        self.next_trial()                          # 直接进入下一试次，不显示任何作答提示

    def on_key(self, event):
        if self.phase != "response":
            return
        k = event.keysym.lower()
        if k == "f":
            self.record_response(True)
        elif k == "j":
            self.record_response(False)

    def finish_experiment(self):
        self.phase = "done"
        self.start_btn.configure(state="normal", text="再做一次")
        self.progress.configure(text="实验结束")
        self.status.configure(text="实验完成，请查看弹出窗口中的结果与可视化。")
        self.draw_fixation()
        self.show_results(compute_sdt(self.trials))

    # ---------------- 结果窗口 ----------------
    def show_results(self, r):
        win = tk.Toplevel(self.root)
        win.title("实验结果 · 信号检测论")
        win.geometry("1000x760")

        text = tk.Text(win, font=("Microsoft YaHei", 11), relief="flat", bg="#fafbfc",
                       height=24, wrap="word")
        text.pack(fill="x", padx=14, pady=(12, 6))

        lines = [
            "==================== 实验结果 ====================",
            f"击中 Hit        {r['hit']:>3}      P(Hit)  = {r['p_h']:.3f}",
            f"漏报 Miss       {r['miss']:>3}      P(Miss) = {1 - r['p_h']:.3f}",
            f"虚报 FA         {r['fa']:>3}      P(FA)   = {r['p_f']:.3f}",
            f"正确拒绝 CR     {r['cr']:>3}      P(CR)   = {1 - r['p_f']:.3f}",
            "-" * 52,
            f"辨别力   d′ = {r['d_prime']:.2f}",
            f"判断标准 c  = {r['c']:.2f}",
            f"似然比   β  = {r['beta']:.2f}",
            f"回答「有真爱」  {r['n_yes']} / {len(self.trials)} 次（{r['yes_rate'] * 100:.1f}%）",
            f"总体正确率      {r['acc'] * 100:.1f}%",
            "",
        ]
        dp = r["d_prime"]
        dp_desc = ("辨别力很强，你几乎总能区分真爱与路人" if dp >= 2.5 else
                   "辨别力中等，真爱与干扰对你有一定可分辨性" if dp >= 1.2 else
                   "辨别力较弱，判断在很大程度上接近猜测" if dp >= 0.5 else
                   "辨别力接近 0，你基本是在瞎猜（信号与噪声难以区分）")
        c = r["c"]
        c_desc = ("你的判断标准偏保守——更倾向回答「没有」，不易虚报但容易漏报真爱" if c > 0.2 else
                  "你的判断标准偏宽松——更倾向回答「有」，容易找到真爱但也常错把路人当真爱" if c < -0.2 else
                  "你的判断标准居中，没有明显的「有/没有」倾向")
        lines.append(f"结果解读：{dp_desc}。{c_desc}。")
        if abs(c) < 0.005:
            lines.append(f"本次 c 恰好等于 0：你一共回答「有真爱」{r['n_yes']} 次，"
                         f"正好是信号与噪声试次各 {r['n_s']} 次时的一半，此时 z(H) = −z(FA)，"
                         "β 也正好等于 1——这不是计算错误，而是「零偏向」的标准定义。")
        lines.append("说明：c 完全由回答「有」的总比例决定（恰好一半 -> c=0；多于一半 -> c<0 偏宽松；"
                     "少于一半 -> c>0 偏保守），P(S)=0.5 时 c≈0 正是理论最优的中性策略；"
                     "d′ 只反映辨别力，与偏向无关。")
        lines.append("")
        lines.append("==================== 逐试次记录 ====================")
        lines.append(f"{'试次':>4}  {'实际':<6}{'你的判断':<8}{'结果':<8}{'反应时(ms)':>8}")
        for i, t in enumerate(self.trials):
            outcome = ("击中" if t["said_yes"] else "漏报") if t["is_signal"] else \
                      ("虚报" if t["said_yes"] else "正确拒绝")
            lines.append(f"{i + 1:>4}  {'有真爱' if t['is_signal'] else '无真爱':<6}"
                         f"{'有' if t['said_yes'] else '没有':<8}{outcome:<8}{t['rt']:>8}")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

        # ---- 可视化区：d′/c 滑块 + 双分布图 + ROC 曲线 ----
        panel = tk.Frame(win)
        panel.pack(fill="both", expand=True, padx=14, pady=6)
        ctl = tk.Frame(panel)
        ctl.pack(fill="x")
        tk.Label(ctl, text="d′（敏感度）").pack(side="left")
        sl_dp = tk.Scale(ctl, from_=0, to=4, resolution=0.1, orient="horizontal", length=200)
        sl_dp.set(min(4, max(0, r["d_prime"])))
        sl_dp.pack(side="left", padx=8)
        tk.Label(ctl, text="c（判断标准）").pack(side="left", padx=(16, 0))
        sl_c = tk.Scale(ctl, from_=-2, to=2, resolution=0.05, orient="horizontal", length=200)
        sl_c.set(max(-2, min(2, r["c"])))
        sl_c.pack(side="left", padx=8)

        cv_dist = tk.Canvas(panel, width=940, height=290, bg="white",
                            highlightthickness=1, highlightbackground="#e5e7eb")
        cv_dist.pack(pady=(6, 2))
        cv_roc = tk.Canvas(panel, width=460, height=440, bg="white",
                           highlightthickness=1, highlightbackground="#e5e7eb")
        cv_roc.pack(pady=6)
        tk.Label(panel, text="左：信号/噪声双分布与判断标准（拖动滑块观察击中区/虚报区变化）；"
                             "右：ROC 曲线，蓝线为你的 d′，红点为你的操作点。",
                 fg="#6b7280", justify="left", wraplength=940).pack(anchor="w")

        def redraw():
            self._draw_dist(cv_dist, sl_dp.get(), sl_c.get())
            self._draw_roc(cv_roc, r)
        sl_dp.configure(command=lambda _e: redraw())
        sl_c.configure(command=lambda _e: redraw())
        redraw()

    # ---- 双分布图（噪声 N(0,1) 与信号 N(d′,1)，竖直虚线为判断标准 c）----
    def _draw_dist(self, cv, dp, c):
        W, H = 940, 290
        pad_l, pad_r, pad_t, pad_b = 46, 20, 42, 46
        x_min, x_max = -4.0, max(4.0, dp + 4.0)
        X = lambda x: pad_l + (x - x_min) / (x_max - x_min) * (W - pad_l - pad_r)
        Y = lambda y: H - pad_b - y / 0.46 * (H - pad_t - pad_b)
        pdf = lambda x, mu: math.exp(-((x - mu) ** 2) / 2) / math.sqrt(2 * math.pi)
        cv.delete("all")

        def area(mu, fill):
            pts, x = [], max(c, x_min)
            while x <= x_max:
                pts += [X(x), Y(pdf(x, mu))]
                x += 0.05
            pts += [X(x_max), Y(0), X(max(c, x_min)), Y(0)]
            cv.create_polygon(pts, fill=fill, width=0)
        area(dp, "#cdeee8")                        # 击中区（信号 > c）
        area(0, "#fdeeca")                         # 虚报区（噪声 > c）

        def curve(mu, color):
            pts, x = [], x_min
            while x <= x_max:
                pts += [X(x), Y(pdf(x, mu))]
                x += 0.05
            cv.create_line(pts, fill=color, width=2)
        curve(0, "#0ea5e9")
        curve(dp, "#f43f5e")

        cv.create_line(pad_l, Y(0), W - pad_r, Y(0), fill="#cbd5e1")
        t = math.ceil(x_min)
        while t <= x_max:
            cv.create_text(X(t), Y(0) + 16, text=str(t), fill="#64748b", font=("Arial", 9))
            t += 1
        cv.create_text((pad_l + W - pad_r) / 2, H - 10,
                       text="感觉强度（噪声均值 0，信号均值 d′）", fill="#64748b", font=("Arial", 10))
        cv.create_text(X(c), pad_t - 22, text=f"判断标准 c = {c:.2f}",
                       fill="#111827", font=("Arial", 10, "bold"))
        cv.create_line(X(c), pad_t - 8, X(c), Y(0), fill="#111827", dash=(7, 5))
        p_h = 1 - norm_cdf(c - dp)
        p_f = 1 - norm_cdf(c)
        cv.create_text(W - pad_r - 8, pad_t + 6, anchor="e",
                       text=f"理论击中率 P(Hit) = {p_h * 100:.1f}%", fill="#0d9488",
                       font=("Arial", 10, "bold"))
        cv.create_text(W - pad_r - 8, pad_t + 24, anchor="e",
                       text=f"理论虚报率 P(FA) = {p_f * 100:.1f}%", fill="#d97706",
                       font=("Arial", 10, "bold"))
        d_txt = f"{inv_norm(p_h) - inv_norm(p_f):.2f}" if 0 < p_h < 1 and 0 < p_f < 1 else "∞"
        cv.create_text(W - pad_r - 8, pad_t + 42, anchor="e", text=f"d′ = {d_txt}",
                       fill="#64748b", font=("Arial", 9))

    # ---- ROC 曲线族 + 实验操作点 ----
    def _draw_roc(self, cv, r):
        W, H, pad = 460, 440, 52
        X = lambda v: pad + v * (W - pad - 16)
        Y = lambda v: H - pad - v * (H - pad - 20)
        cv.delete("all")
        t = 0.0
        while t <= 1.001:
            cv.create_line(X(t), Y(0), X(t), Y(1), fill="#e2e8f0")
            cv.create_line(X(0), Y(t), X(1), Y(t), fill="#e2e8f0")
            cv.create_text(X(t), Y(0) + 16, text=f"{t:.2f}", fill="#64748b", font=("Arial", 9))
            cv.create_text(X(0) - 16, Y(t), text=f"{t:.2f}", fill="#64748b", font=("Arial", 9))
            t += 0.25
        cv.create_line(X(0), Y(0), X(1), Y(0), fill="#94a3b8")
        cv.create_line(X(0), Y(0), X(0), Y(1), fill="#94a3b8")
        cv.create_text((X(0) + X(1)) / 2, H - 12, text="P(FA) 虚报率",
                       fill="#64748b", font=("Arial", 10))
        cv.create_text(18, (Y(0) + Y(1)) / 2, text="P(Hit) 击中率", fill="#64748b",
                       font=("Arial", 10))

        def roc_curve(dp, color, width):
            pts, c = [], -3.5
            while c <= 3.5:
                pts += [X(1 - norm_cdf(c)), Y(1 - norm_cdf(c - dp))]
                c += 0.04
            cv.create_line(pts, fill=color, width=width, smooth=True)
        for dp in (0, 1, 2, 3):
            roc_curve(dp, "#cbd5e1", 1)
        cv.create_text(X(0.62), Y(0.55), text="d′=0", fill="#94a3b8", font=("Arial", 8))
        cv.create_text(X(0.30), Y(0.62), text="d′=1", fill="#94a3b8", font=("Arial", 8))
        cv.create_text(X(0.135), Y(0.72), text="d′=2", fill="#94a3b8", font=("Arial", 8))
        cv.create_text(X(0.062), Y(0.80), text="d′=3", fill="#94a3b8", font=("Arial", 8))
        roc_curve(r["d_prime"], "#2563eb", 2)
        cv.create_oval(X(r["p_f"]) - 6, Y(r["p_h"]) - 6, X(r["p_f"]) + 6, Y(r["p_h"]) + 6,
                       fill="#e11d63", outline="white", width=2)
        cv.create_text(X(r["p_f"]) + 10, Y(r["p_h"]) - 12, anchor="w",
                       text=f"你的操作点 (FA={r['p_f']:.2f}, Hit={r['p_h']:.2f})",
                       fill="#e11d63", font=("Arial", 9, "bold"))


def main():
    root = tk.Tk()
    root.title("寻找真爱 · 信号检测论实验（Python 版）")
    ExperimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
