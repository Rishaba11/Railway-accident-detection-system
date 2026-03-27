# ============================================================
#  launcher.py  –  RAPS  Launcher  (Professor Demo Edition)
#  Run:  python launcher.py
# ============================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess, sys, os, threading

BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
RED    = "#f85149"
RED2   = "#b91c1c"
AMBER  = "#d29922"
GREEN  = "#3fb950"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
WHITE  = "#ffffff"

F_TITLE = ("Segoe UI", 22, "bold")
F_HEAD  = ("Segoe UI", 11, "bold")
F_BODY  = ("Segoe UI", 10)
F_SMALL = ("Segoe UI", 8)
F_BTN   = ("Segoe UI", 11, "bold")


# ── Reusable widgets ─────────────────────────────────────────

class Card(tk.Frame):
    def __init__(self, p, **kw):
        super().__init__(p, bg=PANEL,
                         highlightbackground=BORDER,
                         highlightthickness=1, **kw)

class DarkEntry(tk.Entry):
    def __init__(self, p, **kw):
        super().__init__(p, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                         relief="flat", highlightbackground=BORDER,
                         highlightthickness=1, font=F_BODY, **kw)

class DarkSpin(tk.Spinbox):
    def __init__(self, p, **kw):
        super().__init__(p, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                         buttonbackground=PANEL, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1,
                         font=F_BODY, **kw)

class RedButton(tk.Button):
    def __init__(self, p, text, cmd, **kw):
        super().__init__(p, text=text, command=cmd,
                         bg=RED, fg=WHITE, activebackground=RED2,
                         activeforeground=WHITE, relief="flat",
                         cursor="hand2", font=F_BTN, bd=0, **kw)
        self.bind("<Enter>", lambda e: self.config(bg=RED2))
        self.bind("<Leave>", lambda e: self.config(bg=RED))

class GhostBtn(tk.Button):
    def __init__(self, p, text, cmd, **kw):
        super().__init__(p, text=text, command=cmd,
                         bg=PANEL, fg=MUTED, activebackground=BG,
                         activeforeground=TEXT, relief="flat",
                         cursor="hand2", font=F_BODY,
                         highlightbackground=BORDER, highlightthickness=1,
                         bd=0, **kw)
        self.bind("<Enter>", lambda e: self.config(fg=TEXT))
        self.bind("<Leave>", lambda e: self.config(fg=MUTED))

def sec_label(parent, text):
    tk.Label(parent, text=text.upper(), font=("Segoe UI", 8, "bold"),
             fg=MUTED, bg=BG, anchor="w").pack(anchor="w", pady=(0, 4))

def divider(parent):
    tk.Frame(parent, height=1, bg=BORDER).pack(fill="x", pady=14)


# ============================================================

class RAPSLauncher(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("RAPS — Railway Accident Prevention System")
        self.geometry("680x560")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._source   = tk.StringVar(value="file")
        self._filepath = tk.StringVar(value="")
        self._cam_idx  = tk.IntVar(value=0)
        self._far_ppm  = tk.StringVar(value="30")
        self._near_ppm = tk.StringVar(value="500")
        self._status   = tk.StringVar(value="Ready")
        self._running  = False
        self._proc     = None

        self._build()
        self._refresh_source()

    # ── UI ────────────────────────────────────────────────────

    def _build(self):
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=24, pady=20)

        # Header
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill="x", pady=(0, 16))
        tk.Label(hdr, text="RAPS", font=F_TITLE, fg=RED, bg=BG).pack(side="left")
        tk.Label(hdr, text="  Railway Accident Prevention System",
                 font=F_BODY, fg=MUTED, bg=BG).pack(side="left", pady=(8, 0))
        badge = tk.Frame(hdr, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1)
        badge.pack(side="right", pady=(6, 0))
        tk.Label(badge, text="  YOLOv8 + OpenCV  ", font=("Segoe UI", 8, "bold"),
                 fg=AMBER, bg=PANEL).pack(padx=4, pady=4)

        divider(root)

        # ── Source section ───────────────────────────────────
        sec_label(root, "Input Source")
        src_card = Card(root)
        src_card.pack(fill="x", pady=(0, 14))

        rrow = tk.Frame(src_card, bg=PANEL)
        rrow.pack(fill="x", padx=12, pady=(10, 6))
        tk.Radiobutton(rrow, text="  Video File", variable=self._source,
                       value="file", font=F_HEAD, fg=TEXT, bg=PANEL,
                       selectcolor=BG, activebackground=PANEL,
                       activeforeground=RED, cursor="hand2",
                       command=self._refresh_source).pack(side="left")
        tk.Radiobutton(rrow, text="  Live Camera", variable=self._source,
                       value="camera", font=F_HEAD, fg=TEXT, bg=PANEL,
                       selectcolor=BG, activebackground=PANEL,
                       activeforeground=RED, cursor="hand2",
                       command=self._refresh_source).pack(side="left", padx=(24, 0))

        # file picker row (hidden/shown)
        self._file_row = tk.Frame(src_card, bg=PANEL)
        fr = tk.Frame(self._file_row, bg=PANEL)
        fr.pack(fill="x", padx=12, pady=(0, 10))
        self._path_e = DarkEntry(fr, textvariable=self._filepath, width=46)
        self._path_e.pack(side="left", ipady=5, fill="x", expand=True)
        GhostBtn(fr, "Browse…", self._browse).pack(side="left", padx=(8,0),
                                                    ipady=4, ipadx=10)

        # camera row (hidden/shown)
        self._cam_row = tk.Frame(src_card, bg=PANEL)
        cr = tk.Frame(self._cam_row, bg=PANEL)
        cr.pack(padx=12, pady=(0, 10), anchor="w")
        tk.Label(cr, text="Camera index:", font=F_BODY, fg=MUTED,
                 bg=PANEL).pack(side="left")
        DarkSpin(cr, textvariable=self._cam_idx, from_=0, to=10,
                 width=4).pack(side="left", padx=8, ipady=4)
        tk.Label(cr, text="(0 = built-in webcam)", font=F_SMALL,
                 fg=MUTED, bg=PANEL).pack(side="left")

        # ── Calibration section ──────────────────────────────
        sec_label(root, "Perspective Calibration")
        cal_card = Card(root)
        cal_card.pack(fill="x", pady=(0, 14))
        cal_body = tk.Frame(cal_card, bg=PANEL)
        cal_body.pack(fill="x", padx=12, pady=10)

        fields = [
            ("FAR_PPM",  self._far_ppm,
             "Pixels/metre when train is far (top of frame). Default: 30"),
            ("NEAR_PPM", self._near_ppm,
             "Pixels/metre when train fills frame. Default: 500"),
        ]
        for col, (name, var, tip) in enumerate(fields):
            g = tk.Frame(cal_body, bg=PANEL)
            g.grid(row=0, column=col, padx=(0, 30), sticky="nw")
            tk.Label(g, text=name, font=("Segoe UI", 9, "bold"),
                     fg=TEXT, bg=PANEL).pack(anchor="w")
            tk.Label(g, text=tip, font=F_SMALL, fg=MUTED,
                     bg=PANEL, wraplength=270, justify="left").pack(anchor="w",
                                                                    pady=(1, 4))
            DarkEntry(g, textvariable=var, width=8).pack(anchor="w", ipady=4)

        hint_row = tk.Frame(cal_card, bg=PANEL)
        hint_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(hint_row,
                 text="ⓘ  Pre-calibrated for the uploaded 1920×1080 video. "
                      "Change only if using a different camera angle.",
                 font=F_SMALL, fg=MUTED, bg=PANEL,
                 wraplength=620, justify="left").pack(anchor="w")

        divider(root)

        # ── Launch row ───────────────────────────────────────
        btn_row = tk.Frame(root, bg=BG)
        btn_row.pack(fill="x")

        self._launch_btn = RedButton(btn_row, "▶   Launch System",
                                     self._launch)
        self._launch_btn.pack(side="left", ipady=10, ipadx=24)

        self._stop_btn = GhostBtn(btn_row, "■  Stop", self._stop)
        self._stop_btn.pack(side="left", padx=(10, 0), ipady=9, ipadx=16)
        self._stop_btn.config(state="disabled")

        # ── Status bar ───────────────────────────────────────
        sb = tk.Frame(root, bg=BG)
        sb.pack(fill="x", pady=(12, 0))
        self._dot = tk.Label(sb, text="●", font=("Segoe UI", 9),
                              fg=MUTED, bg=BG)
        self._dot.pack(side="left")
        tk.Label(sb, textvariable=self._status, font=("Segoe UI", 9),
                 fg=MUTED, bg=BG).pack(side="left", padx=6)

        self.bind("<Return>", lambda e: self._launch())

    # ── helpers ──────────────────────────────────────────────

    def _refresh_source(self, *_):
        if self._source.get() == "file":
            self._cam_row.pack_forget()
            self._file_row.pack(fill="x")
        else:
            self._file_row.pack_forget()
            self._cam_row.pack(fill="x")

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        if p:
            self._filepath.set(p)
            self._setstatus(f"Selected: {os.path.basename(p)}", GREEN)

    def _setstatus(self, msg, color=None):
        self._status.set(msg)
        if color:
            self._dot.config(fg=color)

    # ── launch / stop ─────────────────────────────────────────

    def _launch(self):
        if self._running:
            return

        src = self._source.get()
        if src == "file":
            path = self._filepath.get().strip()
            if not path:
                messagebox.showwarning("No file", "Please browse for a video file first.")
                return
            if not os.path.exists(path):
                messagebox.showerror("Not found", f"Cannot find:\n{path}")
                return
            vsrc = path
        else:
            vsrc = str(self._cam_idx.get())

        try:
            far  = float(self._far_ppm.get())
            near = float(self._near_ppm.get())
            assert far > 0 and near > 0 and far < near
        except Exception:
            messagebox.showerror("Bad calibration",
                "FAR_PPM and NEAR_PPM must be positive numbers,\n"
                "with FAR_PPM < NEAR_PPM.")
            return

        self._running = True
        self._launch_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._setstatus("System running — see the OpenCV window", GREEN)

        # Scale the quadratic PPM curve to the user-supplied far/near values
        patch = (
            "import main as _m\n"
            f"_m.VIDEO_SOURCE = {vsrc!r}\n"
            f"_m.PPM_MIN      = {far}\n"
            f"_m.PPM_A        = -0.00128\n"
            f"_m.PPM_B        =  1.082  * ({near}/500.0)\n"
            f"_m.PPM_C        =  276.5  * ({far}/30.0)\n"
            "_m.main()\n"
        )

        def _run():
            self._proc = subprocess.Popen(
                [sys.executable, "-c", patch],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in self._proc.stdout:
                print(line, end="")
            rc = self._proc.wait()
            self.after(0, self._proc_done, rc)

        threading.Thread(target=_run, daemon=True).start()

    def _stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _proc_done(self, rc):
        self._running = False
        self._launch_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        if rc == 0:
            self._setstatus("Session ended.", GREEN)
        else:
            self._setstatus(f"Process exited (code {rc}).", AMBER)


# ============================================================

if __name__ == "__main__":
    app = RAPSLauncher()
    app.mainloop()