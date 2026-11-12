# -*- coding: utf-8 -*-
"""
_style.py - bridge to the thesis figure style.

Prefers the shared module (Microscale/Case_Studies/_shared/figstyle.py) so the
figures produced here obey the same rules as the thesis figures: line charts
BLACK with identity by dash + marker, colour reserved for field maps, curves
direct-labelled instead of boxed in a legend, readable tick density, never a
dual axis.

If that module is not importable (folder moved, running elsewhere) a minimal
compliant fallback is used instead, so no script in this folder ever falls back
to matplotlib's default colour cycle and a legend box.

    from _style import fs, HAVE_FIGSTYLE
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(
    _HERE, "..", "..", "Microscale", "Case_Studies", "_shared"))

HAVE_FIGSTYLE = False
fs = None

if os.path.isdir(_SHARED):
    if _SHARED not in sys.path:
        sys.path.insert(0, _SHARED)
    try:
        import figstyle as fs           # noqa: F401
        HAVE_FIGSTYLE = True
    except Exception as _ex:            # pragma: no cover
        print(f"[style] shared figstyle found but not importable ({_ex}); "
              f"using the local fallback")

if not HAVE_FIGSTYLE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np

    class _Fallback:
        """Minimal, rule-compliant stand-in for figstyle."""
        CANVAS_SCALE = 1.0
        _CYCLE = [dict(color="k", ls="-",              marker="o"),
                  dict(color="k", ls=(0, (5, 2)),      marker="s"),
                  dict(color="k", ls=(0, (1, 1.5)),    marker="^"),
                  dict(color="k", ls=(0, (6, 2, 1, 2)), marker="D"),
                  dict(color="k", ls=(0, (3, 1, 1, 1, 1, 1)), marker="v")]

        def apply(self, **kw):
            _plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                                  "lines.linewidth": 1.2,
                                  "figure.constrained_layout.use": True})

        def canvas(self, print_frac=1.0, aspect=1.0, n_panels=1, **kw):
            w = 6.69*print_frac
            return (w, w/max(aspect, 1e-6)/max(n_panels, 1))

        def line(self, index, **kw):
            if index >= len(self._CYCLE):
                raise ValueError("more than 5 series on a line chart - facet it")
            s = dict(self._CYCLE[index])
            s.update(dict(ms=4, mfc="white", mew=1.0, lw=1.3))
            s.update(kw)
            return s

        def guide(self, ax, x=None, y=None, **kw):
            st = dict(color="0.55", ls=(0, (1, 2)), lw=0.6, zorder=0)
            st.update(kw)
            return ax.axvline(x, **st) if x is not None else ax.axhline(y, **st)

        def label_line(self, ax, x, y, text, side="right", dx=0.02, dy=0.0, **kw):
            i = -1 if side == "right" else 0
            lo, hi = ax.get_xlim()
            xt = x[i] + (hi - lo)*(dx if side == "right" else -dx)
            ha = "left" if side == "right" else "right"
            return ax.text(xt, y[i] + dy, text, ha=ha, va="center", **kw)

        def data_axes(self, ax, nticks=6, axis="both"):
            from matplotlib.ticker import MaxNLocator
            st = [1, 2, 2.5, 5, 10]
            if axis in ("x", "both"):
                ax.xaxis.set_major_locator(MaxNLocator(nticks, steps=st))
            if axis in ("y", "both"):
                ax.yaxis.set_major_locator(MaxNLocator(nticks, steps=st))
            return ax

        def image_axes(self, ax, extent=None, xlabel=None, ylabel=None,
                       nticks=6, show_y=True, **kw):
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nticks))
            ax.yaxis.set_major_locator(MaxNLocator(nticks))
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel and show_y:
                ax.set_ylabel(ylabel)
            return ax

        def add_colorbar(self, ax, mappable, label="", **kw):
            cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            if label:
                cb.set_label(label)
            return cb

        def panel_label(self, ax, index, **kw):
            t = f"({chr(ord('a')+index)})"
            ax.set_xlabel((ax.get_xlabel() + "\n" + t).strip())
            return t

        def symmetric_limits(self, data, percentile=98.0):
            d = _np.asarray(data, float)
            d = d[_np.isfinite(d)]
            if d.size == 0:
                return -1.0, 1.0, 0.0
            lim = float(_np.percentile(_np.abs(d), percentile)) or 1.0
            return -lim, lim, float((_np.abs(d) > lim).mean()*100.0)

        def robust_limits(self, arrays, percentile=1.0):
            st = _np.concatenate([_np.asarray(a, float)[_np.isfinite(a)].ravel()
                                  for a in arrays if a is not None])
            return tuple(_np.percentile(st, [percentile, 100 - percentile]))

        def save(self, fig, directory, stem, formats=("pdf", "png"), **kw):
            out = []
            for ext in formats:
                p = os.path.join(directory, f"{stem}.{ext}")
                fig.savefig(p, dpi=200, bbox_inches="tight")
                out.append(p)
            return out

    fs = _Fallback()
