#!/usr/bin/env python3
# File       : mpl_latex.py
# Created    : Tue Aug 27 2024 10:03:58 (+0200)
# Author     : Fabian Wermelinger
# Description: Matplotlib utilities for interfacing with LaTeX

import shlex
import re
import os
import glob
import math
import inspect
import cycler
import subprocess as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.markers import MarkerStyle

class MyMarkerStyle(MarkerStyle):

    def __init__(self, marker, *args, **kwargs):
        kwargs.setdefault("joinstyle", "round")
        kwargs.setdefault("capstyle", "round")
        super().__init__(marker, *args, **kwargs)


def getAxes_(
    spec,
    fig,
    nrows,
    ncols,
    *,
    flatten_handles=False,
    sharex=False,
    sharey=False,
    visibley=False,
    visiblex=False,
    **kwargs
):
    ax = []
    for i in range(nrows):
        acol = []
        for j in range(ncols):
            acol.append(fig.add_subplot(spec[i, j], **kwargs))
        ax.append(acol)
    if sharex:
        for j in range(ncols):
            col = []
            for i in range(nrows):
                col.append(ax[i][j])
            for aa in col[1:]:
                aa.sharex(col[0])
            if not visiblex:
                for aa in (col[::-1])[1:]:
                    plt.setp(aa.get_xticklabels(), visible=False)
    if sharey:
        for i in range(nrows):
            for aa in ax[i][1:]:
                aa.sharey(ax[i][0])
            if not visibley:
                for aa in ax[i][1:]:
                    plt.setp(aa.get_yticklabels(), visible=False)
    if nrows == 1 or ncols == 1 or flatten_handles:
        # C-like
        ax = [x for col in ax for x in col]
        if nrows == 1 and ncols == 1:
            ax = ax[0]
    return ax


def figSize(
    width, *args, fraction=1, aspect_ratio=None, aspect_scale=(1.0, 1.0)
):
    # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    nrows: int
            Number of rows (if subplots)
    ncols: int
            Number of columns (if subplots)
    fraction: float
            Fraction of the width which you wish the figure to occupy
    aspect_ratio: float
            User specified figure aspect ratio.  If `None` use golden ratio
    aspect_scale: (float, float)
            Additional scaling for (width, height) figure size after aspect ratio is applied

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # pre-defined widths
    if width == 'article':
        width = 360.0
    elif width == 'report':
        width = 360.0

    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        # Golden ratio to set aesthetic figure height
        hratio = (5**.5 - 1) / 2
    else:
        # ratio is specified as width/height, e.g. 16/9, 4/3 etc.
        hratio = 1 / aspect_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * hratio

    return (fig_width_in * aspect_scale[0], fig_height_in * aspect_scale[1])


def fontConf(fontsize, subfontsize=None, *args, update=True):
    """Setup fontsizes for LaTeX integration without scaling"""
    if subfontsize is None:
        subfontsize = fontsize - 2  # 2pt smaller

    conf = {
        # font sizes
        "font.size": subfontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": subfontsize,
        "xtick.labelsize": subfontsize,
        "ytick.labelsize": subfontsize,
        "figure.titlesize": fontsize
    }

    if update:
        mpl.rcParams.update(conf)

    return conf


def getFigure(
    nrows=1,
    ncols=1,
    *args,
    width='report',
    fontsize=10,
    aspect_ratio=None,
    aspect_scale=(1.0, 1.0),
    fraction=1.0,
    figonly=False,
    fig=None,
    left=None,
    bottom=None,
    right=None,
    top=None,
    wspace=None,
    hspace=None,
    width_ratios=None,
    height_ratios=None,
    figsize=None,
    **kwargs
):
    """Get a set of figure and axes handles with default style and configuration
    set up properly (convenience function)"""
    fontConf(fontsize)

    if fig is None:
        size = figsize
        if size is None:
            size = figSize(
                width,
                aspect_ratio=aspect_ratio,
                aspect_scale=aspect_scale,
                fraction=fraction
            )
        fig = plt.figure(figsize=size)
    if figonly:
        return fig
    spec = fig.add_gridspec(
        nrows,
        ncols,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios
    )
    ax = getAxes_(spec, fig, nrows, ncols, **kwargs)
    return fig, ax


def getCycler(which=None):
    """Return individual color, linestyle and marker cyclers depending on
    mpl.rcParams"""
    N = len(mpl.rcParams['axes.prop_cycle'])  # number of elements in cycler
    kcycler = mpl.rcParams['axes.prop_cycle'].by_key()
    split = {}
    keys = ['color', 'c', 'linestyle', 'ls', 'marker']
    for k in keys:
        if k in kcycler:
            split[k] = cycler.cycler(k, kcycler[k])
    if 'marker' not in split:
        markers = [MyMarkerStyle(m) for m in list('os^d<*pv8h>DPHX')]
        assert N <= len(markers)
        split['marker'] = cycler.cycler('marker', markers[:N])
    if which is None:
        return split
    return split[which]


def getLinestyles():
    """Get list of linestyles defined in cycler"""
    return [x['linestyle'] for x in getCycler('linestyle')]


def getMarkers():
    """Get list of markers defined in cycler"""
    return [x['marker'] for x in getCycler('marker')]


def getStickmanMarkers():
    """Get list of stickman markers not defined in cycler"""
    return [MyMarkerStyle(m) for m in list('+1x2_3|4')]


def getColors():
    """Get list of colors defined in cycler"""
    return [x['color'] for x in getCycler('color')]


def getMonokai():
    # https://chriskempson.github.io/base16/
    return [
        '#272822', '#383830', '#49483E', '#75715E', '#A59F85', '#F8F8F2',
        '#F5F4F1', '#F9F8F5', '#F92672', '#66D9EF', '#A6E22E', '#FD971F',
        '#AE81FF', '#CC6633', '#A1EFE4', '#F4BF75'
    ]


def getPastel():
    # pastel colors used in defense slides
    return [
        '#272822', '#383830', '#49483E', '#75715E', '#A59F85', '#F8F8F2',
        '#F5F4F1', '#F9F8F5', '#E8A59F', '#A1B9CF', '#B9D6B2', '#EBC896',
        '#CBB9D1', '#EBEBB9', '#D1C5AB', '#EBC9DA'
    ]


def getGrayscale():
    """Get list of gray scale colors"""
    return ['#000000', '#808080', '#A9A9A9', '#D3D3D3', '#F5F5F5', '#FFFFFF']


def getPlotStyles():
    """Get lists for linestyles, colors and markers"""
    return (getLinestyles(), getColors(), getMarkers())


def exportKeys(
    lcycler,
    mcycler,
    ccycler=None,
    *args,
    fontsize=11,
    handlelength=38,
    linewidth=1.4,
    markersize=4.0,
    margins=[0.0, 0.0, 0.0, 0.0],
    outdir='./keys'
):
    """
    Export legend key entries into chopped up pdf pieces, to be used in LaTeX
    captions etc.

    Arguments:
        lcycler: line cycler
        mcycler: marker cycler
        ccycler: color cycler
        fontsize: relative fontsize to be used for keys
        handlelength: length of key handles in pt
        linewidth: linewidth to be used for keys
        markersize: marker size to be used for keys
        margins: additional margins for ghostscript `[left, top, right, bottom]`
        outdir: output directory for keys
    """
    if linewidth is not None:
        mpl.rcParams['lines.linewidth'] = linewidth
    if markersize is not None:
        mpl.rcParams['lines.markersize'] = markersize
    # truncate to the shortest cycler, so mismatched lengths remain usable
    n_min = min(len(c) for c in (lcycler, mcycler, ccycler) if c is not None)
    lcycler = lcycler[:n_min]
    mcycler = mcycler[:n_min]
    kcycler = cycler.cycler(color=['k' for x in range(n_min)])  # black

    prefix = '012345tmp'

    # determine font height from fontsize and compute bounding box height dy
    minimal = inspect.cleandoc(
        r"""
    \documentclass[a4paper, fontsize={fontsize:d}pt]{{article}}
    \begin{{document}}
    \newlength{{\fheight}}
    \settoheight{{\fheight}}{{f}}
    \showthe\fheight
    \end{{document}}
    """.format(fontsize=fontsize)
    )
    fname = prefix + '.tex'
    with open(fname, 'w') as out:
        out.write(minimal)
    cmd = "pdflatex -interaction nonstopmode {fname:s}".format(fname=fname)
    out = sp.run(shlex.split(cmd), capture_output=True).stdout.decode('utf-8')
    dy = float(re.search(r'^>\s*([0-9\.]*)pt\.$', out, re.MULTILINE).group(1))
    for trash in glob.glob(prefix + '*'):  # clean up
        os.remove(trash)

    # export keys to outdir (requires pdfcrop)
    predir = os.path.join(outdir, 'preview')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isdir(predir):
        os.mkdir(predir)

    ftmp = prefix + 'key.pdf'

    def make_key(key, styles, *, size=18):
        """
        Helper function to generate specific key

        Arguments:
            key: name of the key
            linestyle: line style definitions
            markerstyle: marker style definitions
            size: figure size in points
        """
        dst = os.path.join(outdir, key + '.pdf')
        # generate key
        figsize = size / 72.27
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        x0 = 1
        if 'line' in styles:
            ax.plot((-x0, x0), (0, 0), **styles['line'])
        if 'marker' in styles:
            ax.plot((0, 0), (0, 0), **styles['marker'])
        ax.set_axis_off()
        dx = 0.1
        ax.set_xlim([-(x0 + dx), x0 + dx])
        fig.savefig(ftmp, bbox_inches='tight')
        plt.close(fig)

        # compute hires bounding box via pdfcrop
        cmd = "pdfcrop --hires --verbose {fname:s} /dev/null".format(fname=ftmp)
        out = sp.check_output(shlex.split(cmd)).decode('utf-8')
        bbox = re.search(
            r'^\s*%%HiResBoundingBox:\s*([0-9\s\.]*)$', out, re.MULTILINE
        ).group(1)
        bbox = [float(x) for x in bbox.split()]

        # crop and save
        cx = bbox[0]
        dx = bbox[2] - cx
        cy = 0.5 * (bbox[1] + bbox[3] - dy)
        if dx < 1:  # too small for ghost script
            cx -= 0.5 * (1 - dx)
            dx = 1
        cmd = "pdfcrop --hires --bbox '{xs:f} {ys:f} {xe:f} {ye:f}' --margins '{mlr:f} {mtb:f}' {fname:s} {dst:s}".format(
            xs=cx,
            ys=cy,
            xe=cx + dx,
            ye=cy + dy,
            mlr=margins[0],  # margins left/right
            mtb=margins[1],  # margins top/bottom
            fname=ftmp,
            dst=dst
        )
        sp.run(shlex.split(cmd))

    with open(os.path.join(predir, 'preview.tex'), 'w') as preview:
        nolcycler = cycler.cycler(
            linestyle=['none' for x in range(len(mcycler))]
        )
        smcycler = cycler.cycler(
            marker=getStickmanMarkers()[:n_min]
        )  # light stickman markers
        gcycler = cycler.cycler(color=getGrayscale()[1:-1][:n_min])  # gray scales
        colors = {'k': kcycler, 'g': gcycler}
        if ccycler is not None:
            colors['c'] = ccycler[:n_min]

        def write_preview(cid, sid, styles, *, offset=0, **kwargs):
            for i, style in enumerate(styles):
                key = '{id:02d}{cid:s}{sid:s}'.format(
                    id=offset + i + 1, cid=cid, sid=sid
                )
                preview.write(
                    "\\item[\\texttt{{{key:s}}}:] fff \\tkey{{{key:s}}}~Xxx (\\tkey{{{key:s}}})\n"
                    .format(key=key)
                )
                make_key(key, style, **kwargs)

        texbuf = inspect.cleandoc(
            r"""
        \documentclass[a4paper, twocolumn, fontsize={fontsize:d}pt]{{article}}
        \usepackage{{graphicx}}
        \usepackage{{mathpazo}}
        \newcommand{{\tkey}}[1]{{\protect\raisebox{{-0.8pt}}{{\protect\includegraphics{{#1.pdf}}}}}}
        \graphicspath{{{{../}}}}
        \begin{{document}}
        \begin{{itemize}}
        """.format(fontsize=fontsize)
        )
        preview.write(texbuf + '\n')
        preview.write(r"\section*{Primary keys}" + '\n')
        # additive combinations (primary keys)
        for cid, cc in colors.items():
            n = len(cc)
            assert n <= len(lcycler)
            assert n <= len(mcycler)
            assert n <= len(nolcycler)
            assert n <= len(smcycler)
            # lines
            write_preview(
                cid,
                'l', [{
                    'line': l
                } for l in cc + lcycler[:n]],
                size=handlelength
            )

            # markers
            write_preview(
                cid,
                'm', [{
                    'marker': m
                } for m in cc + mcycler[:n] + nolcycler[:n]],
                size=handlelength
            )

            # lines + markers
            write_preview(
                cid,
                'lm', [
                    {
                        'line': l,
                        'marker': m
                    } for l, m in zip(cc + lcycler[:n], cc + mcycler[:n])
                ],
                size=handlelength
            )

            # stickman markers
            write_preview(
                cid,
                'sm',
                [{
                    'marker': m
                } for m in cc + smcycler[:n] + nolcycler[:n]],
                size=handlelength
            )
        # multiplicative color-line and color-marker combinations (secondary
        # keys)
        preview.write(r"\section*{Secondary keys}" + '\n')
        for k, v in colors.items():
            if k == 'c' or k == 'g':  # only for colors and gray scales
                assert v is not None
                n = len(v)
                # color-line
                write_preview(
                    k,
                    'l', [{
                        'line': l
                    } for l in v * lcycler],
                    offset=n + 1,
                    size=handlelength
                )

                # color-marker
                write_preview(
                    k,
                    'm', [{
                        'marker': m
                    } for m in v * (mcycler + nolcycler)],
                    offset=n + 1,
                    size=handlelength
                )

        # stickman markers
        preview.write(r"\section*{Stickman markers}" + '\n')
        for k, v in colors.items():
            if k == 'c' or k == 'g':  # only for colors and gray scales
                assert v is not None
                n = len(v)
                write_preview(
                    k,
                    'sm', [{
                        'marker': m
                    } for m in v * (smcycler + nolcycler)],
                    offset=n + 1,
                    size=handlelength
                )
        preview.write(r"\end{itemize}\end{document}")
    os.remove(ftmp)  # clean up


def latexTextwidth(class_name, fontsize=10):
    """Create a minimal LaTeX document to determine the textwidth in pt used for
    the given class"""
    minimal = inspect.cleandoc(
        r"""
    \documentclass[a4paper, fontsize={fontsize:d}]{{{class_name:s}}}
    \begin{{document}}
    % gives the width of the current document in pts (full)
    % for 2-columns: \showthe\columnwidth
    % \showthe\linewidth
    \showthe\textwidth
    \end{{document}}
    """.format(class_name=class_name, fontsize=fontsize)
    )

    with open('latexTextwidth.tex', 'w') as out:
        out.write(minimal)


def annotate(ax, label, *, xpos=-0.15, ypos=-0.2):
    """
    Put a figure label annotation

    Arguments:
        ax: axes artist
        label: figure label
        xpos: x-coordinate for label in axes coordinates
        ypos: y-coordinate for label in axes coordinates
    """
    fsize = mpl.rcParams['axes.labelsize']
    ax.annotate(
        f"({label})", (xpos, ypos),
        xycoords='axes fraction',
        va='center',
        ha='center',
        fontsize=fsize
    )


def getBasename(fpath, *, suffix=True):
    """
    Return the basename of path `fpath`

    Arguments:
        fpath: File path
        suffix: If `False` strip off file suffix
    """
    head = os.path.basename(fpath)
    if not suffix:
        head = head.split('.')[0]
    return head


def savefig(
    fig, path, *args, bbox_inches='tight', pad_inches=1 / 72.27, **kwargs
):
    """Wrapper around `fig.savefig(...)` with common parameters."""
    # fig.tight_layout(pad=0.4) # or use: figure.constrained_layout.use: True
    fig.savefig(
        path, *args, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs
    )


def closefig(fig=None):
    plt.close(fig)
