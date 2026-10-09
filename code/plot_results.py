"""Reproducible Python rendering of frozen pulse-selection evidence.

Run: python plot_results.py
Only the clean tables bundled beside this script are read. No inference is run.
"""
from pathlib import Path
import os
import hashlib
import json
import string
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('CO_SOURCE_DATA', ROOT / 'source_data'))
RESULT_ROOT = Path(os.environ.get('CO_RESULTS_DIR', ROOT))
OUT = RESULT_ROOT / 'figures'
QA = RESULT_ROOT / 'qa'
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.55, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.labelcolor': '#202124', 'text.color': '#202124',
    'xtick.color': '#202124', 'ytick.color': '#202124',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.unicode_minus': True,
})

METHODS = ['FIRST', 'LAST', 'T', 'W', 'TW']
COLORS = dict(zip(METHODS, ['#575B65', '#9CA1A9', '#758AB4', '#B18160', '#214F72']))
MARKERS = dict(zip(METHODS, ['v', '^', 's', 'D', 'o']))
STYLES = {'FIRST': '--', 'LAST': ':', 'T': '-.', 'W': '--', 'TW': '-'}
METRICS = ['P_cor', 'P_C_given_C', 'P_C_given_E', 'Mean_L_NC']
TITLES = ['Correct selection', 'Correct-state persistence', 'Recovery from error', 'Observed non-correct run']
YLABELS = ['P(cor) (%)', 'P(C | C) (%)', 'P(C | E) (%)', 'Mean observed NC length (cycles)']
SHORT = ['P(cor)', 'P(C | C)', 'P(C | E)', 'Obs. NC length']
FILES = []
ARTIST_QA = []
CAPTIONS = {
    'fig01_hprf': 'HPRF-only interference',
    'fig02_f_only': 'F-only interference',
    'fig03_composite': 'Composite interference',
    'fig04_tw_minus_t': 'Paired differences: TW minus T',
    'fig05_tw_minus_w': 'Paired differences: TW minus W',
    'fig06_width_strata': 'True-width separation at 300 kHz',
    'figS01_precision': 'Precision audit: all 216 prespecified checks',
}


def load_data():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    tables = {}
    for name, info in manifest['datasets'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == info['sha256'], name
        tables[name] = pd.read_csv(DATA / name, float_precision='round_trip')
        assert len(tables[name]) == info['rows']
    individual = tables['individual.csv']
    paired = tables['paired.csv']
    assert len(individual) == 360 and len(paired) == 144
    assert not individual.duplicated(['ConditionOrdinal', 'MethodOrdinal', 'Metric']).any()
    assert not paired.duplicated(['ConditionOrdinal', 'ContrastOrdinal', 'Metric']).any()
    assert set(individual.ConditionOrdinal) == set(range(1, 19))
    assert individual.PointDefined.all() and individual.CI95Defined.all()
    assert paired.DeltaPointDefined.all() and paired.CI95Defined.all()
    for frame, estimate in [(individual, 'PointEstimate'), (paired, 'DeltaPoint')]:
        assert np.isfinite(frame[[estimate, 'CI95Lower', 'CI95Upper']]).all().all()
        assert (frame.CI95Lower <= frame.CI95Upper).all()
    assert np.allclose(individual.PointEstimate, individual.FormalSumA / individual.FormalSumB, rtol=1e-13, atol=1e-15)
    assert np.allclose(paired.DeltaPoint, paired.PointEstimateTW - paired.PointEstimateComparator, atol=1e-14)
    assert (individual.BootstrapB == 2000).all() and (paired.BootstrapB == 2000).all()
    assert len(tables['precision_limitations.csv']) == 6
    assert (~tables['precision.csv'].MeetsStage8PrecisionTarget).sum() == 6
    return tables


def mark_panel(ax, letter):
    ax.text(-0.16, 1.065, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='left', va='bottom')


def method_handles():
    return [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], marker=MARKERS[m],
                   lw=1.05, ms=3.5, markerfacecolor='white' if m in METHODS[:2] else COLORS[m],
                   markeredgewidth=0.65, label=m) for m in METHODS]


def top(fig, title, subtitle, methods=True):
    fig.text(.08, .976, title, fontsize=7, weight='bold', va='top')
    fig.text(.08, .950, subtitle, fontsize=6, va='top')
    if methods:
        fig.legend(handles=method_handles(), loc='upper right', bbox_to_anchor=(.985, .988),
                   ncol=5, frameon=False, handlelength=2.1, columnspacing=1.05)


def save(fig, stem):
    # Locators may create undrawn off-axis tick artists; retain only in-view ticks.
    for ax in fig.axes:
        if isinstance(ax.xaxis.get_major_locator(), MaxNLocator):
            xmin, xmax = sorted(ax.get_xlim())
            ticks = ax.get_xticks()
            ax.xaxis.set_major_locator(FixedLocator(ticks[(ticks >= xmin) & (ticks <= xmax)]))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
            outside.append(text.get_text())
    ARTIST_QA.append({'figure': stem, 'text_outside_canvas': outside,
                      'width_mm': fig.get_figwidth()*25.4, 'height_mm': fig.get_figheight()*25.4})
    fig.savefig(OUT / f'{stem}.svg')
    fig.savefig(OUT / f'{stem}.pdf', metadata={'Title': CAPTIONS[stem], 'Creator': 'Python matplotlib'})
    fig.savefig(OUT / f'{stem}.png', dpi=300)
    fig.savefig(OUT / f'{stem}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    FILES.append(stem)
    plt.close(fig)


def plot_series(ax, data, metric, xcol, title=True):
    idx = METRICS.index(metric)
    factor = 1 if idx == 3 else 100
    spacing = min(np.diff(sorted(data[xcol].unique())))
    for mi, method in enumerate(METHODS):
        sub = data[(data.MethodCode == method) & (data.Metric == metric)].sort_values(xcol)
        x = sub[xcol].to_numpy(float)
        y = sub.PointEstimate.to_numpy(float) * factor
        lo = sub.CI95Lower.to_numpy(float) * factor
        hi = sub.CI95Upper.to_numpy(float) * factor
        # Small deterministic marker offsets reveal overlapping methods at one condition.
        xp = x + (mi - 2) * spacing * .018
        ax.plot(x, y, color=COLORS[method], ls=STYLES[method], lw=1.05, zorder=2+mi)
        ax.vlines(xp, lo, hi, color=COLORS[method], lw=.65, zorder=4+mi)
        ax.plot(xp, y, ls='none', marker=MARKERS[method], ms=3.5,
                markerfacecolor='white' if mi < 2 else COLORS[method],
                markeredgecolor=COLORS[method], markeredgewidth=.65, zorder=5+mi)
    if title:
        ax.set_title(TITLES[idx], loc='left', pad=9, weight='bold')
    ax.set_ylabel(YLABELS[idx])
    vals = sorted(data[xcol].unique())
    ax.set_xticks(vals)
    ax.set_xlim(min(vals)-spacing*.18, max(vals)+spacing*.18)
    if metric == 'Mean_L_NC':
        ax.set_yscale('log')
        ax.set_ylim(.85, 255)
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 50, 200]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:g}'))
        ax.yaxis.set_minor_locator(FixedLocator([]))
    else:
        ax.set_ylim(-3, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(pad=2)


def simple_scene(individual, scene, stem, xcol, xlabel):
    fig, axs = plt.subplots(2, 2, figsize=(183/25.4, 138/25.4))
    fig.subplots_adjust(left=.105, right=.976, bottom=.12, top=.85, hspace=.65, wspace=.38)
    data = individual[individual.Scene == scene]
    top(fig, CAPTIONS[stem], 'n = 2,000 repeats per condition; 200 cycles per repeat; 95% bootstrap intervals')
    for idx, ax in enumerate(axs.flat):
        plot_series(ax, data, METRICS[idx], xcol)
        mark_panel(ax, string.ascii_lowercase[idx])
        ax.set_xlabel(xlabel)
    fig.text(.105, .031, 'Markers are slightly offset for visibility; lines use the true condition values. Run-length axis is logarithmic.', fontsize=5.7)
    save(fig, stem)


def composite(individual):
    stem = 'fig03_composite'
    fig, axs = plt.subplots(4, 3, figsize=(183/25.4, 205/25.4))
    fig.subplots_adjust(left=.115, right=.975, bottom=.09, top=.88, hspace=.61, wspace=.29)
    top(fig, CAPTIONS[stem], 'n = 2,000 repeats per condition; 200 cycles per repeat; 95% bootstrap intervals')
    for col, rate in enumerate([200, 300, 500]):
        data = individual[(individual.Scene == 'COMPOSITE') & (individual.HPRF_kHz == rate)]
        for row, metric in enumerate(METRICS):
            ax = axs[row, col]
            plot_series(ax, data, metric, 'FDelay_us', title=False)
            mark_panel(ax, string.ascii_lowercase[row*3+col])
            if row == 0:
                ax.set_title(f'HPRF {rate} kHz', loc='left', weight='bold', pad=10)
            if col:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)
            if row == 3:
                ax.set_xlabel('F delay (μs)')
    fig.text(.115, .039, 'All nine composite conditions are shown. Small horizontal marker offsets reveal overlapping methods.', fontsize=5.7)
    fig.text(.115, .023, 'Run-length axes are logarithmic; probabilities use the same 0–100% scale across conditions.', fontsize=5.7)
    save(fig, stem)


def forests(paired, contrast, stem):
    detail = contrast == 'TW-T'
    fig, axs = plt.subplots(1, 4, figsize=(183/25.4, (189 if detail else 145)/25.4), sharey=True)
    fig.subplots_adjust(left=.145, right=.975, bottom=.355 if detail else .145, top=.86 if detail else .845, wspace=.33)
    top(fig, CAPTIONS[stem], 'Paired point estimates and pointwise 95% bootstrap intervals; all 18 interference conditions', methods=False)
    conditions = paired[['ConditionOrdinal', 'ConditionCode']].drop_duplicates().sort_values('ConditionOrdinal')
    yy = np.r_[np.arange(5), np.arange(4)+5.8, np.arange(9)+10.6]
    for mi, metric in enumerate(METRICS):
        ax = axs[mi]
        sub = paired[(paired.ContrastCode == contrast) & (paired.Metric == metric)].sort_values('ConditionOrdinal')
        factor = 1 if mi == 3 else 100
        y = sub.DeltaPoint.to_numpy(float) * factor
        lo = sub.CI95Lower.to_numpy(float) * factor
        hi = sub.CI95Upper.to_numpy(float) * factor
        ax.axhspan(5.3, 9.3, color='#F2F4F7', zorder=0)
        ax.axvline(0, color='#6E7177', lw=.65, ls=(0, (3, 3)), zorder=1)
        ax.scatter(y, yy, s=9, facecolor='white', edgecolor=COLORS['TW'], linewidth=.65, zorder=2)
        ax.hlines(yy, lo, hi, lw=.8, color=COLORS['TW'], zorder=3)
        ax.set_title(TITLES[mi].replace('Correct-state ', 'Correct-state\n').replace('Recovery from ', 'Recovery from\n').replace('Observed non-correct run', 'Observed\nnon-correct run'),
                     loc='left', fontsize=6.7, weight='bold', pad=16)
        ax.text(0, 1.012, 'Lower is better' if mi == 3 else 'Higher is better', transform=ax.transAxes, fontsize=5.7)
        mark_panel(ax, string.ascii_lowercase[mi])
        full_min, full_max = min(0,lo.min()), max(0,hi.max())
        span = max(full_max-full_min, 1e-6)
        ax.set_xlim(full_min-span*.10, full_max+span*.10)
        xmin, xmax = ax.get_xlim()
        ticks = MaxNLocator(nbins=4, min_n_ticks=3).tick_values(xmin, xmax)
        ax.set_xticks(ticks[(ticks >= xmin) & (ticks <= xmax)])
        ax.set_xlabel('Difference (cycles)' if mi == 3 else 'Difference (percentage points)', fontsize=5.7, labelpad=7)
        ax.set_yticks(yy)
        ax.set_yticklabels(conditions.ConditionCode)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(19.35, -.75)
    if detail:
        for i, (metric, x0) in enumerate(zip(METRICS[:2], [.145, .61])):
            ax = fig.add_axes([x0, .125, .365, .13])
            sub = paired[(paired.ContrastCode == contrast) & (paired.Metric == metric) & (paired.Scene == 'IDF')].sort_values('ConditionOrdinal')
            y = sub.DeltaPoint.to_numpy(float)*100
            lo = sub.CI95Lower.to_numpy(float)*100
            hi = sub.CI95Upper.to_numpy(float)*100
            ax.axvline(0, color='#6E7177', lw=.6, ls=(0,(3,3)))
            ax.scatter(y, np.arange(4), s=12, facecolor='white', edgecolor=COLORS['TW'], linewidth=.7, zorder=2)
            ax.hlines(np.arange(4), lo, hi, color=COLORS['TW'], lw=.9, zorder=3)
            ax.set_yticks(range(4), sub.ConditionCode)
            ax.set_ylim(3.55, -.55)
            ax.set_title('F-only detail: '+TITLES[i].lower(), loc='left', fontsize=6.5, weight='bold', pad=9)
            ax.text(-.105,1.11,string.ascii_lowercase[4+i],transform=ax.transAxes,fontsize=8,weight='bold')
            ax.set_xlabel('TW − T (percentage points)', fontsize=6)
            ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        fig.text(.145, .056, 'Positive values mean TW is larger. Smaller observed non-correct run length is favourable.', fontsize=5.7)
        fig.text(.145, .034, 'e, f enlarge the F-only rows in a, b without changing values or intervals. All intervals are pointwise.', fontsize=5.7)
    else:
        fig.text(.145, .064, 'Positive values mean TW is larger. Smaller observed non-correct run length is favourable.', fontsize=6)
        fig.text(.145, .038, 'Rows: HPRF-only, F-only (shaded), then composite. Intervals are pointwise; no multiplicity correction is applied.', fontsize=5.7)
    save(fig, stem)


def strata(width):
    stem = 'fig06_width_strata'
    fig, axs = plt.subplots(1, 2, figsize=(183/25.4, 108/25.4), gridspec_kw={'width_ratios':[1.5,1]})
    fig.subplots_adjust(left=.105, right=.965, bottom=.235, top=.80, wspace=.35)
    top(fig, CAPTIONS[stem], 'Descriptive pooled counts; n = 2,000 repeats; no stratum-specific confidence intervals')
    order = ['WIDTH_STRATUM_0NS','WIDTH_STRATUM_10NS','WIDTH_STRATUM_GE20NS']
    labels = ['0', '10', '≥20']
    for mi, method in enumerate(METHODS):
        sub = width[width.MethodCode == method].set_index('WidthStratum').loc[order]
        x = np.arange(3)+(mi-2)*.035
        axs[0].plot(x, sub.P_cor_stratum*100, marker=MARKERS[method], ms=3.7, color=COLORS[method],
                    ls=STYLES[method], lw=1.1, markerfacecolor='white' if mi < 2 else COLORS[method])
    axs[0].set_ylim(-3, 106)
    axs[0].set_yticks([0,25,50,75,100])
    axs[0].set_ylabel('P(cor | width stratum) (%)')
    axs[0].set_title('Correctness by true-width separation', loc='left', weight='bold', pad=11)
    support = width[width.MethodCode == 'TW'].set_index('WidthStratum').loc[order]
    for mi in METHODS:
        assert (width[width.MethodCode == mi].set_index('WidthStratum').loc[order].TotalCycleCount.values == support.TotalCycleCount.values).all()
    counts = support.TotalCycleCount.to_numpy(int)
    axs[1].bar(np.arange(3), counts/1000, width=.55, color='#A8B4C6')
    for x, n in enumerate(counts):
        axs[1].text(x, n/1000+10, f'{n:,}', ha='center', fontsize=6)
    axs[1].set_ylim(0, 380)
    axs[1].set_yticks([0,100,200,300])
    axs[1].set_ylabel('Pooled cycle count (×1,000)')
    axs[1].set_title('Stratum support', loc='left', weight='bold', pad=11)
    for idx, ax in enumerate(axs):
        mark_panel(ax, string.ascii_lowercase[idx])
        ax.set_xticks(range(3), labels)
        ax.set_xlabel('Absolute true-width difference (ns)')
    fig.text(.105, .100, 'Difference is relative to the nominal reference width for the current cycle. Counts are shared across methods.', fontsize=5.7)
    fig.text(.105, .071, 'Cycle counts describe stratum support; they are not independent replicate counts. Lines are visual guides only.', fontsize=5.7)
    save(fig, stem)


def precision_plot(precision):
    stem='figS01_precision'
    conditions=precision[['ConditionOrdinal','ConditionCode']].drop_duplicates().sort_values('ConditionOrdinal')
    columns=[(m,metric) for m in ['T','TW','TW-T'] for metric in METRICS]
    mat=np.empty((18,12))
    failed=np.zeros((18,12),dtype=bool)
    for r,cond in enumerate(conditions.ConditionOrdinal):
        for c,(method,metric) in enumerate(columns):
            row=precision[(precision.ConditionOrdinal==cond)&(precision.MethodOrContrast==method)&(precision.Metric==metric)].iloc[0]
            mat[r,c]=row.ActualMaxOneSidedHalfWidth/row.TargetHalfWidth
            failed[r,c]=not row.MeetsStage8PrecisionTarget
    assert failed.sum()==6
    fig,ax=plt.subplots(figsize=(183/25.4,154/25.4))
    fig.subplots_adjust(left=.16,right=.965,bottom=.19,top=.82)
    top(fig,CAPTIONS[stem],'Maximum one-sided CI half-width / prespecified target; ratios greater than 1 miss the target',methods=False)
    cmap=LinearSegmentedColormap.from_list('precision_ratio',['#F8FAFC','#AAC3D9','#D19C79'])
    im=ax.imshow(mat,cmap=cmap,norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=4),aspect='auto',interpolation='nearest')
    ax.set_yticks(range(18),conditions.ConditionCode)
    ax.set_xticks(range(12), SHORT*3,rotation=45,ha='right',rotation_mode='anchor')
    ax.tick_params(length=0,pad=4)
    for (r,c),v in np.ndenumerate(mat):
        # Values close to the target retain three decimals to avoid rounding a miss to 1.00.
        label=f'{v:.3f}' if abs(v-1)<.02 else f'{v:.2f}'
        ax.text(c,r,label,ha='center',va='center',fontsize=5.2,color='#202124')
        if failed[r,c]:
            ax.add_patch(Rectangle((c-.46,r-.46),.92,.92,fill=False,edgecolor='#773E28',lw=1.05,zorder=6))
    for x in [3.5,7.5]:
        ax.axvline(x,color='white',lw=3)
    for y in [4.5,8.5]:
        ax.axhline(y,color='white',lw=2.3)
    for c,name in zip([1.5,5.5,9.5],['T','TW','TW − T (paired)']):
        ax.text(c,-1.1,name,ha='center',va='bottom',fontsize=7,weight='bold')
    ax.spines[['left','bottom']].set_visible(False)
    cax=fig.add_axes([.34,.080,.40,.019])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,1,2,3,4])
    cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=5.5,length=2)
    fig.text(.16,.028,'Outlined cells: 6 / 216 targets missed. Probability target: 0.02; observed run-length target: 0.25 cycles.',fontsize=5.7)
    save(fig,stem)


def export_qa():
    combined=fitz.open()
    records=[]
    preview_items=[]
    for stem in FILES:
        svg=ET.parse(OUT/f'{stem}.svg')
        text_nodes=svg.findall('.//{http://www.w3.org/2000/svg}text')
        assert text_nodes, stem
        pdf=fitz.open(OUT/f'{stem}.pdf')
        page=pdf[0]
        assert page.get_text().strip(), stem
        fonts=page.get_fonts()
        assert not any('Type3' in str(font) for font in fonts)
        assert abs(page.rect.width/72*25.4-183)<.05
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
        pix.save(QA/f'{stem}_pdf.png')
        # Grayscale preview is created in Python from the actual PDF render.
        with Image.open(QA/f'{stem}_pdf.png') as src:
            ImageOps.grayscale(src).save(QA/f'{stem}_grayscale.png')
            thumb=src.convert('RGB')
            thumb.thumbnail((740,860))
            preview_items.append((stem,thumb.copy()))
        combined.insert_pdf(pdf)
        with Image.open(OUT/f'{stem}.tiff') as tif:
            dpi=tif.info.get('dpi')
            assert dpi and abs(dpi[0]-600)<1
            tiff_size=list(tif.size)
        records.append({'figure':stem,'svg_editable_text_nodes':len(text_nodes),'pdf_text_extractable':True,
                        'pdf_font_types':sorted(set(f[2] for f in fonts)), 'width_mm':round(page.rect.width/72*25.4,3),
                        'tiff_pixels':tiff_size,'tiff_dpi':[float(v) for v in dpi]})
        pdf.close()
    combined.save(RESULT_ROOT/'all_figures.pdf')
    combined.close()
    cols=2;cellw=790;cellh=925
    sheet=Image.new('RGB',(cols*cellw,((len(preview_items)+cols-1)//cols)*cellh),'#E8EAED')
    draw=ImageDraw.Draw(sheet)
    for n,(stem,im) in enumerate(preview_items):
        x=(n%cols)*cellw;y=(n//cols)*cellh
        draw.text((x+24,y+14),stem,fill='#202124')
        sheet.paste(im,(x+(cellw-im.width)//2,y+45))
    sheet.save(RESULT_ROOT/'figure_overview.png')
    report={'backend':'Python','numeric_data_unchanged':True,'individual_rows':360,'paired_rows':144,
            'precision_rows':216,'precision_targets_missed':6,'figures':records,'layout':ARTIST_QA,
            'caveat':'Text editability, figure dimensions, TIFF resolution and canvas bounds checked. Visual QA is also required.'}
    (QA/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    failures=[r for r in ARTIST_QA if r['text_outside_canvas']]
    if failures:
        print('LAYOUT WARNINGS:',json.dumps(failures))
    print(f'Exported {len(FILES)} figure sets, combined PDF and Python-rendered QA previews.')


def main():
    tables=load_data()
    individual=tables['individual.csv']
    simple_scene(individual,'HPRF','fig01_hprf','HPRF_kHz','HPRF (kHz)')
    simple_scene(individual,'IDF','fig02_f_only','FDelay_us','F delay (μs)')
    composite(individual)
    forests(tables['paired.csv'],'TW-T','fig04_tw_minus_t')
    forests(tables['paired.csv'],'TW-W','fig05_tw_minus_w')
    strata(tables['width_strata.csv'])
    precision_plot(tables['precision.csv'])
    export_qa()


if __name__=='__main__':
    main()
