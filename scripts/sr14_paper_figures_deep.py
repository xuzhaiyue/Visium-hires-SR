"""Main figures Fig 1–5 as submitted: sr13 layout plus image evidence and mechanism panels.
Reads existing results only (patches, CellViT contours, spot features, annotation items); newly computed quantities (radial power spectra) are saved as csv.
Writes results/sr_paper_figures_2026-11-12_deep/. Usage: python scripts/sr14_paper_figures_deep.py [1 2 3 4 5]"""
import os
try:
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
except PermissionError:
    pass
import sys, json, gzip, io, base64
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr13_paper_figures_pub as P
from sr13_paper_figures_pub import C, LAB, MM, W, tag, micro, paired, F
from sr10_paper_figures import R, R9, S65, PA, HB, CT, img, hires, lapvar
import sr11_supplementary as S

OUT = R / 'sr_paper_figures_2026-11-12_deep'
OUT.mkdir(exist_ok=True)
P.OUT = OUT
save = P.save
cv2.setNumThreads(2)
DONOR = {'GSM6505133': 'Donor A', 'GSM6505134': 'Donor B', 'GSM6505135': 'Donor C'}
TCOL = {1: '#e8112d', 2: '#22b14c', 3: '#1e90ff', 4: '#000000', 5: '#ffb000'}
OLD = PA / 'old_SR08_correct_scale/samples'
HO = PA / 'hires/samples'
_cont = {}


def contours(mosaic, pids):
    key = str(mosaic)
    miss = [p for p in pids if (key, p) not in _cont]
    if miss:
        got = S.mosaic_contours(mosaic, miss)
        for p in miss: _cont[(key, p)] = got[p]
    return {p: _cont[(key, p)] for p in pids}


def overlay(ax, cs, col=None, lw=0.6, offset=0):
    for ct, t in cs:
        ct = ct - offset
        ax.plot(*np.vstack([ct, ct[:1]]).T, color=col or TCOL.get(t, 'yellow'), lw=lw)


def spot_circle(ax, c=255.5, r=110, **kw):
    ax.add_patch(Circle((c, c), r, fill=False, color='white', lw=0.8, ls=(0, (3, 2)), **kw))


def positions(path):
    t = pd.read_csv(path, header=None)
    if not str(t.iloc[0, 0]).endswith('-1'): t = pd.read_csv(path)
    t.columns = ['barcode', 'in_tissue', 'row', 'col', 'py', 'px']
    return t


# ---------------------------------------------------------------- Fig 1
NUC = '#ffe14d'  # Fig 1 nuclear contours in one colour (type bias is shown in Supplementary Fig. S6)
LEGACY = {'native\uff08\u4e0a\u9650\uff09': 'native (upper bound)', '\u51b0\u51bb': 'frozen', '\u672a\u6807\u6ce8': 'not stated'}  # Chinese labels in legacy result files
TCGA_TILE = R / 'sr08/data/hr/train/TCGA-3A-A9J0-01Z-00-DX1_038.png'


def count_box(ax, n):
    ax.text(14, 14, f'{n} nuclei in spot', color='white', fontsize=6, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55, lw=0))


def fig1():
    fig = plt.figure(figsize=(W, 150 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.08, 0.82], hspace=0.36, left=0.06, right=0.99, top=0.96, bottom=0.02)
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    # a GEO by year
    ax = fig.add_subplot(r1[0, 0]); tag(ax, 'a')
    sv = pd.read_csv(R / 'sr_survey/geo_visium_image_survey.csv')
    sv['year'] = pd.to_datetime(sv.pdat).dt.year
    grp = {'hires_only': ('Hires PNG only', C['sr']), 'fullres': ('Full-resolution', '#333333'),
           'bundle_only': ('In archive only', '#9a9a9a'), 'fullres_unc': ('Full-res (unconfirmed)', '#c8c8c8'), 'no_image': ('No image', '#e3e3e3')}
    yrs = np.arange(sv.year.min(), sv.year.max() + 1)
    cum = {k: [((sv.category == k) & (sv.year <= y)).sum() for y in yrs] for k in grp}
    ax.stackplot(yrs, [cum[k] for k in grp], colors=[v[1] for v in grp.values()], lw=0)
    tot = np.sum([cum[k] for k in grp], 0)
    ax.set_xlim(yrs[0], yrs[-1]); ax.set_ylim(0, tot[-1] * 1.05)
    ax.set_xlabel('GEO release year'); ax.set_ylabel('Human Visium series (cumulative)')
    ax.text(yrs[-1] - 0.15, cum['hires_only'][-1] * 0.5, f"Hires only\n{cum['hires_only'][-1]} ({100 * cum['hires_only'][-1] / tot[-1]:.0f}%)", ha='right', va='center', fontsize=6.3, color='white', fontweight='bold')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v[1]) for v in grp.values()], labels=[v[0] for v in grp.values()],
              loc='upper left', fontsize=5.8, handlelength=0.8, labelspacing=0.2)
    ax.set_title(f'The hires-only record ({tot[-1]} series)', loc='left')
    # b scale: spots drawn at true size on real hires; the consequence of the wrong scale is stated in the legend
    g = 'GSM6505133'
    sa = pd.read_csv(PA / 'scale_audit.csv').set_index('gsm')
    src = json.loads((PA / 'C/A_paired' / g / 'image_sources.json').read_text())
    H = hires(src['hires']); sf = src['sf']
    pos = positions(sa.at[g, 'positions_path']); pos = pos[pos.in_tissue == 1]
    um_px_full = 100 / sa.at[g, 'neighbor_pitch_fullres_px']
    um_h = um_px_full / sf
    pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('patch_id')
    pid = 'p00093'
    cx, cy = pt.loc[pid, 'fullres_x'] * sf, pt.loc[pid, 'fullres_y'] * sf
    sub = r1[0, 1].subgridspec(1, 2, wspace=0.06, width_ratios=[1, 1])
    a0 = fig.add_subplot(sub[0, 0]); tag(a0, 'b')
    a0.imshow(H)
    m_ = 60 / um_h  # crop to the extent of in-tissue spots, avoiding the fiducial frame
    x0b, x1b, y0b, y1b = pos.px.min() * sf - m_, pos.px.max() * sf + m_, pos.py.min() * sf - m_, pos.py.max() * sf + m_
    a0.set_xlim(x0b, x1b); a0.set_ylim(y1b, y0b); a0.axis('off')
    half = 250 / um_h
    a0.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec=C['sr'], lw=1.0))
    L = 1000 / um_h; xb = x1b - 8; yb = y1b - 8
    a0.plot([xb - L, xb], [yb, yb], color='black', lw=1.5, solid_capstyle='butt'); a0.text(xb - L / 2, yb - 8, '1 mm', ha='center', va='bottom', fontsize=6)
    a0.set_title(f'Hires image, {um_h:.1f} µm/px', loc='left', fontsize=6.5, fontweight='normal')
    a1 = fig.add_subplot(sub[0, 1])
    crop = H[int(cy - half):int(cy + half), int(cx - half):int(cx + half)]
    a1.imshow(crop, interpolation='nearest', extent=(0, 2 * half * um_h, 2 * half * um_h, 0))
    near = pos[(abs(pos.px * sf - cx) < half) & (abs(pos.py * sf - cy) < half)]
    for _, r in near.iterrows():
        x, y = (r.px * sf - (cx - half)) * um_h, (r.py * sf - (cy - half)) * um_h
        a1.add_patch(Circle((x, y), 27.5, fill=False, color='white', lw=1.0))
        a1.add_patch(Circle((x, y), 32.5, fill=False, color=C['sr'], lw=1.0, ls=(0, (2.2, 1.4))))
    r0 = near.iloc[(((near.px * sf - cx) ** 2 + (near.py * sf - cy) ** 2)).argmin()]
    x0, y0 = (r0.px * sf - (cx - half)) * um_h, (r0.py * sf - (cy - half)) * um_h
    nb = near[(near.row == r0.row) & (abs(near.col - r0.col) == 2)]
    if len(nb):
        xn = (nb.iloc[0].px * sf - (cx - half)) * um_h
        a1.annotate('', (xn, y0), (x0, y0), arrowprops=dict(arrowstyle='<->', color='white', lw=1.0, shrinkA=0, shrinkB=0))
        a1.text((x0 + xn) / 2, y0 - 8, '100 µm', color='white', ha='center', va='bottom', fontsize=6, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6, lw=0))
    a1.set_xlim(0, 2 * half * um_h); a1.set_ylim(2 * half * um_h, 0); a1.set_xticks([]); a1.set_yticks([])
    for s_ in a1.spines.values(): s_.set_color(C['sr']); s_.set_linewidth(1.0)
    a1.set_title('Spots at true scale (500 µm field)', loc='left', fontsize=6.5, fontweight='normal')
    a1.legend(handles=[Line2D([], [], color='#555555', lw=1.0, label='55 µm capture area'),
                       Line2D([], [], color=C['sr'], lw=1.0, ls=(0, (2.2, 1.4)), label='65 µm = spot_diameter_fullres')],
              loc='upper left', bbox_to_anchor=(-0.02, -0.01), fontsize=5.8, handlelength=1.6, labelspacing=0.25)
    a0.text(0, -0.03, 'Reading 65 µm as 55 µm shrinks every\nlength by 15% and every area by 28%', transform=a0.transAxes, fontsize=5.8, va='top', color='#555555')
    # c four images of the same spot + nuclear contours (one colour)
    base = S65 / 'paired/samples' / g
    cs = {m: contours(base / 'mosaic' / m, [pid])[pid] for m in ['native', 'sr', 'bicubic']}
    r2 = gs[1].subgridspec(1, 4, wspace=0.04)
    tiles = [('lr', 'Hires pixels (4 µm)', None), ('bicubic', 'Bicubic ×16', 'bicubic'),
             ('sr', 'SR (calibrated) ×16', 'sr'), ('native', 'Native scan (0.25 µm)', 'native')]
    for k, (m, t, ck) in enumerate(tiles):
        a = fig.add_subplot(r2[0, k]); im_ = img(base / 'patches' / m / f'{pid}.png')
        if im_.shape[0] < 512: im_ = np.repeat(np.repeat(im_, 512 // im_.shape[0], 0), 512 // im_.shape[1], 1)  # enlarge 4 µm pixels as is, same coordinates as the other cells
        micro(a, im_, t, nearest=True, um_px=128 / im_.shape[1], label=k == 0)
        if ck:
            inside = [(c_, t_) for c_, t_ in cs[ck] if ((c_.mean(0) - 255.5) ** 2).sum() < 110 ** 2]
            overlay(a, cs[ck], col=NUC, lw=0.6); count_box(a, len(inside))
        spot_circle(a); a.set_xlim(0, 512); a.set_ylim(512, 0)
        if k == 0: tag(a, 'c')
    fig.legend(handles=[Line2D([], [], color=NUC, lw=1.2, label='CellViT++ nucleus contour'),
                        Line2D([], [], color='#777777', lw=0.8, ls=(0, (3, 2)), label='55 µm spot (same spot in every panel; Donor A)')],
               loc='upper center', bbox_to_anchor=(0.525, gs[1].get_position(fig).y0 - 0.005), ncol=2, fontsize=6, handlelength=1.8, columnspacing=2.5)
    # d pipeline: training (TCGA synthetic pairs) → application (real hires)
    ax = fig.add_subplot(gs[2]); tag(ax, 'd'); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    from sr09_scale65_degrade import degrade_real2
    hr = np.array(Image.open(TCGA_TILE).convert('RGB'))
    lr, _ = degrade_real2(hr, 32, np.random.default_rng(7))
    sr_ = img(base / 'patches/sr' / f'{pid}.png'); lr_r = img(base / 'patches/lr' / f'{pid}.png')
    th_w, th_y, th_h = 11.2, 8.5, 17.0
    def thumb(x, im, near_=False, cont=None, txt=''):
        ia = ax.inset_axes([x, th_y, th_w, th_h], transform=ax.transData)
        ia.imshow(im, interpolation='nearest' if near_ else 'antialiased', extent=(0, 512, 512, 0)); ia.set_xticks([]); ia.set_yticks([])
        for s_ in ia.spines.values(): s_.set_linewidth(0.4); s_.set_color('#444444')
        if cont is not None: overlay(ia, cont, col=NUC, lw=0.35); spot_circle(ia)
        ia.set_xlim(0, 512); ia.set_ylim(512, 0)
        ax.text(x + th_w / 2, th_y - 1.2, txt, ha='center', va='top', fontsize=5.8, linespacing=1.15)
        return ia
    xs_ = [0.5, 15.5, 41.5, 56.5, 71.5, 86.5]
    thumb(xs_[0], hr, txt='TCGA-PAAD FFPE\n0.5 µm (69 slides)')
    thumb(xs_[1], lr, near_=True, txt='Calibrated degradation\nblur σ 1.5–1.75 µm → 4 µm')
    thumb(xs_[2], lr_r, near_=True, txt='Real Visium hires\n4 µm/px')
    thumb(xs_[3], sr_, txt='SwinIR ×8 + bicubic ×2\n0.25 µm/px')
    thumb(xs_[4], sr_, cont=cs['sr'], txt='CellViT++ (SAM-H)\nnuclei')
    sxy = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('barcode')
    zs = pd.read_csv(PA / 'C/A_paired' / g / 'spot_expression_sr.csv').set_index('barcode')
    im_ax = ax.inset_axes([xs_[5], th_y, th_w, th_h], transform=ax.transData)
    emt_map(im_ax, sxy.reindex(zs.index)[['fullres_x', 'fullres_y']].to_numpy(), zs['eccentricity_median'].to_numpy(), sf, '')
    for s_ in im_ax.spines.values(): s_.set_visible(False)
    ax.text(xs_[5] + th_w / 2, th_y - 1.2, 'Eccentricity (1 of 7 features)\nper 55 µm spot, vs expression', ha='center', va='top', fontsize=5.8, linespacing=1.15)
    for xa, xb_ in [(xs_[0], xs_[1]), (xs_[2], xs_[3]), (xs_[3], xs_[4]), (xs_[4], xs_[5])]:
        ax.annotate('', (xb_ - 0.4, th_y + th_h / 2), (xa + th_w + 0.4, th_y + th_h / 2), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#555555', mutation_scale=7))
    # training pairs → model; the model is used in the hires → SR step
    ym = th_y + th_h / 2
    ax.annotate('', (39.6, ym), (xs_[1] + th_w + 0.4, ym), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=C['sr'], mutation_scale=8))
    ax.text(33.0, ym + 1.0, 'train\nSwinIR ×8', color=C['sr'], fontsize=5.8, ha='center', va='bottom', linespacing=1.1)
    ax.text((xs_[2] + th_w + xs_[3]) / 2, ym + 1.0, 'apply', color=C['sr'], fontsize=5.8, ha='center', va='bottom')
    for x0_, x1_, t in [(xs_[0], xs_[1] + th_w, 'Training (synthetic pairs)'), (xs_[2], xs_[5] + th_w, 'Application (real hires; shown: Donor A)')]:
        ax.plot([x0_, x1_], [29.2, 29.2], color='#777777', lw=0.6)
        ax.text((x0_ + x1_) / 2, 29.8, t, ha='center', va='bottom', fontsize=6.3, fontweight='bold')
    ax.text(50, -1.5, 'Blur calibrated on GSE327056 (4 native–hires pairs) · tested on GSE211895 (3 held-out donors) and HEST (9 slides, 6 cancer types) · reference: CellViT++ on the native scan',
            ha='center', va='top', fontsize=5.6, color='#444444')
    save(fig, 'Fig1')


# ---------------------------------------------------------------- Fig 2
BENCH = R / 'sr_bench'
TRACE = R / 'sr08/contours_x4_supplement_2026-10-23/final_analysis_2026-10-23/derived_bundle'
EX_PATCH = 'TENX152_0074'  # patch with the most hand-traced nuclei (3 valid contours); detections of the three methods match the overall rates
X8 = 'real2_4.0_x8run01'


def bench_field(cond, pid):
    """Image (0.25 µm/px) and CellViT contours (patch coordinates) of one 512 px patch in the sr_bench mosaic."""
    import tifffile, zarr
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from hemorph import cellvit_io as CV
    d = BENCH / 'mosaic' / f'bicubic2_{cond}'
    r = pd.read_csv(d / 'layout.csv').set_index('patch_id').loc[pid]
    z = zarr.open(tifffile.imread(d / 'he_bicubic2_0.25mpp.tiff', aszarr=True), mode='r')
    z = z['0'] if hasattr(z, 'keys') else z
    im = np.asarray(z[r.oy:r.oy + 512, r.ox:r.ox + 512, :3])
    cells, conts = CV.load_cells(d / 'cellvit_bicubic2_0.25mpp')
    k = np.flatnonzero((cells.cx >= r.ox) & (cells.cx < r.ox + 512) & (cells.cy >= r.oy) & (cells.cy < r.oy + 512))
    return im, [(conts[i] - [r.ox, r.oy], 0) for i in k]


def fig2():
    fig = plt.figure(figsize=(W, 170 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.95, 1.05], hspace=0.5, left=0.065, right=0.99, top=0.95, bottom=0.05)
    top = gs[0].subgridspec(1, 4, width_ratios=[1.25, 1, 1, 0.62], wspace=0.55)
    sr8 = 'SR08 (original simulation)'
    # a F1 by resolution
    ax = fig.add_subplot(top[0, 0]); tag(ax, 'a')
    rows = []
    for k, (s, um) in enumerate([('x2', 1), ('x4', 2), ('x8', 4)]):
        d = pd.read_csv(R / f'sr08/eval_{s}/metrics_by_sample.csv'); d = d[d.testset.str.startswith('HEST') & (d.group == 'FFPE')]
        w = d.pivot_table(index='sample', columns='method', values='f1'); w['um'] = um; rows.append(w)
    w = pd.concat(rows)
    nsl = w.index.nunique()
    for m, col in [('bicubic', C['bicubic']), ('SwinIR', C['old'])]:
        for sname, z in w.groupby(level=0): ax.plot(z.um, z[m], color=col, lw=0.35, alpha=0.5)
        mu = w.groupby('um')[m].mean(); ax.plot(mu.index, mu.values, '-o', color=col, lw=1.6, ms=4, zorder=4, label='Bicubic' if m == 'bicubic' else 'SR08')
    ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4], ['1', '2', '4']); ax.set_xlim(0.85, 4.9); ax.set_ylim(0, 1)
    ax.set_xlabel('Simulated input pixel size (µm)'); ax.set_ylabel('Nucleus detection F1')
    ax.text(4.3, w[w.um == 4].SwinIR.mean(), f"{w[w.um == 4].SwinIR.mean():.2f}", color=C['old'], fontsize=6.3, va='center')
    ax.text(4.3, w[w.um == 4].bicubic.mean(), f"{w[w.um == 4].bicubic.mean():.2f}", color='#777777', fontsize=6.3, va='center')
    ax.legend(loc='lower left', fontsize=6.3); ax.set_title(f'Detection vs resolution\n({nsl} HEST FFPE slides, all 3 scales)', loc='left')
    d = pd.read_csv(R / 'sr08/eval_x8/metrics_by_sample.csv').replace({'group': LEGACY})
    for j, (col, yl, t, tg, lim) in enumerate([('rho_density', 'Tile density ρ vs reference', 'Nuclear density\nat 4 µm', 'b', (0, 1)),
                                               ('abs_log_area_med', '|log area ratio|, matched nuclei', 'Nuclear size error\nat 4 µm', 'c', (0, 0.7))]):
        ax = fig.add_subplot(top[0, 1 + j]); tag(ax, tg)
        for k, ts in enumerate(['HEST', 'Xenium']):
            z = d[d.testset.str.startswith(ts)].pivot_table(index='sample', columns='method', values=col)
            paired(ax, 2.5 * k, 2.5 * k + 1, z.bicubic, z.SwinIR, C['bicubic'], C['old'], s=9)
            better = (z.SwinIR > z.bicubic).sum() if j == 0 else (z.SwinIR < z.bicubic).sum()
            ax.text(2.5 * k + 0.5, lim[1] * 1.04, f'{better}/{len(z)} better', ha='center', va='bottom', fontsize=6)
        ax.set_xticks([0.5, 3], ['HEST\nn = 28', 'Xenium\nn = 12']); ax.set_xlim(-0.6, 4.1); ax.set_ylim(lim[0], lim[1] * 1.12); ax.set_yticks(np.linspace(*lim, 6 if j == 0 else 8))
        ax.spines['left'].set_bounds(*lim); ax.set_ylabel(yl); ax.set_title(t, loc='left')
    # d blinded label counts
    ax = fig.add_subplot(top[0, 3]); tag(ax, 'd')
    lab = pd.read_csv(R / 'sr08/annotation_x8/analysis_review_2026-10-25/labels_adjudicated.csv')
    lab = lab[(lab.cat == 'unmatched') & ~lab.is_repeat.astype(bool)]
    cnt = lab.groupby(['method', 'final_label']).size().unstack(fill_value=0).reindex(['bicubic', 'SwinIR'])
    bottom = np.zeros(2)
    for k, col, nm in [('yes', '#5a5a5a', 'Nucleus'), ('uncertain', '#cfcfcf', 'Uncertain'), ('no', '#c62828', 'Not a nucleus')]:
        v = cnt.get(k, pd.Series(0, index=cnt.index)).to_numpy(); ax.bar([0, 1], v, bottom=bottom, color=col, width=0.66, label=nm); bottom += v
    for i in range(2): ax.text(i, bottom[i] + 4, f"{int(cnt.get('no', pd.Series([0, 0])).iloc[i])}\nfalse", ha='center', va='bottom', fontsize=6.3, fontweight='bold', color='#c62828', linespacing=1.0)
    ax.set_xticks([0, 1], ['Bicubic', 'SR08']); ax.set_ylabel('Unmatched detections, blinded'); ax.set_ylim(0, 245); ax.set_xlim(-0.55, 1.55)
    for i in range(2): ax.text(i, 95, 'Real nucleus', color='white', ha='center', va='center', fontsize=6, rotation=90)
    ax.set_title('Invented\nnuclei?', loc='left')
    # e one field: simulated 4 µm input → bicubic → SR08 → native; white hand traces, yellow CellViT
    key = pd.read_csv(TRACE / 'key.csv'); key = key[key.patch_id == EX_PATCH]
    tr = []
    for r in key.itertuples():
        f = TRACE / 'raw/contours' / f'{r.id}.json'
        j_ = json.loads(f.read_text())
        if j_.get('status') == 'drawn': tr.append(np.asarray(j_['pts'], float) + [r.crop_x0, r.crop_y0])
    hn = pd.read_csv(R / 'sr08/eval_x8/contours/per_nucleus.csv').replace({'method': LEGACY})
    det = hn[hn.id.isin(key.id)].groupby('method').detected.sum()
    smp = EX_PATCH.split('_')[0]
    lr = img(BENCH / 'patches' / smp / X8 / f'{EX_PATCH}.png'); lr = np.repeat(np.repeat(lr, 16, 0), 16, 1)
    r2 = gs[1].subgridspec(1, 4, wspace=0.04)
    panels = [(lr, 'Simulated 4 µm input', None, None), (X8, 'Bicubic ×16', 'bicubic ×8', None),
              (f'swinir8_{X8}', 'SR08 ×16', 'SwinIR ×8', None), ('native', 'Native scan (0.25 µm)', 'native (upper bound)', None)]
    for k, (cond, t, hk, _) in enumerate(panels):
        a = fig.add_subplot(r2[0, k])
        if isinstance(cond, np.ndarray): im_, cs_ = cond, None
        else: im_, cs_ = bench_field(cond, EX_PATCH)
        micro(a, im_, t, nearest=k == 0, um_px=0.25, label=k == 0)
        if cs_ is not None: overlay(a, cs_, col=NUC, lw=0.6)
        for h_ in tr: a.plot(*np.vstack([h_, h_[:1]]).T, color='white', lw=0.8, path_effects=[P.pe.withStroke(linewidth=1.5, foreground='black')])
        if hk: a.text(14, 14, f'{int(det.get(hk, 0))}/{len(tr)} traced nuclei found\n{len(cs_)} CellViT nuclei in field', color='white', fontsize=6, ha='left', va='top', linespacing=1.15,
                      bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55, lw=0))
        a.set_xlim(0, 512); a.set_ylim(512, 0)
        if k == 0: tag(a, 'e')
    fig.legend(handles=[Line2D([], [], color='white', lw=0.9, path_effects=[P.pe.withStroke(linewidth=1.8, foreground='black')], label='Hand-traced nucleus'),
                        Line2D([], [], color=NUC, lw=1.2, label='CellViT++ nucleus contour')],
               loc='upper center', bbox_to_anchor=(0.525, gs[1].get_position(fig).y0 - 0.004), ncol=2, fontsize=6, handlelength=1.8, columnspacing=2.5)
    # f–h hand-traced nuclei: detection by size; area of detected nuclei; per-slide F1 gain
    mm = {'native (upper bound)': ('native', 'Native (upper bound)'), 'bicubic ×8': ('bicubic', 'Bicubic'), 'SwinIR ×8': ('old', 'SR08')}
    bot = gs[2].subgridspec(1, 3, width_ratios=[1.1, 1, 0.85], wspace=0.5)
    ax = fig.add_subplot(bot[0, 0]); tag(ax, 'f')
    hn['bin'] = pd.qcut(hn.human_area_um2, 3, labels=False)
    edges = hn.human_area_um2.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    for k, (m, (ck, nm)) in enumerate(mm.items()):
        z = hn[hn.method == m].groupby('bin').detected.agg(['mean', 'size'])
        se = np.sqrt(z['mean'] * (1 - z['mean']) / z['size'])
        ax.errorbar(np.arange(3) + (k - 1) * 0.12, z['mean'], yerr=se, color=C[ck], marker='o', ms=4, lw=1.2, capsize=1.5)
        ax.text(2.25, z['mean'].iloc[-1], nm.replace(' (', '\n('), color=C[ck] if ck != 'bicubic' else '#777777', fontsize=6, va='center', linespacing=1.0)
    ax.set_xticks(range(3), [f'{edges[i]:.0f}–{edges[i + 1]:.0f}' for i in range(3)]); ax.set_xlabel('Hand-traced nuclear area (µm²)'); ax.set_xlim(-0.3, 3.0)
    ax.spines['bottom'].set_bounds(-0.3, 2.2)
    ax.set_ylabel('Fraction detected'); ax.set_ylim(0, 1.05)
    ax.set_title(f'Hand-traced nuclei (n = {hn.id.nunique()}, {hn["sample"].nunique()} slides)', loc='left')
    ax = fig.add_subplot(bot[0, 1]); tag(ax, 'g')
    hdl = []
    for m, (ck, nm) in mm.items():
        z = hn[(hn.method == m) & hn.detected]
        bias = np.median(np.log2(z.pred_area_um2 / z.human_area_um2))
        ax.scatter(z.human_area_um2, z.pred_area_um2, s=9, color=C[ck], edgecolor='white', linewidth=0.3, alpha=0.9)
        hdl.append(Line2D([], [], marker='o', ls='', ms=3.5, color=C[ck], label=f"{nm.split(' ')[0]}, bias {bias:+.2f}".replace('-', '−')))
    ax.plot([5, 150], [5, 150], color='#555555', lw=0.6, ls='--'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(5, 150); ax.set_ylim(5, 600)
    ax.set_xticks([10, 30, 100], ['10', '30', '100']); ax.set_yticks([10, 30, 100], ['10', '30', '100'])
    ax.set_xlabel('Hand-traced area (µm²)'); ax.set_ylabel('Segmented area (µm²)'); ax.set_title('Size of detected nuclei', loc='left')
    ax.spines['left'].set_bounds(5, 150); ax.set_yticks([6, 7, 8, 9, 20, 40, 50, 60, 70, 80, 90], minor=True); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(handles=hdl, loc='upper left', fontsize=5.8, title='Median log$_2$(segmented / traced)', title_fontsize=5.8, alignment='left', borderpad=0.1)
    ax = fig.add_subplot(bot[0, 2]); tag(ax, 'h')
    d8 = d[d.testset.str.startswith('HEST')].pivot_table(index=['sample', 'stratum', 'group'], columns='method', values='f1').reset_index()
    d8['gain'] = d8.SwinIR - d8.bicubic
    d8.to_csv(OUT / 'fig2h_f1_gain_by_slide.csv', index=False)
    grp = [('FFPE', 'FFPE', '#333333'), ('frozen', 'Frozen', '#7fb2e5'), ('not stated', 'Not\nstated', '#bbbbbb')]
    for i, (g_, nm, col) in enumerate(grp):
        v = d8[d8.group == g_].gain.to_numpy()
        ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=12, color=col, edgecolor='white', linewidth=0.3, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color='black', lw=0.9, zorder=4)
        ax.text(i, 0.565, f'n = {len(v)}', ha='center', fontsize=6)
    ax.axhline(0, color='#bbbbbb', lw=0.5); ax.set_xticks(range(3), [g[1] for g in grp]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.02, 0.6)
    ax.set_ylabel('F1 gain of SR08 over bicubic, 4 µm'); ax.set_title(f'Every HEST slide gains\n({(d8.gain > 0).sum()}/{len(d8)})', loc='left')
    save(fig, 'Fig2')


# ---------------------------------------------------------------- Fig 3
def radial_power(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g = g - g.mean(); n = g.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n)); f = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    yy, xx = np.indices(f.shape); r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    return np.bincount(r.ravel(), f.ravel())[: n // 2] / np.bincount(r.ravel())[: n // 2]


def psd_table():
    path = OUT / 'fig3_radial_power.csv'
    if path.exists(): return pd.read_csv(path)
    import importlib
    v2 = importlib.import_module('sr02_degrade_benchmark').degrade_real2
    cal = importlib.import_module('sr09_scale65_degrade').degrade_real2
    rows = []
    for g in DONOR:
        pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv')
        for i, pid in enumerate(pt.patch_id.sample(120, random_state=1)):
            nat = img(S65 / 'paired/samples' / g / 'patches/native' / f'{pid}.png'); real = img(S65 / 'paired/samples' / g / 'patches/lr' / f'{pid}.png')
            if (nat.mean(2) > 215).mean() > 0.5: continue
            gt = cv2.resize(nat, (256, 256), interpolation=cv2.INTER_AREA)
            ims = {'real': real, 'orig': v2(gt, 32, np.random.default_rng(i))[0], 'cal': cal(gt, 32, np.random.default_rng(i))[0],
                   'ideal': cv2.resize(nat, (32, 32), interpolation=cv2.INTER_AREA)}
            for k, im_ in ims.items():
                for fi, pw in enumerate(radial_power(im_)): rows.append((g, pid, k, fi, pw))
    t = pd.DataFrame(rows, columns=['gsm', 'patch_id', 'source', 'freq_bin', 'power']); t.to_csv(path, index=False)
    return t


def fig3():
    fig = plt.figure(figsize=(W, 200 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2.4, 1], hspace=0.3, left=0.075, right=0.99, top=0.96, bottom=0.05)
    r1 = gs[0].subgridspec(1, 3, width_ratios=[0.8, 1.25, 1], wspace=0.42)
    # a original pipeline fails: same sections, simulated input vs real hires
    ax = fig.add_subplot(r1[0, 0]); tag(ax, 'a')
    L = pd.read_csv(R9 / 'B_sim_vs_real_2026-10-27_legacy_scale55/comparison_side_by_side.csv'); L = L[L.feature == 'density']
    for k, m in enumerate(['bicubic', 'sr']):
        z = L[L.method == m]; cc = C['bicubic'] if m == 'bicubic' else C['old']
        paired(ax, 2.2 * k, 2.2 * k + 1, z['rho__simulated_real2_4.0'], z['rho__true_hires'], cc, cc, s=12)
        dlt = (z['rho__true_hires'] - z['rho__simulated_real2_4.0']).mean()
        ax.text(2.2 * k + 0.5, 0.97, f'Δ = {dlt:+.2f}'.replace('-', '−'), ha='center', va='top', fontsize=6.3, fontweight='bold', color='#555555' if m == 'bicubic' else C['old'])
    ax.set_xticks([0, 1, 2.2, 3.2], ['Sim.', 'Real', 'Sim.', 'Real']); ax.set_ylim(0, 1); ax.set_xlim(-0.5, 3.7)
    for x, t in [(0.5, 'Bicubic'), (2.7, 'SR08')]: ax.text(x, -0.15, t, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.5, fontweight='bold')
    ax.set_ylabel('Spot density ρ vs native'); ax.set_title('SR08 fails on real hires\n(same sections, 3 donors)', loc='left')
    # b radial power spectra; Laplacian sharpness ratios as inset text (distribution in Supplementary Fig. S1d)
    ax = fig.add_subplot(r1[0, 1]); tag(ax, 'b')
    t = psd_table()
    piv = t.pivot_table(index=['gsm', 'patch_id', 'freq_bin'], columns='source', values='power')
    fr = np.arange(1, 16) / 32 / 4.0
    for src_, col, nm in [('ideal', '#333333', 'Ideal area downsampling'), ('orig', C['old'], 'Simulated, original blur'), ('cal', C['sr'], 'Simulated, calibrated blur')]:
        lr_ = np.log2(piv[src_] / piv['real']).groupby(level='freq_bin').agg(['median', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]).iloc[1:16]
        ax.plot(fr, lr_.iloc[:, 0], color=col, lw=1.3, label=nm); ax.fill_between(fr, lr_.iloc[:, 1], lr_.iloc[:, 2], color=col, alpha=0.15, lw=0)
    ax.axhline(0, color='#555555', lw=0.6, ls='--'); ax.text(fr[0], -0.45, 'Real hires', ha='left', va='bottom', fontsize=6, color='#555555')
    lv = pd.read_csv(R / 'sr_paper_figures_2026-11-01/fig3c_laplacian_ratio.csv')
    ax.text(0.03, 0.56, f'Laplacian sharpness vs real\n({len(lv)} fields): original {2 ** lv.v2.median():.2f}×,\ncalibrated {2 ** lv.cal.median():.2f}×',
            transform=ax.transAxes, ha='left', va='top', fontsize=5.8, color='#444444', linespacing=1.2)
    ax.set_xlabel('Spatial frequency (cycles/µm)'); ax.set_ylabel('log$_2$ power / real hires'); ax.legend(loc='upper left', fontsize=5.8)
    ax.set_title('Cause: the simulation was too sharp', loc='left')
    # c calibration curves
    ax = fig.add_subplot(r1[0, 2]); tag(ax, 'c')
    cc = pd.read_csv(S65 / 'calibration/calibration_sample_curves.csv')
    ax.axvspan(0.3, 1.2, color=C['old'], alpha=0.12, lw=0); ax.axvspan(1.5, 1.75, color=C['sr'], alpha=0.3, lw=0)
    for gg, z in cc.groupby('gsm'): ax.plot(z.sigma_um, z.loss, color='#555555', lw=0.7, marker='o', ms=1.8)
    ax.set_xscale('log'); ax.set_xticks([0.3, 1, 3, 10], ['0.3', '1', '3', '10']); ax.set_xlim(0.28, 14); ax.set_ylim(0, 8)
    ax.set_xlabel('Gaussian blur σ (µm)'); ax.set_ylabel('Sharpness/spectrum mismatch\nto real hires (loss)')
    ax.text(0.6, 7.7, 'SR08', fontsize=6, ha='center', va='top', color=C['old']); ax.text(1.62, 7.7, 'Calibrated', fontsize=6, va='top', ha='left', color=C['sr'])
    ax.text(0.33, 4.4, 'one line per\nGSE327056 pair', fontsize=5.8, ha='left', va='center', color='#555555')
    ax.set_title('Fix: calibrate blur on 4 public pairs', loc='left')
    # d reconstructions of real hires: SR08 vs calibrated SR vs native (Donor B)
    g = 'GSM6505134'
    ptn = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv'); pto = pd.read_csv(OLD / g / 'patches.csv')
    fn = pd.read_csv(S65 / 'paired/samples' / g / 'spot_features_native.csv')
    cand = fn[(fn.n >= 15) & (fn.n <= 30)].merge(ptn[['patch_id', 'barcode']], on='patch_id')
    cand = cand.merge(pto[['patch_id', 'barcode']].rename(columns={'patch_id': 'old_pid'}), on='barcode')
    pick = cand.sample(2, random_state=4)
    r2 = gs[1].subgridspec(2, 4, wspace=0.03, hspace=0.06)
    for i, r in enumerate(pick.itertuples()):
        cn = contours(S65 / 'paired/samples' / g / 'mosaic/native', [r.patch_id])[r.patch_id]
        cs = contours(S65 / 'paired/samples' / g / 'mosaic/sr', [r.patch_id])[r.patch_id]
        co = contours(OLD / g / 'mosaic/sr', [r.old_pid])[r.old_pid]
        lr_ = img(S65 / 'paired/samples' / g / 'patches/lr' / f'{r.patch_id}.png'); lr_ = np.repeat(np.repeat(lr_, 512 // lr_.shape[0], 0), 512 // lr_.shape[1], 1)
        panels = [(lr_, 'Real hires (4 µm)', None, True),
                  (img(OLD / g / 'patches/sr' / f'{r.old_pid}.png'), 'SR08 (simulation-trained)', co, False),
                  (img(S65 / 'paired/samples' / g / 'patches/sr' / f'{r.patch_id}.png'), 'SR (calibrated)', cs, False),
                  (img(S65 / 'paired/samples' / g / 'patches/native' / f'{r.patch_id}.png'), 'Native scan (0.25 µm)', cn, False)]
        for k, (im_, t, cc_, nn_) in enumerate(panels):
            a = fig.add_subplot(r2[i, k]); micro(a, im_, t if i == 0 else None, nearest=nn_, um_px=128 / im_.shape[1], label=(i == 0 and k == 0))
            if cc_ is not None:
                overlay(a, cc_, col=NUC, lw=0.6)
                count_box(a, sum(((c_.mean(0) - 255.5) ** 2).sum() < 110 ** 2 for c_, _ in cc_))
            spot_circle(a); a.set_xlim(0, 512); a.set_ylim(512, 0)
            if i == 0 and k == 0: tag(a, 'd')
            if k == 0: a.set_ylabel(f'Donor B, field {i + 1}', fontsize=6.3)
    # e density hexbins: native vs method (three donors pooled); f per-donor ρ
    r3 = gs[2].subgridspec(1, 4, wspace=0.42, width_ratios=[1, 1, 1, 1.25])
    feats = {}
    for g in DONOR:
        nat = pd.read_csv(S65 / 'paired/samples' / g / 'spot_features_native.csv')
        bc = pd.read_csv(S65 / 'paired/samples' / g / 'spot_features_bicubic.csv'); sr = pd.read_csv(S65 / 'paired/samples' / g / 'spot_features_sr.csv')
        ptn = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv')[['patch_id', 'barcode']]
        od = pd.read_csv(OLD / g / 'spot_features_sr.csv').merge(pd.read_csv(OLD / g / 'patches.csv')[['patch_id', 'barcode']], on='patch_id')
        base_ = nat.merge(ptn, on='patch_id')[['barcode', 'n']].rename(columns={'n': 'native'})
        for m, df in [('bicubic', bc.merge(ptn, on='patch_id')), ('sr', sr.merge(ptn, on='patch_id')), ('old', od)]:
            z = base_.merge(df[['barcode', 'n']], on='barcode'); z['gsm'] = g
            feats.setdefault(m, []).append(z)
    for k, m in enumerate(['bicubic', 'old', 'sr']):
        z = pd.concat(feats[m]); ax = fig.add_subplot(r3[0, k])
        if k == 0: tag(ax, 'e')
        ax.hexbin(z.native, z.n, gridsize=24, extent=(0, 45, 0, 45), cmap='Greys', mincnt=1, bins='log', lw=0.1)
        ax.plot([0, 45], [0, 45], color=C[m], lw=0.8, ls='--')
        rho = np.mean([z[z.gsm == g_][['native', 'n']].corr('spearman').iloc[0, 1] for g_ in DONOR])
        ax.text(2, 43, f'{LAB[m]}\nmean ρ = {rho:.2f}', fontsize=6.2, va='top', color=C[m] if m != 'bicubic' else '#555555', fontweight='bold')
        ax.set_xlim(0, 45); ax.set_ylim(0, 45); ax.set_aspect('equal'); ax.set_xticks([0, 20, 40]); ax.set_yticks([0, 20, 40])
        ax.set_xlabel('Nuclei per spot, native'); ax.set_ylabel('Nuclei per spot,\nreconstructed' if k == 0 else '')
        if k == 0: ax.set_title('Counts fall below identity; rank is kept', loc='left')
    ax = fig.add_subplot(r3[0, 3]); tag(ax, 'f')
    ab = pd.read_csv(PA / 'model_ablation_long.csv')
    feats_ = [('density', 'Density'), ('area_median', 'Area'), ('area_cv', 'Area CV')]
    ms_ = [('bicubic', 'bicubic'), ('old_SR08', 'old'), ('calibrated_SR', 'sr')]
    for j, (fe, fl) in enumerate(feats_):
        z = ab[ab.feature == fe].pivot_table(index='gsm', columns='method', values='rho')[[mk for mk, _ in ms_]]
        xs = j * 3.6 + np.arange(3)
        for _, row in z.iterrows(): ax.plot(xs, row.values, color='#cfcfcf', lw=0.5, zorder=1)
        for i, (mk, m) in enumerate(ms_):
            ax.scatter([xs[i]] * len(z), z[mk], s=11, color=C[m], edgecolor='white', linewidth=0.3, zorder=3)
            ax.plot([xs[i] - 0.32, xs[i] + 0.32], [z[mk].mean()] * 2, color='black', lw=0.9, zorder=4)
    ax.axhline(0, color='#bbbbbb', lw=0.5)
    ax.set_xticks([1, 4.6, 8.2], [f for _, f in feats_]); ax.set_ylabel('ρ vs native, per donor'); ax.set_ylim(-0.12, 0.95); ax.set_xlim(-0.6, 9.8)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=3.5, color=C[m], label=LAB[m]) for _, m in ms_], fontsize=5.8, loc='upper right', handletextpad=0.2)
    ax.set_title('Real hires, 3 donors', loc='left')
    save(fig, 'Fig3')


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig = plt.figure(figsize=(W, 155 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.3, left=0.07, right=0.99, top=0.95, bottom=0.08)
    sm = pd.read_csv(HB / 'spot_metrics_by_sample.csv')
    cancer = sm.drop_duplicates('sample').set_index('sample').cancer
    rho_aa = sm[(sm.feature == 'density') & (sm.condition == 'lower_aa0') & (sm.method == 'calibrated_SR')].set_index('sample').rho
    order = rho_aa.sort_values().index.tolist()  # ascending density ρ of calibrated SR under the generator shown in the gallery (AA)
    # a gallery of 9 slides: one field each; top hires (AA generator), middle SR, bottom native; CellViT contours on SR and native
    r1 = gs[0].subgridspec(3, 9, wspace=0.04, hspace=0.06)
    for j, sid in enumerate(order):
        pt = pd.read_csv(HB / 'reference/samples' / sid / 'patches.csv')
        pool = pt[pt.image_eligible].patch_id.sample(min(200, int(pt.image_eligible.sum())), random_state=3)
        cs_ = contours(HB / 'reference/samples' / sid / 'mosaic/native', pool.tolist())
        # field with the most native CellViT nuclei in the central 64 µm (200 random candidates per slide, fixed seed)
        cnt = pool.map(lambda p_: sum(((c_.mean(0) > 128) & (c_.mean(0) < 384)).all() for c_, _ in cs_[p_]))
        pid = pool.iloc[int(np.argmax(cnt.to_numpy()))]
        csr = contours(HB / 'lower_aa0/samples' / sid / 'mosaic/sr', [pid])[pid]
        for i, (cond, m, t, nn_, cc_) in enumerate([('lower_aa0', 'lr', 'Hires (AA)', True, None), ('lower_aa0', 'sr', 'SR (calibrated)', False, csr),
                                                    ('reference', 'native', 'Native', False, cs_[pid])]):
            a = fig.add_subplot(r1[i, j]); im_ = img(HB / cond / 'samples' / sid / 'patches' / m / f'{pid}.png')
            if im_.shape[0] < 512: im_ = np.repeat(np.repeat(im_, 512 // im_.shape[0], 0), 512 // im_.shape[1], 1)
            micro(a, im_[128:384, 128:384], nearest=nn_, um_px=0.25, bar_um=10, label=(i == 2 and j == 0))
            if cc_ is not None:
                inside = [(c_ - 128, t_) for c_, t_ in cc_ if ((c_.mean(0) > 128) & (c_.mean(0) < 384)).all()]
                overlay(a, inside, col=NUC, lw=0.45); a.set_xlim(-0.5, 255.5); a.set_ylim(255.5, -0.5)
                a.text(6, 6, f'{len(inside)}', color='white', fontsize=5.8, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0))
            if i == 0: a.set_title(f'{cancer[sid]}\n{sid}\nρ = {rho_aa[sid]:.2f}', fontsize=6, fontweight='normal', pad=2, linespacing=1.1)
            if j == 0: a.set_ylabel(t, fontsize=6.3)
            if i == 0 and j == 0: tag(a, 'a')
    r2 = gs[1].subgridspec(1, 3, width_ratios=[0.55, 1.3, 1], wspace=0.5)
    # b the two generators bracket real hires sharpness
    ax = fig.add_subplot(r2[0, 0]); tag(ax, 'b')
    mc = pd.read_csv(R9 / 'hires_reproduction_2026-10-29/method_comparison_all.csv')
    for k, (m, nm, cc) in enumerate([('opencv_area', 'Box', '#555555'), ('skimage_resize_rescale_aa_order0', 'AA', '#9c9c9c')]):
        v = np.log2(mc[mc.method == m].laplacian_variance_ratio)
        ax.scatter(k + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=12, color=cc, edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='#333333', ls='--', lw=0.6); ax.text(1.5, 0.04, 'Real hires', fontsize=5.8, ha='right', va='bottom')
    ax.set_xticks([0, 1], ['Box', 'AA']); ax.set_xlim(-0.5, 1.5); ax.set_ylabel('log$_2$ sharpness, generated / real')
    ax.set_title(f'Two generators bracket\nreal hires ({mc[mc.method == "opencv_area"].shape[0]} real pairs)', loc='left')
    # c Δρ heatmap (rows in the column order of a)
    ax = fig.add_subplot(r2[0, 1]); tag(ax, 'c')
    cols = [(f, c) for f in ['density', 'area_median', 'area_cv'] for c in ['upper_area', 'lower_aa0']]
    M = np.array([[sm[(sm['sample'] == s) & (sm.feature == f) & (sm.condition == c) & (sm.method == 'calibrated_SR')].rho.iloc[0] -
                   sm[(sm['sample'] == s) & (sm.feature == f) & (sm.condition == c) & (sm.method == 'bicubic')].rho.iloc[0] for f, c in cols] for s in order])
    im_ = ax.imshow(M, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]): ax.text(j, i, f'{M[i, j]:+.2f}'.replace('-', '−'), ha='center', va='center', fontsize=5.6, color='white' if abs(M[i, j]) > 0.3 else 'black')
            else: ax.text(j, i, 'n/a', ha='center', va='center', fontsize=5.4, color='#555555')
    ax.set_yticks(range(len(order)), order, fontsize=6)
    ax.set_xticks(range(len(cols)), ['Box', 'AA'] * 3, fontsize=6); ax.tick_params(length=0)
    for j, fl in enumerate(['Density', 'Area', 'Area CV']): ax.text(2 * j + 0.5, -0.75, fl, ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    for x in [1.5, 3.5]: ax.axvline(x, color='white', lw=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im_, ax=ax, fraction=0.05, pad=0.03); cb.set_label('ρ(SR) − ρ(bicubic)', fontsize=6); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    npos = int((M[np.isfinite(M)] > 0).sum()); nall = int(np.isfinite(M).sum())
    ax.set_title(f'SR beats bicubic in {npos}/{nall} comparisons', loc='left', pad=14)
    # d density ρ: three methods under both generators; mean difference of calibrated SR vs SR08 annotated
    ax = fig.add_subplot(r2[0, 2]); tag(ax, 'd')
    for j, (cond, mk, nm) in enumerate([('upper_area', 'o', 'Box'), ('lower_aa0', 's', 'AA')]):
        z = sm[(sm.feature == 'density') & (sm.condition == cond)].pivot(index='sample', columns='method', values='rho').reindex(order)
        for i in range(len(z)): ax.plot([j * 4, j * 4 + 1, j * 4 + 2], z.loc[z.index[i], ['bicubic', 'old_SR08', 'calibrated_SR']], color='#dddddd', lw=0.4, zorder=0)
        for i, m in enumerate(['bicubic', 'old_SR08', 'calibrated_SR']):
            x = j * 4 + i; cm = {'bicubic': 'bicubic', 'old_SR08': 'old', 'calibrated_SR': 'sr'}[m]
            ax.scatter(x + np.random.default_rng(i).uniform(-0.15, 0.15, len(z)), z[m], s=9, color=C[cm], marker=mk, lw=0.3, edgecolor='white')
            ax.plot([x - 0.3, x + 0.3], [z[m].mean()] * 2, color='black', lw=0.9)
        dd = (z.calibrated_SR - z.old_SR08).mean()
        ax.text(j * 4 + 1.5, 1.0, f'calibrated − SR08\n{dd:+.3f}'.replace('-', '−').replace('calibrated −', 'calibrated −'), ha='center', va='bottom', fontsize=5.8, color=C['sr'], linespacing=1.1)
    ax.set_xticks([1, 5], ['Box\n(sharper)', 'AA\n(blurrier)']); ax.set_ylabel('Spot density ρ vs native'); ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.05, 1.3)
    ax.spines['left'].set_bounds(0, 1); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title('Calibration matters more\nunder the blurrier bracket', loc='left')
    P.method_legend(ax, ['bicubic', 'old', 'sr'], loc='upper center', fontsize=5.8, ncol=3, bbox_to_anchor=(0.5, 1.01), handletextpad=0.1, columnspacing=0.6)
    save(fig, 'Fig4')


# ---------------------------------------------------------------- Fig 5
def emt_map(ax, xy, v, sf, title):
    from scipy.spatial import cKDTree
    Pp = np.c_[xy[:, 0], xy[:, 1]]; ok = np.isfinite(v); _, nb = cKDTree(Pp[ok]).query(Pp[ok], k=7)
    vs = pd.Series(pd.Series(v[ok]).rank(pct=True).to_numpy()[nb].mean(1)).rank(pct=True).to_numpy()
    sc = ax.scatter(Pp[ok, 0] * sf, Pp[ok, 1] * sf, c=vs, s=0.9, cmap='viridis', vmin=0, vmax=1, lw=0, rasterized=True)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off'); ax.set_title(title, fontsize=6.3, fontweight='normal', pad=2)
    return sc


def fig5():
    """Millimetre-fixed layout: panels in a row share top and bottom edges and titles share a baseline; f shows EMT and SR eccentricity per section (bicubic stated below)."""
    WM, HM = 178, 172
    fig = plt.figure(figsize=(WM * MM, HM * MM))
    def AX(x, y, w, h): return fig.add_axes([x / WM, 1 - (y + h) / HM, w / WM, h / HM])
    def head(x, y, t, tg=None, tx=None):
        fig.text(x / WM, 1 - y / HM, t, fontsize=7, fontweight='bold', ha='left', va='bottom')
        if tg: fig.text(tx / WM, 1 - y / HM, tg, fontsize=9, fontweight='bold', ha='left', va='bottom')
    def maplim(ax, ex):
        ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[3], ex[2]); ax.set_aspect('equal', adjustable='datalim'); ax.set_anchor('C')
    g = 'GSM6505133'
    # ---------- row 1: a association matrices; b matrix recovery; c per-program prediction
    T1, Y1, H1 = 6, 12, 26
    A = pd.read_csv(PA / 'C/A_paired' / g / 'associations.csv')
    feats = ['density', 'area_median', 'area_cv', 'eccentricity_median', 'h_mean', 'h_std', 'gray_lapvar']
    fl = ['Density', 'Area', 'Area CV', 'Eccentricity', 'Haem. mean', 'Haem. SD', 'Texture']
    progs = ['EMT', 'hypoxia', 'proliferation', 'classical', 'basal', 'interferon', 'immune']
    pl = ['EMT', 'Hypoxia', 'Prolif.', 'Classical', 'Basal-like', 'IFN', 'Immune']
    head(6, T1, 'Feature × program correlation (Donor A)', 'a', 1)
    for k, m in enumerate(['native', 'sr', 'bicubic']):
        a = AX(17 + k * 19, Y1, 18, H1)
        z = A[A.method == m].pivot(index='feature', columns='program', values='rho').reindex(index=feats, columns=progs)
        im_ = a.imshow(z, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        a.set_xticks(range(7), pl, rotation=60, ha='right', rotation_mode='anchor', fontsize=6); a.set_yticks(range(7), fl if k == 0 else [], fontsize=6)
        a.tick_params(length=0); [s_.set_visible(False) for s_ in a.spines.values()]
        a.set_title(LAB[m], fontsize=6.5, fontweight='normal', pad=2)
    cb = fig.colorbar(im_, cax=AX(74.5, Y1, 1.4, H1), ticks=[-0.4, 0, 0.4]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4); cb.ax.set_title('ρ', fontsize=6.5, pad=2)
    head(88, T1, 'Matrix recovery', 'b', 83)
    ax = AX(96, Y1, 20, H1)
    rp = pd.read_csv(R9 / 'association_recovery_2026-10-29/recovery_by_sample.csv').pivot(index='gsm', columns='method', values='matrix_element_pearson_r')
    rh = pd.read_csv(HB / 'association_recovery_by_sample.csv')
    sets = [('Real\nhires', rp.bicubic, rp.sr)]
    for cond, t in [('upper_area', 'HEST\nbox'), ('lower_aa0', 'HEST\nAA')]:
        z = rh[rh.condition == cond].pivot(index='sample', columns='method', values='matrix_element_pearson_r'); sets.append((t, z.bicubic, z.calibrated_SR))
    for k, (t, b_, s_) in enumerate(sets):
        paired(ax, 2.5 * k, 2.5 * k + 1, b_, s_, C['bicubic'], C['sr'], s=7); ax.text(2.5 * k + 0.5, 1.12, f'{(s_ > b_).sum()}/{len(s_)}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks([0.5, 3, 5.5], [t for t, _, _ in sets], fontsize=6); ax.set_ylim(-0.5, 1.1); ax.set_xlim(-0.5, 6.5); ax.set_yticks([-0.4, 0, 0.4, 0.8])
    ax.set_ylabel('r with native matrix', fontsize=6.5)
    head(122, T1, 'Predicting each program', 'c', 117)
    pr = pd.read_csv(CT / 'prediction_cv_r2.csv'); pr = pr[(pr.feature_set == 'morph+types')]
    allp = ['EMT', 'hypoxia', 'proliferation', 'classical', 'basal', 'interferon', 'immune', 'epithelial']
    apl = ['EMT', 'Hypoxia', 'Prolif.', 'Classical', 'Basal-like', 'IFN', 'Immune', 'Epithelial']
    for j, (cnd, t) in enumerate([('real_hires', 'Real hires'), ('upper_area', 'HEST box'), ('lower_aa0', 'HEST AA')]):
        ax = AX(131 + j * 15.2, Y1, 14.4, H1)
        for i, p_ in enumerate(allp):
            v = [pr[(pr.condition == cnd) & (pr.method == m) & (pr.program == p_)].cv_r2.mean() for m in ['native', 'sr', 'bicubic']]
            ax.plot([min(v), max(v)], [i, i], color='#dddddd', lw=0.6, zorder=1)
            for m, v_ in zip(['native', 'sr', 'bicubic'], v): ax.scatter(v_, i, s=7, color=C[m], zorder=3, edgecolor='white', linewidth=0.25)
        ax.axvline(0, color='#bbbbbb', lw=0.5); ax.set_xlim(-0.06, 0.3); ax.set_xticks([0, 0.1, 0.2], ['0', '.1', '.2'], fontsize=6); ax.set_ylim(len(allp) - 0.5, -0.5)
        ax.set_yticks(range(len(allp)), apl if j == 0 else [], fontsize=6); ax.tick_params(axis='y', length=0)
        ax.set_title(t, fontsize=6.3, fontweight='normal', pad=2)
        if j == 1: ax.set_xlabel('Cross-validated R²', fontsize=6.5, labelpad=1)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=3.3, color=C[m], label=LAB[m]) for m in ['native', 'sr', 'bicubic']],
               loc='upper center', bbox_to_anchor=(153 / WM, 1 - (Y1 + H1 + 8.5) / HM), ncol=3, fontsize=5.8, handletextpad=0.1, columnspacing=0.8)
    # ---------- row 2: d Donor A maps; e seven paired sections
    T2, Y2, H2 = 56, 63, 31
    head(6, T2, 'EMT program and nuclear eccentricity (Donor A, paired)', 'd', 1)
    src = json.loads((PA / 'C/A_paired' / g / 'image_sources.json').read_text()); sf = src['sf']
    pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('barcode')
    asc = pd.read_csv(PA / 'C/A_paired' / g / 'associations.csv')
    allxy = pt[['fullres_x', 'fullres_y']].to_numpy() * sf
    exd = (allxy[:, 0].min(), allxy[:, 0].max(), allxy[:, 1].min(), allxy[:, 1].max())
    for k, (m, col, t) in enumerate([('native', 'EMT', 'EMT program\n(expression)'), ('native', 'eccentricity_median', 'Eccentricity, native'),
                                     ('sr', 'eccentricity_median', 'Eccentricity, SR'), ('bicubic', 'eccentricity_median', 'Eccentricity, bicubic')]):
        a = AX(7 + k * 31.5, Y2, 30, H2); z = pd.read_csv(PA / 'C/A_paired' / g / f'spot_expression_{m}.csv').set_index('barcode'); xy = pt.reindex(z.index)
        if col != 'EMT': t += f"\npartial ρ = {asc[(asc.method == m) & (asc.feature == col) & (asc.program == 'EMT')].partial_rho.iloc[0]:+.2f}"
        sc = emt_map(a, xy[['fullres_x', 'fullres_y']].to_numpy(), z[col].to_numpy(), sf, t); a.title.set_fontsize(6.3); maplim(a, exd)
    cb = fig.colorbar(sc, cax=AX(133.5, Y2 + 4, 1.4, H2 - 8), ticks=[0, 1]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4); cb.set_label('Percentile', fontsize=6, labelpad=-3)
    head(148, T2, '7 paired sections', 'e', 143)
    ax = AX(154, Y2, 22, H2)
    pa = pd.read_csv(PA / 'C/A_paired_associations.csv')
    q = pa[(pa.feature == 'eccentricity_median') & (pa.program == 'EMT')].pivot(index='gsm', columns='method', values='partial_rho')
    for i, m in enumerate(['native', 'sr', 'bicubic']):
        ax.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(q)), q[m], s=8, color=C[m], edgecolor='white', linewidth=0.3, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [q[m].mean()] * 2, color='black', lw=0.9)
        ax.text(i, 0.43, f'{(q[m] > 0).sum()}/{len(q)}', ha='center', va='bottom', fontsize=6)
    ax.axhline(0, color='#bbbbbb', lw=0.5); ax.set_xticks([0, 1, 2], ['Native', 'SR', 'Bicubic'], fontsize=6); ax.set_ylim(-0.1, 0.5); ax.set_xlim(-0.5, 2.5)
    ax.set_yticks([0, 0.2, 0.4]); ax.set_ylabel('Partial ρ, eccentricity ~ EMT', fontsize=6.5)
    ax.set_title('numbers: sections > 0', fontsize=6, fontweight='normal', pad=2)
    # ---------- row 3: f four hires-only sections (EMT, SR eccentricity; bicubic stated below); g partial correlations
    T3, Y3a, Y3b, H3 = 103, 109, 140, 25
    head(6, T3, 'Four PDAC sections deposited as hires only', 'f', 1)
    hb = pd.read_csv(PA / 'C/B_hires_associations.csv')
    hq = hb[(hb.feature == 'eccentricity_median') & (hb.program == 'EMT')].pivot(index='gsm', columns='method', values='partial_rho')
    hn = hb[(hb.feature == 'eccentricity_median') & (hb.program == 'EMT')].pivot(index='gsm', columns='method', values='n')
    gl = hq.sr.sort_values(ascending=False).index.tolist()
    fmt = lambda v: 'n/a' if not np.isfinite(v) else f'{v:+.2f}'.replace('-', '−')
    for j, gh in enumerate(gl):
        srcb = json.loads((PA / 'C/B_hires' / gh / 'image_sources.json').read_text()); sfb = srcb['sf']
        ptb = pd.read_csv(HO / gh / 'patches.csv').set_index('barcode')
        zsr = pd.read_csv(PA / 'C/B_hires' / gh / 'spot_expression_sr.csv').set_index('barcode'); zbc = pd.read_csv(PA / 'C/B_hires' / gh / 'spot_expression_bicubic.csv').set_index('barcode')
        xy = ptb.reindex(zsr.index)[['fullres_x', 'fullres_y']].to_numpy()
        P_ = xy[np.isfinite(zsr['EMT'].to_numpy())] * sfb
        ex = (*np.percentile(P_[:, 0], [0.5, 99.5]), *np.percentile(P_[:, 1], [0.5, 99.5]))  # 0.5–99.5% percentiles set the extent, excluding isolated outlying spots
        mg = 0.03 * max(ex[1] - ex[0], ex[3] - ex[2]); ex = (ex[0] - mg, ex[1] + mg, ex[2] - mg, ex[3] + mg)
        x0 = 7 + j * 31.5
        a = AX(x0, Y3a, 30, H3); emt_map(a, xy, zsr['EMT'].to_numpy(), sfb, gh); a.title.set_fontsize(6.3); maplim(a, ex)
        cov_s, cov_b = np.isfinite(zsr.eccentricity_median.to_numpy()).mean(), np.isfinite(zbc.eccentricity_median.to_numpy()).mean()
        a = AX(x0, Y3b, 30, H3)
        sc = emt_map(a, xy, zsr['eccentricity_median'].to_numpy(), sfb, f'SR {100 * cov_s:.0f}% of spots · ρ {fmt(hq.at[gh, "sr"])}'); a.title.set_fontsize(6); maplim(a, ex)
        fig.text((x0 + 15) / WM, 1 - (Y3b + H3 + 1.2) / HM, f'Bicubic {100 * cov_b:.0f}% · ρ {fmt(hq.at[gh, "bicubic"])}',
                 ha='center', va='top', fontsize=6, color='#666666')
    for y_, t in [(Y3a, 'EMT program'), (Y3b, 'Eccentricity, SR')]:
        fig.text(5.5 / WM, 1 - (y_ + H3 / 2) / HM, t, rotation=90, ha='right', va='center', fontsize=6.3)
    cb = fig.colorbar(sc, cax=AX(133.5, Y3a + 4, 1.4, Y3b + H3 - Y3a - 8), ticks=[0, 1]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4); cb.set_label('Percentile', fontsize=6, labelpad=-3)
    head(148, T3, f'SR > 0 in {int((hq.sr > 0).sum())}/{len(hq)}', 'g', 143)
    ax = AX(154, Y3a, 22, Y3b + H3 - Y3a)
    for gg in gl:
        b_, s_ = hq.at[gg, 'bicubic'], hq.at[gg, 'sr']; few = hn.at[gg, 'sr'] < 500
        if np.isfinite(b_):
            ax.plot([0, 1], [b_, s_], color='#cccccc', lw=0.6); ax.scatter([0], [b_], s=10, color=C['bicubic'], zorder=3, edgecolor='white', linewidth=0.3)
        ax.scatter([1], [s_], s=10, color='white' if few else C['sr'], edgecolor=C['sr'], linewidth=0.8, zorder=3)
    for i, m in enumerate(['bicubic', 'sr']):
        v = hq[m].dropna(); ax.plot([i - 0.22, i + 0.22], [v.median()] * 2, color='black', lw=0.9)
    ax.axhline(0, color='#bbbbbb', lw=0.5); ax.set_xticks([0, 1], ['Bicubic', 'SR'], fontsize=6); ax.set_ylim(-0.1, 0.42); ax.set_xlim(-0.5, 1.5); ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.spines['left'].set_bounds(-0.1, 0.3); ax.set_ylabel('Partial ρ, eccentricity ~ EMT', fontsize=6.5)
    ax.text(0.03, 0.98, f"median SR {fmt(hq.sr.median())}\nmedian bicubic {fmt(hq.bicubic.median())}\nbicubic n/a: 1 section\nopen: < 500 spots", transform=ax.transAxes,
            fontsize=5.6, va='top', ha='left', color='#444444', linespacing=1.15)
    P.TAGS[:] = [t_ for t_ in P.TAGS if t_[0].figure is not fig]
    fig.savefig(OUT / 'Fig5.pdf'); fig.savefig(OUT / 'Fig5.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}); fig.savefig(OUT / 'Fig5.png', dpi=300)
    plt.close(fig); print('saved Fig5', flush=True)


if __name__ == '__main__':
    for w in sys.argv[1:] or '12345':
        globals()[f'fig{w}']()
