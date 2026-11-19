"""Publication-format main figures (Briefings in Bioinformatics): new layout, same data as sr10.
Writes results/sr_paper_figures_2026-11-12_pub/: PDF (vector), 600 dpi TIFF (LZW), 300 dpi PNG (preview).
Style: width 178 mm; Arial; text 7 pt, minimum 6 pt; panel tags 9 pt bold, aligned per row; method names
Bicubic / SR08 / SR (calibrated); micrographs with outlined scale bars. Usage: python scripts/sr13_paper_figures_pub.py [1 2 3 4 5]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr10_paper_figures as F
from sr10_paper_figures import R, R9, S65, PA, HB, CT, img, hires, lapvar

OUT = R / 'sr_paper_figures_2026-11-12_pub'
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
W = 178 * MM
C = {'native': '#333333', 'sr': '#e8612c', 'old': '#2a78d6', 'bicubic': '#a6a6a6'}
LAB = {'native': 'Native', 'sr': 'SR (calibrated)', 'old': 'SR08', 'bicubic': 'Bicubic'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
                     'axes.titleweight': 'bold', 'axes.titlepad': 4,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'lines.linewidth': 0.8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'legend.frameon': False, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'legend.handletextpad': 0.4, 'legend.borderaxespad': 0.2})
DONOR = {'GSM6505133': 'Donor A', 'GSM6505134': 'Donor B', 'GSM6505135': 'Donor C'}
TAGS = []
STROKE = [pe.withStroke(linewidth=1.6, foreground='black')]


def tag(ax, s):
    TAGS.append((ax, s))


def place_tags(fig):
    """Place tags at the top left of each panel (including axis labels); tags in the same row (tops within 0.03) share a height."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    mine = [(ax, s, ax.get_tightbbox(r).transformed(inv)) for ax, s in TAGS if ax.figure is fig]
    rows = []
    for ax, s, bb in sorted(mine, key=lambda t: -t[2].y1):
        for row in rows:
            if abs(row[0][2].y1 - bb.y1) < 0.03:
                row.append((ax, s, bb)); break
        else:
            rows.append([(ax, s, bb)])
    for row in rows:
        y = max(bb.y1 for _, _, bb in row)
        for ax, s, bb in row:
            fig.text(bb.x0 - 0.004, y + 0.006, s, fontsize=9, fontweight='bold', ha='left', va='bottom')
    TAGS[:] = [t for t in TAGS if t[0].figure is not fig]


def save(fig, name):
    TAGS.extend(F.TAGS); F.TAGS.clear()  # sr11 tags are stored in the sr10 module
    place_tags(fig)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.tif', dpi=600, bbox_inches='tight', pad_inches=0.02, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', name, flush=True)


def micro(ax, im, title=None, nearest=False, um_px=None, bar_um=20, label=True):
    """Micrograph: no axes, thin frame; white scale bar with black outline, label above the bar."""
    ax.imshow(im, interpolation='nearest' if nearest else 'antialiased')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.4); s.set_color('#444444')
    if um_px:
        h, w = im.shape[:2]; L = bar_um / um_px
        x1 = w * 0.93; y = h * 0.92
        # the outline must be wider than the bar itself to be visible on white backgrounds
        ax.plot([x1 - L, x1], [y, y], color='white', lw=1.8, solid_capstyle='butt', path_effects=[pe.withStroke(linewidth=2.8, foreground='black')])
        if label:
            ax.text(x1 - L / 2, y - h * 0.035, f'{bar_um} µm', color='white', fontsize=6, ha='center', va='bottom',
                    path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
    if title: ax.set_title(title, fontsize=6.5, fontweight='normal', pad=2)


def paired(ax, x0, x1, a, b, ca, cb, s=11):
    for u, v in zip(a, b): ax.plot([x0, x1], [u, v], color='#cfcfcf', lw=0.5, zorder=1)
    ax.scatter([x0] * len(a), a, s=s, color=ca, zorder=3, edgecolor='white', linewidth=0.3)
    ax.scatter([x1] * len(b), b, s=s, color=cb, zorder=3, edgecolor='white', linewidth=0.3)


def method_legend(ax_or_fig, keys, **kw):
    h = [Line2D([], [], marker='o', ls='', ms=4, color=C[k], label=LAB[k]) for k in keys]
    return ax_or_fig.legend(handles=h, **kw)


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig = plt.figure(figsize=(W, 112 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.82], width_ratios=[1, 0.95, 1.05], hspace=0.42, wspace=0.32,
                          left=0.03, right=0.99, top=0.95, bottom=0.08)
    # a
    ax = fig.add_subplot(gs[0, 0]); tag(ax, 'a')
    s = pd.read_csv(R / 'sr_survey/geo_visium_image_survey.csv').category.value_counts()
    cats = [('hires_only', 'Hires PNG only'), ('bundle_only', 'Image only in archive'), ('fullres', 'Full-resolution image'),
            ('fullres_unc', 'Full-resolution (unconfirmed)'), ('no_image', 'No image')]
    vals = [s.get(k, 0) for k, _ in cats]
    y = np.arange(len(cats))[::-1]
    ax.barh(y, vals, color=[C['sr']] + ['#bdbdbd'] * 4, height=0.62)
    for yi, v in zip(y, vals): ax.text(v + 4, yi, f'{v}', va='center', fontsize=6.5)
    ax.set_yticks(y, [l for _, l in cats]); ax.set_xlabel('Human Visium series in GEO')
    ax.set_xlim(0, max(vals) * 1.2); ax.tick_params(axis='y', length=0)
    ax.set_title(f'Deposited images ({s.sum()} series)', loc='left')
    # b overview
    g, pid = 'GSM6505133', 'p00093'
    src = json.loads((PA / 'C/A_paired' / g / 'image_sources.json').read_text())
    H = hires(src['hires'])
    pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('patch_id')
    ax = fig.add_subplot(gs[0, 1]); tag(ax, 'b')
    cx, cy = pt.loc[pid, 'fullres_x'] * src['sf'], pt.loc[pid, 'fullres_y'] * src['sf']
    sat = cv2.cvtColor(H, cv2.COLOR_RGB2HSV)[..., 1]; ys, xs = np.where(cv2.medianBlur(sat, 5) > 40)
    ax.imshow(H); ax.set_xlim(xs.min() - 15, xs.max() + 15); ax.set_ylim(ys.max() + 15, ys.min() - 15)
    um_px_h = 65 / (295.381 * src['sf'])
    half = 64 / um_px_h
    ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec=C['sr'], lw=1.4))
    L = 1000 / um_px_h; x1 = xs.max() - 10; yb = ys.max() + 5
    ax.plot([x1 - L, x1], [yb, yb], color='black', lw=1.5, solid_capstyle='butt')
    ax.text(x1 - L / 2, yb - 12, '1 mm', ha='center', va='bottom', fontsize=6)
    ax.axis('off'); ax.set_title(f'Public hires image ({um_px_h:.1f} µm/px)', loc='left')
    # c 2×2 fields
    base = S65 / 'paired/samples' / g / 'patches'
    sub = gs[0, 2].subgridspec(2, 2, wspace=0.06, hspace=0.16)
    for k, (m, t, nn) in enumerate([('lr', 'Hires (4 µm/px)', True), ('bicubic', 'Bicubic', False),
                                     ('sr', 'SR (calibrated)', False), ('native', 'Native scan', False)]):
        a = fig.add_subplot(sub[k // 2, k % 2])
        micro(a, img(base / m / f'{pid}.png'), t, nearest=nn, um_px=0.25 if m == 'native' else None)
        if k == 0: tag(a, 'c')
    # d pipeline
    ax = fig.add_subplot(gs[1, 0:2]); tag(ax, 'd'); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
    boxes = [('TCGA FFPE\nslides\n0.25 µm/px', '#eeeeee'), ('Degrade to\n4 µm/px\n(calibrated blur)', '#fbe0d3'),
             ('SwinIR ×8\n+ bicubic ×2', '#fbe0d3'), ('CellViT\nnuclei', '#eeeeee'), ('Spot features\nvs\nexpression', '#eeeeee')]
    bw, gap = 17.6, 2.9
    for i, (t, col) in enumerate(boxes):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, 26), bw, 15, boxstyle='round,pad=0,rounding_size=1.6', fc=col, ec='#666666', lw=0.6))
        ax.text(x + bw / 2, 33.5, t, ha='center', va='center', fontsize=6.3)
        if i < 4: ax.annotate('', (x + bw + gap - 0.3, 33.5), (x + bw + 0.3, 33.5), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#555555', mutation_scale=7))
    roles = [('Training', 'TCGA-PAAD, 69 slides'), ('Blur calibration', 'GSE327056, 4 native–hires pairs'),
             ('Real-hires test', 'GSE211895, 3 held-out donors'), ('External test', 'HEST, 9 slides, 6 cancer types'),
             ('Reference', 'CellViT on the native scan of the same section')]
    for i, (k, v) in enumerate(roles):
        yy = 16 - i * 3.6
        ax.text(0, yy, k, fontsize=6.3, fontweight='bold', va='center'); ax.text(19, yy, v, fontsize=6.3, va='center')
    # e scale
    ax = fig.add_subplot(gs[1, 2]); tag(ax, 'e')
    a = pd.read_csv(PA / 'scale_audit.csv'); a = a[a.gse.isin(['GSE211895', 'GSE327056'])]
    p55 = a.neighbor_pitch_fullres_px * 55 / a.spot_diameter_fullres
    p65 = a.neighbor_pitch_fullres_px * 65 / a.spot_diameter_fullres
    rng = np.random.default_rng(0)
    for k, (v, col) in enumerate([(p55, '#bdbdbd'), (p65, C['sr'])]):
        ax.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=14, color=col, edgecolor='white', linewidth=0.3, zorder=3)
    ax.axhline(100, color='#333333', lw=0.6, ls='--'); ax.text(1.45, 100.6, 'Design pitch, 100 µm', fontsize=6, ha='right', va='bottom')
    ax.set_xticks([0, 1], ['Read as 55 µm', 'Read as 65 µm']); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('Neighbouring-spot pitch (µm)'); ax.set_ylim(80, 104)
    ax.set_title('Pixel size from spot_diameter_fullres', loc='left')
    save(fig, 'Fig1')


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig, axs = plt.subplots(1, 4, figsize=(W, 62 * MM), gridspec_kw=dict(width_ratios=[1.45, 1, 1, 0.95], wspace=0.55,
                                                                       left=0.07, right=0.99, top=0.82, bottom=0.2))
    ax = axs[0]; tag(ax, 'a')
    for k, s in enumerate(['x2', 'x4', 'x8']):
        d = pd.read_csv(R / f'sr08/eval_{s}/metrics_by_sample.csv')
        d = d[d.testset.str.startswith('HEST') & (d.group == 'FFPE')]
        w = d.pivot_table(index='sample', columns='method', values='f1')
        paired(ax, 3 * k, 3 * k + 1, w.bicubic, w.SwinIR, C['bicubic'], C['old'])
    ax.set_xticks([0.5, 3.5, 6.5], ['×2\n1 µm/px', '×4\n2 µm/px', '×8\n4 µm/px']); ax.set_ylabel('Nucleus detection F1')
    ax.set_title('Detection (12 HEST FFPE slides)', loc='left'); ax.set_ylim(0, 1); ax.set_xlim(-0.6, 7.6)
    d = pd.read_csv(R / 'sr08/eval_x8/metrics_by_sample.csv')
    for ax, (col, lab, t, tt) in zip(axs[1:3], [('abs_log_area_med', '|log area ratio|, matched nuclei', 'b', 'Nuclear size error, ×8'),
                                                ('rho_density', 'Tile density ρ vs reference', 'c', 'Nuclear density, ×8')]):
        tag(ax, t)
        for k, ts in enumerate(['HEST', 'Xenium']):
            z = d[d.testset.str.startswith(ts)].pivot_table(index='sample', columns='method', values=col)
            paired(ax, 3 * k, 3 * k + 1, z.bicubic, z.SwinIR, C['bicubic'], C['old'])
        ax.set_xticks([0.5, 3.5], ['HEST\n(n = 28)', 'Xenium DAPI\n(n = 12)']); ax.set_ylabel(lab); ax.set_xlim(-0.6, 4.6)
        ax.set_title(tt, loc='left')
    axs[1].set_ylim(0, 0.7); axs[2].set_ylim(0, 1)
    ax = axs[3]; tag(ax, 'd')
    lc = pd.read_csv(R / 'sr08/annotation_x8/analysis_review_2026-10-25/label_counts.csv')
    lc = lc[(lc.cat == 'unmatched') & lc.stratum.isin(['primary', 'generalization'])].groupby(['method', 'final_label']).n.sum().unstack().fillna(0)
    bottom = np.zeros(2); order = ['bicubic', 'SwinIR']
    for lab_, col, nm in [('yes', '#5a5a5a', 'Nucleus'), ('uncertain', '#cfcfcf', 'Uncertain'), ('no', '#c62828', 'Not a nucleus')]:
        v = lc.reindex(order)[lab_].to_numpy() if lab_ in lc else np.zeros(2)
        ax.bar([0, 1], v, bottom=bottom, color=col, width=0.6, label=nm); bottom += v
    for i, o in enumerate(order):
        r = lc.loc[o]; ax.text(i, bottom[i] + 6, f"{int(r.get('no', 0))}/{int(r.sum())}", ha='center', fontsize=6.5)
    ax.set_xticks([0, 1], ['Bicubic', 'SR08']); ax.set_ylabel('Unmatched detections (n)'); ax.set_ylim(0, 290)
    ax.set_title('Blinded review', loc='left')
    ax.legend(loc='upper center', fontsize=6, ncol=1, bbox_to_anchor=(0.5, 1.0), handlelength=0.9, labelspacing=0.25)
    method_legend(fig, ['bicubic', 'old'], loc='upper left', bbox_to_anchor=(0.06, 0.985), ncol=2, fontsize=6.5)
    save(fig, 'Fig2')


# ---------------------------------------------------------------- Fig 3
def fig3():
    import importlib
    v2 = importlib.import_module('sr02_degrade_benchmark').degrade_real2
    cal = importlib.import_module('sr09_scale65_degrade').degrade_real2
    fig = plt.figure(figsize=(W, 128 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45, left=0.065, right=0.99, top=0.95, bottom=0.08)
    top = gs[0].subgridspec(1, 3, width_ratios=[0.8, 2.3, 0.8], wspace=0.35)
    # a
    ax = fig.add_subplot(top[0, 0]); tag(ax, 'a')
    L = pd.read_csv(R9 / 'B_sim_vs_real_2026-10-27_legacy_scale55/comparison_side_by_side.csv'); L = L[L.feature == 'density']
    for k, m in enumerate(['bicubic', 'sr']):
        z = L[L.method == m]; cc = C['bicubic'] if m == 'bicubic' else C['old']
        paired(ax, 3 * k, 3 * k + 1, z['rho__simulated_real2_4.0'], z['rho__true_hires'], cc, cc)
    ax.set_xticks([0, 1, 3, 4], ['Sim.', 'Real', 'Sim.', 'Real']); ax.set_ylim(0, 1); ax.set_xlim(-0.6, 4.6)
    for x, t in [(0.5, 'Bicubic'), (3.5, 'SR08')]:
        ax.text(x, -0.11, t, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.5, fontweight='bold')
    ax.set_ylabel('Spot density ρ vs native'); ax.set_title('Original pipeline', loc='left')
    # b images
    g, pid = 'GSM6505135', 'p00245'
    nat = img(S65 / 'paired/samples' / g / 'patches/native' / f'{pid}.png')
    real = img(S65 / 'paired/samples' / g / 'patches/lr' / f'{pid}.png')
    gt = cv2.resize(nat, (256, 256), interpolation=cv2.INTER_AREA)
    lv2, _ = v2(gt, 32, np.random.default_rng(7)); lcal, _ = cal(gt, 32, np.random.default_rng(7))
    sub = top[0, 1].subgridspec(1, 4, wspace=0.05)
    for k, (im_, t) in enumerate([(nat, 'Native scan\n '), (real, 'Real hires\n '), (lv2, 'Simulated,\noriginal blur'), (lcal, 'Simulated,\ncalibrated blur')]):
        a = fig.add_subplot(sub[0, k]); micro(a, im_, t, nearest=k > 0, um_px=0.25 if k == 0 else 4.0, label=k == 0)
        if k == 0: tag(a, 'b')
    # c
    ax = fig.add_subplot(top[0, 2]); tag(ax, 'c')
    lv = pd.read_csv(F.OUT / 'fig3c_laplacian_ratio.csv')
    rng = np.random.default_rng(0)
    for k, (col, cc) in enumerate([('v2', C['old']), ('cal', C['sr'])]):
        v = lv[col].to_numpy()
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=2.5, color=cc, alpha=0.45, lw=0, zorder=2)
        q = np.percentile(v, [25, 50, 75])
        ax.plot([k, k], [q[0], q[2]], color='black', lw=1.4, zorder=3); ax.scatter([k], [q[1]], color='white', edgecolor='black', s=14, zorder=4, linewidth=0.8)
    ax.axhline(0, color='#333333', lw=0.6, ls='--'); ax.set_xticks([0, 1], ['Original\nσ 0.3–1.2', 'Calibrated\nσ 1.5–1.75'])
    ax.set_xlim(-0.6, 1.6); ax.set_ylabel('log$_2$ sharpness, simulated / real'); ax.set_title(f'Sharpness (n = {len(lv)} fields)', loc='left')
    # d–g
    bot = gs[1].subgridspec(1, 4, wspace=0.55)
    ax = fig.add_subplot(bot[0, 0]); tag(ax, 'd')
    cc = pd.read_csv(S65 / 'calibration/calibration_sample_curves.csv')
    ax.axvspan(0.3, 1.2, color=C['old'], alpha=0.12, lw=0); ax.axvspan(1.5, 1.75, color=C['sr'], alpha=0.3, lw=0)
    for gg, z in cc.groupby('gsm'): ax.plot(z.sigma_um, z.loss, color='#555555', lw=0.7, marker='o', ms=1.8)
    ax.set_xlim(0, 6.5); ax.set_ylim(0, 5); ax.set_xlabel('Gaussian blur σ (µm)'); ax.set_ylabel('Mismatch to real hires')
    ax.text(0.75, 4.85, 'Original', fontsize=6, ha='center', va='top', color=C['old'])
    ax.text(1.95, 4.85, 'Calibrated', fontsize=6, va='top', color=C['sr'])
    ax.set_title('Blur calibration (4 pairs)', loc='left')
    ab = pd.read_csv(PA / 'model_ablation_long.csv')
    for k, (feat, t, yl) in enumerate([('density', 'Nuclear density', (0.35, 0.92)), ('area_median', 'Median nuclear area', (0.1, 0.45)),
                                       ('area_cv', 'Nuclear area CV', (-0.08, 0.14))]):
        ax = fig.add_subplot(bot[0, k + 1]); tag(ax, 'efg'[k])
        z = ab[ab.feature == feat]
        w = pd.DataFrame({'bicubic': z[z.method == 'bicubic'].groupby('gsm').rho.mean(),
                          'old': z[z.method == 'old_SR08'].groupby('gsm').rho.mean(),
                          'sr': z[z.method == 'calibrated_SR'].groupby('gsm').rho.mean()})
        for _, r in w.iterrows(): ax.plot([0, 1, 2], r[['bicubic', 'old', 'sr']], color='#cfcfcf', lw=0.5, zorder=1)
        for i, m in enumerate(['bicubic', 'old', 'sr']):
            ax.scatter([i] * len(w), w[m], color=C[m], s=13, zorder=3, edgecolor='white', linewidth=0.3)
            ax.plot([i - 0.22, i + 0.22], [w[m].mean()] * 2, color='black', lw=0.9, zorder=4)
            ax.text(i, yl[1] - 0.02 * (yl[1] - yl[0]), f'{w[m].mean():.2f}', ha='center', va='top', fontsize=6)
        ax.set_xticks([0, 1, 2], ['Bicubic', 'SR08', 'SR\n(calib.)']); ax.set_xlim(-0.5, 2.5); ax.set_ylim(*yl)
        ax.set_ylabel('ρ vs native'); ax.set_title(t, loc='left')
    fig.text(0.40, 0.505, 'Real hires, corrected scale (GSE211895, 3 donors; bar = mean)', fontsize=6.5, ha='left', color='#444444')
    save(fig, 'Fig3')


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig = plt.figure(figsize=(W, 122 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.15], hspace=0.42, left=0.065, right=0.99, top=0.95, bottom=0.17)
    top = gs[0].subgridspec(1, 3, width_ratios=[2.5, 0.75, 0.75], wspace=0.45)
    sid = 'TENX65'
    pt = pd.read_csv(HB / 'reference/samples' / sid / 'patches.csv')
    pid = pt[pt.image_eligible].patch_id.iloc[len(pt) // 3]
    sub = top[0, 0].subgridspec(1, 4, wspace=0.05)
    tiles = [(HB / 'reference/samples' / sid / 'patches/native' / f'{pid}.png', 'Native scan\n ', False, 0.25),
             (HB / 'upper_area/samples' / sid / 'patches/lr' / f'{pid}.png', 'Hires, box\n(sharper)', True, 4.0),
             (HB / 'lower_aa0/samples' / sid / 'patches/lr' / f'{pid}.png', 'Hires, anti-aliased\n(blurrier)', True, 4.0),
             (HB / 'lower_aa0/samples' / sid / 'patches/sr' / f'{pid}.png', 'SR of\nanti-aliased', False, 0.25)]
    for k, (p, t, nn, u) in enumerate(tiles):
        a = fig.add_subplot(sub[0, k]); im_ = img(p); micro(a, im_, t, nearest=nn, um_px=128 / im_.shape[1])
        if k == 0: tag(a, 'a'); a.set_ylabel('Ovarian carcinoma (TENX65)', fontsize=6)
    mc = pd.read_csv(R9 / 'hires_reproduction_2026-10-29/method_comparison_all.csv')
    rng = np.random.default_rng(0)
    gen = [('opencv_area', 'Box', '#555555'), ('skimage_resize_rescale_aa_order0', 'Anti-\naliased', '#9c9c9c')]
    for j, (col, yl, t, tg) in enumerate([('laplacian_variance_ratio', 'log$_2$ sharpness, generated / real', 'Sharpness', 'b'),
                                          ('psnr_db', 'PSNR vs real hires (dB)', 'Pixel agreement', 'c')]):
        ax = fig.add_subplot(top[0, 1 + j]); tag(ax, tg)
        for k, (m, _, cc) in enumerate(gen):
            v = mc[mc.method == m][col]; v = np.log2(v) if j == 0 else v
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=13, color=cc, edgecolor='white', linewidth=0.3)
        if j == 0: ax.axhline(0, color='#333333', ls='--', lw=0.6)
        ax.set_xticks([0, 1], [g_[1] for g_ in gen]); ax.set_xlim(-0.5, 1.5); ax.set_ylabel(yl); ax.set_title(t + ' (7 pairs)', loc='left')
    sm = pd.read_csv(HB / 'spot_metrics_by_sample.csv')
    cancer = sm.drop_duplicates('sample').set_index('sample').cancer
    order = sm[(sm.feature == 'density') & (sm.condition == 'upper_area') & (sm.method == 'calibrated_SR')].sort_values('rho')['sample'].tolist()
    bot = gs[1].subgridspec(1, 2, wspace=0.18)
    for k, (feat, t) in enumerate([('density', 'Nuclear density'), ('area_median', 'Median nuclear area')]):
        ax = fig.add_subplot(bot[0, k]); tag(ax, 'de'[k]); wins = []
        for j, (cond, dx, mk) in enumerate([('upper_area', -0.17, 'o'), ('lower_aa0', 0.17, 's')]):
            z = sm[(sm.feature == feat) & (sm.condition == cond)].pivot(index='sample', columns='method', values='rho').reindex(order)
            x = np.arange(len(order)) + dx
            for i in range(len(order)): ax.plot([x[i]] * 2, [z.bicubic.iloc[i], z.calibrated_SR.iloc[i]], color='#cfcfcf', lw=0.6, zorder=1)
            ax.scatter(x, z.bicubic, color=C['bicubic'], s=10, marker=mk, zorder=2, lw=0)
            ax.scatter(x, z.old_SR08, color=C['old'], s=10, marker=mk, zorder=2, lw=0)
            ax.scatter(x, z.calibrated_SR, color=C['sr'], s=13, marker=mk, zorder=3, edgecolor='white', linewidth=0.3)
            wins.append(f"{(z.calibrated_SR > z.bicubic).sum()}/9")
        ax.set_xticks(range(len(order)), [f'{cancer[s]}\n{s}' for s in order], fontsize=5.8, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.6, len(order) - 0.4); ax.axhline(0, color='#dddddd', lw=0.5, zorder=0)
        ax.set_ylabel('Spot-level ρ vs native'); ax.set_title(f'{t}  (SR > bicubic: box {wins[0]}, anti-aliased {wins[1]})', loc='left')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=4, color=C[m], label=LAB[m]) for m in ['bicubic', 'old', 'sr']] +
               [Line2D([], [], marker='o', ls='', ms=4, color='#666666', label='Box generator'),
                Line2D([], [], marker='s', ls='', ms=4, color='#666666', label='Anti-aliased generator')],
               loc='lower center', bbox_to_anchor=(0.53, 0.0), ncol=5, fontsize=6.5)
    save(fig, 'Fig4')


# ---------------------------------------------------------------- Fig 5
def fig5():
    from scipy.spatial import cKDTree
    fig = plt.figure(figsize=(W, 140 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.42, left=0.07, right=0.985, top=0.95, bottom=0.04)
    top = gs[0].subgridspec(1, 3, width_ratios=[2.25, 0.85, 0.95], wspace=0.6)
    g = 'GSM6505133'
    A = pd.read_csv(PA / 'C/A_paired' / g / 'associations.csv')
    feats = ['density', 'area_median', 'area_cv', 'eccentricity_median', 'h_mean', 'h_std', 'gray_lapvar']
    fl = ['Density', 'Area', 'Area CV', 'Eccentricity', 'Haematoxylin mean', 'Haematoxylin SD', 'Texture']
    progs = ['EMT', 'hypoxia', 'proliferation', 'classical', 'basal', 'interferon', 'immune']
    pl = ['EMT', 'Hypoxia', 'Proliferation', 'Classical', 'Basal-like', 'Interferon', 'Immune']
    sub = top[0, 0].subgridspec(1, 4, wspace=0.06, width_ratios=[1, 1, 1, 0.06])
    for k, m in enumerate(['native', 'sr', 'bicubic']):
        a = fig.add_subplot(sub[0, k])
        z = A[A.method == m].pivot(index='feature', columns='program', values='rho').reindex(index=feats, columns=progs)
        im_ = a.imshow(z, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        a.set_xticks(range(7), pl, rotation=55, ha='right', rotation_mode='anchor', fontsize=6)
        a.set_yticks(range(7), fl if k == 0 else [], fontsize=6); a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_title({'native': 'Native', 'sr': 'SR (calibrated)', 'bicubic': 'Bicubic'}[m], fontsize=6.5, fontweight='normal', pad=2)
        if k == 0: tag(a, 'a')
    cax = fig.add_subplot(sub[0, 3]); cb = fig.colorbar(im_, cax=cax); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
    cb.ax.set_title('ρ', fontsize=6.5, pad=3)
    # b
    ax = fig.add_subplot(top[0, 1]); tag(ax, 'b')
    rp = pd.read_csv(R9 / 'association_recovery_2026-10-29/recovery_by_sample.csv').pivot(index='gsm', columns='method', values='matrix_element_pearson_r')
    rh = pd.read_csv(HB / 'association_recovery_by_sample.csv')
    sets = [('Real\nhires', rp.bicubic, rp.sr)]
    for cond, t in [('upper_area', 'HEST\nbox'), ('lower_aa0', 'HEST\nAA')]:
        z = rh[rh.condition == cond].pivot(index='sample', columns='method', values='matrix_element_pearson_r'); sets.append((t, z.bicubic, z.calibrated_SR))
    for k, (t, b, s) in enumerate(sets):
        paired(ax, 2.5 * k, 2.5 * k + 1, b, s, C['bicubic'], C['sr'], s=9)
        ax.text(2.5 * k + 0.5, 1.08, f'{(s > b).sum()}/{len(s)}', ha='center', fontsize=6)
    ax.set_xticks([0.5, 3, 5.5], [t for t, _, _ in sets]); ax.set_ylim(-0.5, 1.15); ax.set_xlim(-0.5, 6.5)
    ax.set_ylabel('Pearson r with native matrix'); ax.set_title('Matrix recovery', loc='left')
    # c
    ax = fig.add_subplot(top[0, 2]); tag(ax, 'c')
    pr = pd.read_csv(CT / 'prediction_cv_r2.csv'); pr = pr[pr.feature_set == 'morph+types']
    s = pr.groupby(['condition', 'sample', 'method']).cv_r2.mean().unstack()
    for k, cond in enumerate(['real_hires', 'upper_area', 'lower_aa0']):
        z = s.loc[cond]
        for i, m in enumerate(['native', 'sr', 'bicubic']):
            x = 3.6 * k + i
            ax.bar(x, z[m].mean(), color=C[m], width=0.8, alpha=0.9)
            ax.scatter(x + np.random.default_rng(i).uniform(-0.18, 0.18, len(z)), z[m], color='white', edgecolor='#222222', s=5, linewidth=0.4, zorder=3)
    ax.axhline(0, color='black', lw=0.5); ax.set_xticks([1, 4.6, 8.2], ['Real\nhires', 'HEST\nbox', 'HEST\nAA'])
    ax.set_ylabel('Cross-validated R² (mean of 8 programs)'); ax.set_ylim(-0.17, 0.58); ax.set_title('Expression prediction', loc='left')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[m]) for m in ['native', 'sr', 'bicubic']], labels=['Native', 'SR (calibrated)', 'Bicubic'],
              loc='upper left', fontsize=6, handlelength=0.8, ncol=1, labelspacing=0.2)
    # d
    bot = gs[1].subgridspec(1, 2, width_ratios=[0.9, 4], wspace=0.25)
    ax = fig.add_subplot(bot[0, 0]); tag(ax, 'd')
    cr = pd.read_csv(CT / 'composition_recovery.csv'); cr = cr[(cr.measure == 'n') & (cr.condition == 'real_hires')]
    for k, ct in enumerate(['connective', 'inflammatory', 'neoplastic']):
        z = cr[cr.cell_type == ct].pivot(index='sample', columns='method', values='rho')
        paired(ax, 2.5 * k, 2.5 * k + 1, z.bicubic, z.sr, C['bicubic'], C['sr'], s=9)
        ax.text(2.5 * k + 0.5, 1.0, f'{(z.sr > z.bicubic).sum()}/{len(z)}', ha='center', va='bottom', fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_xticks([0.5, 3, 5.5], ['Stromal', 'Immune', 'Tumour']); ax.set_xlim(-0.5, 6.5); ax.set_ylim(0, 0.72)
    ax.set_ylabel('Per-spot count ρ vs native'); ax.set_title('Cell-type counts', loc='left', pad=10)
    # e
    src = json.loads((PA / 'C/A_paired' / g / 'image_sources.json').read_text())
    pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('barcode')
    asc = pd.read_csv(PA / 'C/A_paired' / g / 'associations.csv')
    sub = bot[0, 1].subgridspec(1, 5, wspace=0.04, width_ratios=[1, 1, 1, 1, 0.05])
    panels = [('native', 'EMT', 'EMT program (native)'), ('native', 'eccentricity_median', 'Eccentricity, native'),
              ('sr', 'eccentricity_median', 'Eccentricity, SR'), ('bicubic', 'eccentricity_median', 'Eccentricity, bicubic')]
    for k, (m, col, t) in enumerate(panels):
        a = fig.add_subplot(sub[0, k])
        z = pd.read_csv(PA / 'C/A_paired' / g / f'spot_expression_{m}.csv').set_index('barcode')
        xy = pt.reindex(z.index); v = z[col].rank(pct=True)
        if col != 'EMT':
            pr_ = asc[(asc.method == m) & (asc.feature == col) & (asc.program == 'EMT')].partial_rho.iloc[0]; t += f'\npartial ρ = {pr_:+.2f}'
        else:
            t += '\n '
        P = np.c_[xy.fullres_x, xy.fullres_y]; ok = v.notna().to_numpy(); _, nb = cKDTree(P[ok]).query(P[ok], k=7)
        vs = pd.Series(v.to_numpy()[ok][nb].mean(1)).rank(pct=True).to_numpy()
        sc = a.scatter(P[ok, 0] * src['sf'], P[ok, 1] * src['sf'], c=vs, s=1.0, cmap='viridis', vmin=0, vmax=1, lw=0, rasterized=True)
        a.set_xlim(P[:, 0].min() * src['sf'] - 10, P[:, 0].max() * src['sf'] + 10); a.set_ylim(P[:, 1].max() * src['sf'] + 10, P[:, 1].min() * src['sf'] - 10)
        a.set_aspect('equal'); a.axis('off'); a.set_title(t, fontsize=6.5, fontweight='normal', pad=2)
        if k == 0: tag(a, 'e')
    cax = fig.add_subplot(sub[0, 4]); cb = fig.colorbar(sc, cax=cax, ticks=[0, 0.5, 1]); cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4); cb.set_label('Percentile (Donor A)', fontsize=6)
    save(fig, 'Fig5')


def supplementary():
    """Re-export Supplementary Figures S1–S7 with the same font style (content as sr11)."""
    import sr11_supplementary as S
    S.save = save
    for w in '1234567':
        getattr(S, f's{w}')()


if __name__ == '__main__':
    for w in sys.argv[1:] or ['1', '2', '3', '4', '5', 'S']:
        supplementary() if w == 'S' else globals()[f'fig{w}']()
