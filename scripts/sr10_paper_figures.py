"""Original main figures Fig 1–5 (shared helpers for sr11–sr14): read existing results, write results/sr_paper_figures_2026-11-01/."""
import os
try:
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
except PermissionError:
    pass
os.environ['OMP_NUM_THREADS'] = '2'
import sys, gzip, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
R = ROOT / 'results'
R9 = R / 'sr09'
OUT = R / 'sr_paper_figures_2026-11-01'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)

# colours: native reference dark grey; calibrated SR orange; SR08 blue; bicubic light grey
C = {'native': '#333333', 'sr': '#eb6834', 'old': '#2a78d6', 'bicubic': '#a8a8a8'}
LAB = {'native': 'Native', 'sr': 'SR (calibrated)', 'old': 'SR08 (simulation-trained)', 'bicubic': 'Bicubic'}
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
MM = 1 / 25.4
DONOR = {'GSM6505133': 'Donor A', 'GSM6505134': 'Donor B', 'GSM6505135': 'Donor C'}
S65 = R9 / 'scale65_rebuild_2026-10-27'
PA = R9 / 'pitch_ablation_2026-10-28'
HB = R9 / 'hest_bounds_2026-10-29'
CT = R9 / 'celltype_application_2026-11-01'


TAGS = []


def tag(ax, s, **_):
    TAGS.append((ax, s))


def place_tags(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, s in TAGS:
        if ax.figure is not fig: continue
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0 - 0.006, bb.y1 + 0.004, s, fontsize=9, fontweight='bold', ha='left', va='bottom')
    TAGS[:] = [t for t in TAGS if t[0].figure is not fig]


def img(p):
    return np.asarray(Image.open(p).convert('RGB'))


def hires(path):
    b = Path(path).read_bytes()
    b = gzip.decompress(b) if str(path).endswith('.gz') else b
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


def show(ax, im, title=None, nearest=False, um_px=None, bar_um=20):
    ax.imshow(im, interpolation='nearest' if nearest else 'antialiased')
    if um_px:  # white scale bar, bottom right
        h, w = im.shape[:2]; L = bar_um / um_px
        ax.plot([w * 0.9 - L, w * 0.9], [h * 0.92] * 2, color='white', lw=2, solid_capstyle='butt')
        ax.text(w * 0.9 - L / 2, h * 0.87, f'{bar_um} µm', color='white', fontsize=5, ha='center', va='bottom')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.4)
    if title: ax.set_title(title, fontsize=6.5, pad=2)


def paired(ax, x0, x1, a, b, ca, cb, lw=0.5):
    for u, v in zip(a, b): ax.plot([x0, x1], [u, v], color='#cccccc', lw=lw, zorder=1)
    ax.scatter([x0] * len(a), a, s=10, color=ca, zorder=3, edgecolor='white', linewidth=0.3)
    ax.scatter([x1] * len(b), b, s=10, color=cb, zorder=3, edgecolor='white', linewidth=0.3)


def save(fig, name):
    place_tags(fig)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print('saved', name, flush=True)


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.55, wspace=1.0)
    # a public-data survey
    ax = fig.add_subplot(gs[0, 0:2]); tag(ax, 'a')
    s = pd.read_csv(R / 'sr_survey/geo_visium_image_survey.csv').category.value_counts()
    cats = [('hires_only', 'hires PNG only'), ('bundle_only', 'image only in archive'), ('fullres', 'full-resolution image'),
            ('fullres_unc', 'full-res (unconfirmed)'), ('no_image', 'no image')]
    vals = [s.get(k, 0) for k, _ in cats]
    ax.barh(range(len(cats))[::-1], vals, color=['#eb6834'] + ['#bbbbbb'] * 4, height=0.65)
    for i, v in enumerate(vals): ax.text(v + 4, len(cats) - 1 - i, f'{v}', va='center', fontsize=6.5)
    ax.set_yticks(range(len(cats))[::-1], [l for _, l in cats]); ax.set_xlabel('GEO human Visium series')
    ax.set_title(f'Images in {s.sum()} GEO Visium series', loc='left')
    ax.set_xlim(0, max(vals) * 1.18)
    # b overview + inset
    g = 'GSM6505133'
    src = json.loads((PA / 'C/A_paired' / g / 'image_sources.json').read_text())
    H = hires(src['hires'])
    pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('patch_id')
    pid = 'p00093'
    ax = fig.add_subplot(gs[0, 2:4]); tag(ax, 'b', x=-0.12, y=1.02)
    cx, cy = pt.loc[pid, 'fullres_x'] * src['sf'], pt.loc[pid, 'fullres_y'] * src['sf']
    sat = cv2.cvtColor(H, cv2.COLOR_RGB2HSV)[..., 1]; ys, xs = np.where(cv2.medianBlur(sat, 5) > 40)
    ax.imshow(H); ax.set_xlim(xs.min() - 20, xs.max() + 20); ax.set_ylim(ys.max() + 20, ys.min() - 20)
    half = 64 / (65 / (295.381 * src["sf"]))  # 64 µm → hires px (Donor A spot_diameter_fullres = 295.381)
    ax.add_patch(Rectangle((cx - 4 * half, cy - 4 * half), 8 * half, 8 * half, fill=False, ec='#eb6834', lw=1.2))
    ax.set_xticks([]); ax.set_yticks([]); [sp_.set_visible(False) for sp_ in ax.spines.values()]; ax.set_title('Public hires, Donor A (4.3 µm/px)', loc='left')
    base = S65 / 'paired/samples' / g / 'patches'
    sub = gs[0, 4:6].subgridspec(2, 2, wspace=0.08, hspace=0.3)
    for k, (m, t, nn) in enumerate([('lr', 'hires (4 µm/px)', True), ('bicubic', 'bicubic ×16', False),
                                     ('sr', 'SR ×16', False), ('native', 'native scan', False)]):
        a = fig.add_subplot(sub[k // 2, k % 2]); show(a, img(base / m / f'{pid}.png'), t, nearest=nn, um_px=0.25 if m == 'native' else None)
        if k == 0: tag(a, 'c')
    # d pipeline schematic
    ax = fig.add_subplot(gs[1, 0:4]); tag(ax, 'd', x=-0.02, y=1.0); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    boxes = [(1, 'TCGA FFPE\nWSI\n0.25 µm/px', '#eeeeee'), (21, 'degrade\nto 4 µm/px\n(calibrated σ)', '#fde3d6'),
             (41, 'SwinIR ×8\n+ bicubic ×2', '#fde3d6'), (61, 'CellViT\nnuclei', '#eeeeee'),
             (81, '55 µm spot\nfeatures vs\nexpression', '#eeeeee')]
    for x, t, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 18), 17, 16, boxstyle='round,pad=0.4,rounding_size=1.5', fc=col, ec='#555555', lw=0.6))
        ax.text(x + 8.5, 26, t, ha='center', va='center', fontsize=5.8)
    for x in [18.5, 38.5, 58.5, 78.5]:
        ax.annotate('', (x + 2.3, 26), (x, 26), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#555555'))
    ax.text(1, 12, 'Training: TCGA only.   σ calibration: GSE327056 (4 pairs).\n'
               'Test: GSE211895 real hires (3 donors); HEST FFPE (9 slides, 6 cancers).\n'
               'Reference: CellViT on the native scan of the same section.',
            fontsize=5.8, va='top', color='#333333')
    # e scale pitfall
    ax = fig.add_subplot(gs[1, 4:6]); tag(ax, 'e')
    a = pd.read_csv(PA / 'scale_audit.csv')
    p55 = a.neighbor_pitch_fullres_px * 55 / a.spot_diameter_fullres
    p65 = a.neighbor_pitch_fullres_px * 65 / a.spot_diameter_fullres
    ok = a.gse.isin(['GSE211895', 'GSE327056'])
    rng = np.random.default_rng(0)
    for k, (v, col) in enumerate([(p55[ok], '#bbbbbb'), (p65[ok], '#eb6834')]):
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=col, edgecolor='white', linewidth=0.3)
    ax.axhline(100, color='#333333', lw=0.6, ls='--'); ax.text(-0.45, 100.8, 'design pitch 100 µm', fontsize=5.8)
    ax.set_xticks([0, 1], ['spot_diameter\n= 55 µm', 'spot_diameter\n= 65 µm']); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('neighbour spot pitch (µm)'); ax.set_ylim(80, 104)
    ax.set_title('spot_diameter_fullres\nspans 65 µm, not 55 µm', loc='left')
    save(fig, 'Fig1_problem_and_pipeline')


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 58 * MM), gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1.1], wspace=0.7))
    ax = axs[0]; tag(ax, 'a')
    for k, s in enumerate(['x2', 'x4', 'x8']):
        d = pd.read_csv(R / f'sr08/eval_{s}/metrics_by_sample.csv')
        d = d[d.testset.str.startswith('HEST') & (d.group == 'FFPE')]
        w = d.pivot_table(index='sample', columns='method', values='f1')
        paired(ax, 3 * k, 3 * k + 1, w.bicubic, w.SwinIR, C['bicubic'], C['sr'])
    ax.set_xticks([0.5, 3.5, 6.5], ['×2\n1 µm/px', '×4\n2 µm/px', '×8\n4 µm/px']); ax.set_ylabel('nucleus detection F1 vs native')
    ax.set_title('Detection F1, HEST FFPE', loc='left'); ax.set_ylim(0, 1)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=C['bicubic'], label='bicubic'),
                       Line2D([], [], marker='o', ls='', color=C['sr'], label='SwinIR')], frameon=False, loc='lower left')
    d = pd.read_csv(R / 'sr08/eval_x8/metrics_by_sample.csv')
    for ax, (col, lab, t) in zip(axs[1:3], [('abs_log_area_med', '|log area ratio| (matched nuclei)', 'b'),
                                             ('rho_density', 'tile density ρ vs native', 'c')]):
        tag(ax, t)
        for k, (ts, name) in enumerate([('HEST', 'HEST'), ('Xenium', 'Xenium\nDAPI')]):
            z = d[d.testset.str.startswith(ts)].pivot_table(index='sample', columns='method', values=col)
            paired(ax, 3 * k, 3 * k + 1, z.bicubic, z.SwinIR, C['bicubic'], C['sr'])
        ax.set_xticks([0.5, 3.5], ['HEST', 'Xenium\n(DAPI ref.)']); ax.set_ylabel(lab)
    axs[1].set_title('×8 size error', loc='left'); axs[2].set_title('×8 tile density', loc='left')
    ax = axs[3]; tag(ax, 'd')
    lc = pd.read_csv(R / 'sr08/annotation_x8/analysis_review_2026-10-25/label_counts.csv')
    lc = lc[(lc.cat == 'unmatched') & lc.stratum.isin(['primary', 'generalization'])].groupby(['method', 'final_label']).n.sum().unstack().fillna(0)
    bottom = np.zeros(2); order = ['bicubic', 'SwinIR']
    for lab_, col in [('yes', '#555555'), ('uncertain', '#cfcfcf'), ('no', '#d62728')]:
        v = lc.reindex(order)[lab_].to_numpy() if lab_ in lc else np.zeros(2)
        ax.bar([0, 1], v, bottom=bottom, color=col, width=0.6, label={'yes': 'nucleus', 'uncertain': 'uncertain', 'no': 'not a nucleus'}[lab_])
        bottom += v
    for i, o in enumerate(order):
        r = lc.loc[o]; ax.text(i, bottom[i] + 4, f"{int(r.get('no', 0))}/{int(r.sum())}\nnot nuclei", ha='center', fontsize=6)
    ax.set_xticks([0, 1], ['bicubic', 'SwinIR']); ax.set_ylabel('blinded expert labels (n)'); ax.set_ylim(0, bottom.max() * 1.3)
    ax.set_title('Expert review of\nunmatched ×8 detections', loc='left'); ax.legend(frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, handlelength=1, columnspacing=0.6)
    save(fig, 'Fig2_simulated_validation')


# ---------------------------------------------------------------- Fig 3
def lapvar(lr):
    g = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return cv2.Laplacian(g, cv2.CV_32F).var()


def fig3():
    import importlib
    v2 = importlib.import_module('sr02_degrade_benchmark').degrade_real2
    cal = importlib.import_module('sr09_scale65_degrade').degrade_real2
    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.7, height_ratios=[1, 1])
    # a uncalibrated pipeline: works on simulation, fails on real hires
    ax = fig.add_subplot(gs[0, 0]); tag(ax, 'a')
    L = pd.read_csv(R9 / 'B_sim_vs_real_2026-10-27_legacy_scale55/comparison_side_by_side.csv')
    L = L[L.feature == 'density']
    for k, m in enumerate(['bicubic', 'sr']):
        z = L[L.method == m]
        paired(ax, 3 * k, 3 * k + 1, z['rho__simulated_real2_4.0'], z['rho__true_hires'], C[m], C[m])
    ax.set_xticks([0, 1, 3, 4], ['sim.', 'real', 'sim.', 'real']); ax.set_ylim(0, 1)
    ax.text(0.5, -0.27, 'bicubic', transform=ax.get_xaxis_transform(), ha='center', fontsize=6.5)
    ax.text(3.5, -0.27, 'SR08', transform=ax.get_xaxis_transform(), ha='center', fontsize=6.5)
    ax.set_ylabel('spot nuclear density ρ vs native'); ax.set_title('Naive pipeline, same slides\n(55 µm scale, uncalibrated blur)', loc='left')
    # b images: native / real / v2 simulation / calibrated simulation
    g, pid = 'GSM6505135', 'p00245'
    nat = img(S65 / 'paired/samples' / g / 'patches/native' / f'{pid}.png')
    real = img(S65 / 'paired/samples' / g / 'patches/lr' / f'{pid}.png')
    gt = cv2.resize(nat, (256, 256), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(7)
    lv2, _ = v2(gt, 32, rng); lcal, _ = cal(gt, 32, np.random.default_rng(7))
    sub = gs[0, 1:3].subgridspec(1, 4, wspace=0.08)
    for k, (im_, t) in enumerate([(nat, 'native'), (real, 'real hires'), (lv2, 'sim. orig.'), (lcal, 'sim. calib.')]):
        a = fig.add_subplot(sub[0, k]); show(a, im_, t, nearest=k > 0, um_px=0.25 if k == 0 else None)
        if k == 0: tag(a, 'b')
    # c sharpness: simulated / real Laplacian-variance ratio
    ax = fig.add_subplot(gs[0, 3]); tag(ax, 'c')
    rows = []
    for g2 in DONOR:
        pt = pd.read_csv(S65 / 'paired/samples' / g2 / 'patches.csv')
        for i, pid2 in enumerate(pt.patch_id.sample(120, random_state=1)):
            n2 = img(S65 / 'paired/samples' / g2 / 'patches/native' / f'{pid2}.png'); r2 = img(S65 / 'paired/samples' / g2 / 'patches/lr' / f'{pid2}.png')
            if (n2.mean(2) > 215).mean() > 0.5: continue
            gt2 = cv2.resize(n2, (256, 256), interpolation=cv2.INTER_AREA)
            a2, _ = v2(gt2, 32, np.random.default_rng(i)); b2, _ = cal(gt2, 32, np.random.default_rng(i))
            rows.append(dict(donor=DONOR[g2], v2=np.log2(lapvar(a2) / lapvar(r2)), cal=np.log2(lapvar(b2) / lapvar(r2))))
    lv = pd.DataFrame(rows); lv.to_csv(OUT / 'fig3c_laplacian_ratio.csv', index=False)
    bp = ax.boxplot([lv.v2, lv.cal], positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True)
    for b_, col in zip(bp['boxes'], ['#bbbbbb', '#eb6834']): b_.set_facecolor(col); b_.set_linewidth(0.5)
    for k in ['medians', 'whiskers', 'caps']: [l.set_linewidth(0.6) for l in bp[k]]
    ax.axhline(0, color='#333333', lw=0.6, ls='--'); ax.set_xticks([0, 1], ['orig.\n0.3–1.2', 'calib.\n1.5–1.75'])
    ax.set_ylabel('log2 sharpness ratio,\nsimulated / real'); ax.set_title(f'Laplacian variance, same\nfields (n={len(lv)}, 3 donors)', loc='left')
    # d calibration curves
    ax = fig.add_subplot(gs[1, 0]); tag(ax, 'd')
    cc = pd.read_csv(S65 / 'calibration/calibration_sample_curves.csv')
    for gg, z in cc.groupby('gsm'): ax.plot(z.sigma_um, z.loss, color='#555555', lw=0.8, marker='o', ms=2)
    ax.axvspan(0.3, 1.2, color='#dddddd', lw=0); ax.axvspan(1.5, 1.75, color='#eb6834', alpha=0.35, lw=0)
    ax.set_xlim(0, 6.5); ax.set_ylim(0, 5); ax.set_xlabel('Gaussian blur σ (µm)'); ax.set_ylabel('mismatch to real hires')
    ax.set_title('Blur calibration\n(GSE327056, 4 pairs)', loc='left')
    ax.text(0.75, 4.8, 'original\nrange', fontsize=5.8, ha='center', va='top', color='#666666'); ax.text(1.85, 4.8, 'calibrated', fontsize=5.8, va='top', color='#c0461a')
    # e ablation: BC / SR08 / calibrated SR (correct scale, real hires)
    ab = pd.read_csv(PA / 'model_ablation_long.csv')
    for k, (feat, t) in enumerate([('density', 'nuclear density'), ('area_median', 'median nuclear area'), ('area_cv', 'nuclear area CV')]):
        ax = fig.add_subplot(gs[1, k + 1]); tag(ax, 'efg'[k])
        z = ab[ab.feature == feat]
        def pick(meth):
            q = z[z.method.str.contains(meth)]
            return q.groupby('gsm').rho.mean()
        ms = [('bicubic', 'bicubic'), ('old', 'old'), ('sr', 'sr')]
        w = pd.DataFrame({'bicubic': z[z.method == 'bicubic'].groupby('gsm').rho.mean(),
                          'old': z[z.method.str.contains('old', case=False)].groupby('gsm').rho.mean(),
                          'sr': z[(z.method != 'bicubic') & ~z.method.str.contains('old', case=False)].groupby('gsm').rho.mean()})
        for g3, r in w.iterrows(): ax.plot([0, 1, 2], r[['bicubic', 'old', 'sr']], color='#cccccc', lw=0.6, zorder=1)
        for i, m in enumerate(['bicubic', 'old', 'sr']):
            ax.scatter([i] * len(w), w[m], color=C[m], s=12, zorder=3, edgecolor='white', linewidth=0.3)
            ax.text(i, w[m].mean(), f' {w[m].mean():.2f}', fontsize=5.8, va='center', ha='left', color='#333333')
        ax.set_xticks([0, 1, 2], ['bicubic', 'SR08', 'SR\ncalibrated']); ax.set_xlim(-0.4, 2.6)
        ax.set_title(f'Real hires, fixed scale:\n{t}', loc='left'); ax.set_ylabel('ρ vs native')
    save(fig, 'Fig3_real_hires_failure_and_fix')


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig = plt.figure(figsize=(180 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 4, hspace=0.95, wspace=0.75)
    # a HEST examples
    sid = 'TENX65'
    pt = pd.read_csv(HB / 'reference/samples' / sid / 'patches.csv')
    pid = pt[pt.image_eligible].patch_id.iloc[len(pt) // 3]
    sub = gs[0, 0:2].subgridspec(1, 4, wspace=0.06)
    tiles = [(HB / 'reference/samples' / sid / 'patches/native' / f'{pid}.png', 'native', False),
             (HB / 'upper_area/samples' / sid / 'patches/lr' / f'{pid}.png', 'box\n(sharper)', True),
             (HB / 'lower_aa0/samples' / sid / 'patches/lr' / f'{pid}.png', 'anti-aliased\n(blurrier)', True),
             (HB / 'lower_aa0/samples' / sid / 'patches/sr' / f'{pid}.png', 'SR of\nanti-aliased', False)]
    for k, (p, t, nn) in enumerate(tiles):
        a = fig.add_subplot(sub[0, k]); show(a, img(p), t, nearest=nn, um_px=0.25 if k == 0 else None)
        if k == 0: tag(a, 'a'); a.set_ylabel(f'{sid} (HGSOC)', fontsize=6)
    # b the two generators bracket real hires
    ax = fig.add_subplot(gs[0, 2]); tag(ax, 'b')
    mc = pd.read_csv(R9 / 'hires_reproduction_2026-10-29/method_comparison_all.csv')
    rng = np.random.default_rng(0)
    for k, (m, t, col) in enumerate([('opencv_area', 'box', '#555555'), ('skimage_resize_rescale_aa_order0', 'anti-aliased', '#eb6834')]):
        v = np.log2(mc[mc.method == m].laplacian_variance_ratio)
        ax.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=12, color=col, edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='#333333', ls='--', lw=0.6); ax.set_xticks([0, 1], ['box', 'anti-\naliased']); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('log2 sharpness,\ngenerated / real hires'); ax.set_title('Generators bracket real\nhires (7 pairs)', loc='left')
    # c PSNR
    ax = fig.add_subplot(gs[0, 3]); tag(ax, 'c')
    for k, m in enumerate(['opencv_area', 'skimage_resize_rescale_aa_order0']):
        v = mc[mc.method == m].psnr_db; ax.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=12, color=['#555555', '#eb6834'][k], edgecolor='white', linewidth=0.3)
    ax.set_xticks([0, 1], ['box', 'anti-\naliased']); ax.set_xlim(-0.5, 1.5); ax.set_ylabel('PSNR vs real hires (dB)')
    ax.set_title('Pixel agreement', loc='left')
    # d–f nine HEST slides
    sm = pd.read_csv(HB / 'spot_metrics_by_sample.csv')
    cancer = sm.drop_duplicates('sample').set_index('sample').cancer
    order = sm[(sm.feature == 'density') & (sm.condition == 'upper_area') & (sm.method == 'calibrated_SR')].sort_values('rho')['sample'].tolist()
    for k, (feat, t) in enumerate([('density', 'nuclear density'), ('area_median', 'median nuclear area')]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2] if k == 1 else gs[1, 0:2]); tag(ax, 'de'[k], x=-0.12)
        for j, (cond, dx) in enumerate([('upper_area', -0.18), ('lower_aa0', 0.18)]):
            z = sm[(sm.feature == feat) & (sm.condition == cond)].pivot(index='sample', columns='method', values='rho').reindex(order)
            x = np.arange(len(order)) + dx
            for i in range(len(order)): ax.plot([x[i]] * 2, [z.bicubic.iloc[i], z.calibrated_SR.iloc[i]], color='#cccccc', lw=0.6, zorder=1)
            mk = 'o' if cond == 'upper_area' else 's'
            ax.scatter(x, z.bicubic, color=C['bicubic'], s=10, marker=mk, zorder=2)
            ax.scatter(x, z.old_SR08, color=C['old'], s=10, marker=mk, zorder=2)
            ax.scatter(x, z.calibrated_SR, color=C['sr'], s=12, marker=mk, zorder=3, edgecolor='white', linewidth=0.3)
            win = (z.calibrated_SR > z.bicubic).sum()
            wins = getattr(ax, '_wins', []); wins.append(f"{'box' if j == 0 else 'anti-aliased'} {win}/9"); ax._wins = wins
        ax.set_xticks(range(len(order)), [f'{cancer[s]}\n{s}' for s in order], fontsize=5.2, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel('ρ vs native'); ax.set_title(f'HEST FFPE: {t}\nSR > bicubic: ' + ', '.join(ax._wins), loc='left')
        if k == 0:
            fig.legend(handles=[Line2D([], [], marker='o', ls='', color=C[m], label=LAB[m]) for m in ['bicubic', 'old', 'sr']] +
                      [Line2D([], [], marker='o', ls='', color='#777777', label='box (circle)'), Line2D([], [], marker='s', ls='', color='#777777', label='anti-aliased (square)')],
                      frameon=False, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=5)
    save(fig, 'Fig4_external_validation_HEST')


# ---------------------------------------------------------------- Fig 5
def fig5():
    fig = plt.figure(figsize=(180 * MM, 130 * MM))
    gs = fig.add_gridspec(2, 4, hspace=0.7, wspace=1.2, height_ratios=[1, 1.1], width_ratios=[1, 1.15, 1, 1])
    # a example association matrices
    g = 'GSM6505133'
    A = pd.read_csv(PA / 'C/A_paired' / g / 'associations.csv')
    feats = ['density', 'area_median', 'area_cv', 'eccentricity_median', 'h_mean', 'h_std', 'gray_lapvar']
    progs = ['EMT', 'hypoxia', 'proliferation', 'classical', 'basal', 'interferon', 'immune']
    sub = gs[0, 0:2].subgridspec(1, 4, wspace=0.08, width_ratios=[1, 1, 1, 0.07])
    for k, m in enumerate(['native', 'sr', 'bicubic']):
        a = fig.add_subplot(sub[0, k])
        z = A[A.method == m].pivot(index='feature', columns='program', values='rho').reindex(index=feats, columns=progs)
        im_ = a.imshow(z, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        a.set_xticks(range(7), progs, rotation=70, fontsize=5.3); a.set_yticks(range(7), [f.replace('_median', '').replace('gray_', '') for f in feats] if k == 0 else [], fontsize=5.3)
        a.set_title({'native': 'Native', 'sr': 'SR', 'bicubic': 'Bicubic'}[m], fontsize=6.5, pad=2)
        if k == 0: tag(a, 'a')
    cb = fig.colorbar(im_, cax=fig.add_subplot(sub[0, 3])); cb.ax.tick_params(labelsize=5.5); cb.ax.set_title('ρ', fontsize=6)
    # b matrix recovery
    ax = fig.add_subplot(gs[0, 2]); tag(ax, 'b')
    rp = pd.read_csv(R9 / 'association_recovery_2026-10-29/recovery_by_sample.csv').pivot(index='gsm', columns='method', values='matrix_element_pearson_r')
    rh = pd.read_csv(HB / 'association_recovery_by_sample.csv')
    sets = [('real\nhires', rp.bicubic, rp.sr)]
    for cond, t in [('upper_area', 'HEST\nbox'), ('lower_aa0', 'HEST\nAA')]:
        z = rh[rh.condition == cond].pivot(index='sample', columns='method', values='matrix_element_pearson_r'); sets.append((t, z.bicubic, z.calibrated_SR))
    for k, (t, b, s) in enumerate(sets):
        paired(ax, 3 * k, 3 * k + 1, b, s, C['bicubic'], C['sr'])
        ax.text(3 * k + 0.5, 1.03, f'{(s > b).sum()}/{len(s)}', ha='center', fontsize=6)
    ax.set_xticks([0.5, 3.5, 6.5], [t for t, _, _ in sets], fontsize=5.8); ax.set_ylim(-0.5, 1.1)
    ax.set_ylabel('r vs native matrix'); ax.set_title('Morphology–expression\nmatrix recovery', loc='left')
    # c prediction R²
    ax = fig.add_subplot(gs[0, 3]); tag(ax, 'c')
    pr = pd.read_csv(CT / 'prediction_cv_r2.csv'); pr = pr[pr.feature_set == 'morph+types']
    s = pr.groupby(['condition', 'sample', 'method']).cv_r2.mean().unstack()
    for k, (cond, t) in enumerate([('real_hires', 'real'), ('upper_area', 'HEST box'), ('lower_aa0', 'HEST AA')]):
        z = s.loc[cond]
        for i, m in enumerate(['native', 'sr', 'bicubic']):
            x = 4 * k + i
            ax.bar(x, z[m].mean(), color=C[m], width=0.8)
            ax.scatter(x + np.random.default_rng(i).uniform(-0.2, 0.2, len(z)), z[m], color='black', s=3, zorder=3)
    ax.axhline(0, color='black', lw=0.5); ax.set_xticks([1, 5, 9], ['real\nhires', 'HEST\nbox', 'HEST\nAA'], fontsize=5.8)
    ax.set_ylabel('CV R² (mean, 8 programs)'); ax.set_ylim(-0.17, 0.42); ax.set_title('Predicting expression\nfrom image features', loc='left')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[m]) for m in ['native', 'sr', 'bicubic']], labels=['native', 'SR', 'bicubic'], frameon=False, fontsize=5.6, loc='upper right', handlelength=0.8, borderaxespad=0)
    # d cell-count recovery (connective / inflammatory)
    ax = fig.add_subplot(gs[1, 0]); tag(ax, 'd')
    cr = pd.read_csv(CT / 'composition_recovery.csv'); cr = cr[(cr.measure == 'n') & (cr.condition == 'real_hires')]
    for k, ct in enumerate(['connective', 'inflammatory', 'neoplastic']):
        z = cr[cr.cell_type == ct].pivot(index='sample', columns='method', values='rho')
        paired(ax, 3 * k, 3 * k + 1, z.bicubic, z.sr, C['bicubic'], C['sr'])
        ax.text(3 * k + 0.5, 0.97, f'{(z.sr > z.bicubic).sum()}/{len(z)}', ha='center', fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_xticks([0.5, 3.5, 6.5], ['stromal', 'immune', 'tumour'], fontsize=5.8, rotation=30, ha='right', rotation_mode='anchor'); ax.set_ylabel('per-spot count ρ vs native')
    ax.set_title('CellViT cell-type counts,\nreal hires (n=7)', loc='left')
    # e eccentricity–EMT maps
    g = 'GSM6505133'
    src = json.loads((PA / 'C/A_paired' / g / 'image_sources.json').read_text()); H = hires(src['hires'])
    pt = pd.read_csv(S65 / 'paired/samples' / g / 'patches.csv').set_index('barcode')
    asc = pd.read_csv(PA / 'C/A_paired' / g / 'associations.csv')
    sub = gs[1, 1:4].subgridspec(1, 4, wspace=0.05)
    panels = [('native', 'EMT', 'EMT program'), ('native', 'eccentricity_median', 'eccentricity: native'),
              ('sr', 'eccentricity_median', 'eccentricity: SR'), ('bicubic', 'eccentricity_median', 'eccentricity: bicubic')]
    for k, (m, col, t) in enumerate(panels):
        a = fig.add_subplot(sub[0, k])
        z = pd.read_csv(PA / 'C/A_paired' / g / f'spot_expression_{m}.csv').set_index('barcode')
        xy = pt.reindex(z.index)
        v = z[col]
        if col != 'EMT':
            v = v.rank(pct=True)
            pr_ = asc[(asc.method == m) & (asc.feature == col) & (asc.program == 'EMT')].partial_rho.iloc[0]; t += f'\npartial ρ = {pr_:+.2f}'
        else:
            v = v.rank(pct=True)
        # display only: average of each spot and its 6 nearest neighbours (statistics unsmoothed)
        from scipy.spatial import cKDTree
        P = np.c_[xy.fullres_x, xy.fullres_y]; ok = v.notna().to_numpy(); tr = cKDTree(P[ok]); _, nb = tr.query(P[ok], k=7)
        vs = v.to_numpy()[ok][nb].mean(1); vs = pd.Series(vs).rank(pct=True).to_numpy()
        a.scatter(P[ok, 0] * src['sf'], P[ok, 1] * src['sf'], c=vs, s=1.1, cmap='viridis', vmin=0, vmax=1, lw=0)
        a.set_xlim(P[:, 0].min() * src['sf'] - 10, P[:, 0].max() * src['sf'] + 10); a.set_ylim(P[:, 1].max() * src['sf'] + 10, P[:, 1].min() * src['sf'] - 10); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([]); [sp_.set_visible(False) for sp_ in a.spines.values()]; a.set_title(t, fontsize=6, pad=2)
        if k == 0: tag(a, 'e'); a.set_ylabel('Donor A (percentile, 7-NN smoothed)', fontsize=5.5)
    save(fig, 'Fig5_transcriptome_linked_information')


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5']
    for w in which:
        globals()[f'fig{w}']()
