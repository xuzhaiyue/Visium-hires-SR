"""Supplementary Figures S1–S7 and Supplementary Table S1: read existing results, write results/sr_paper_figures_2026-11-01/supplementary/.
Uses the colours, font sizes and tag helpers of sr10_paper_figures. Usage: python scripts/sr11_supplementary.py [1 2 ... 7 T]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr10_paper_figures as F
from sr10_paper_figures import C, LAB, MM, DONOR, S65, PA, HB, CT, R, R9, tag, show, paired, img

OUT = F.OUT / 'supplementary'
OUT.mkdir(exist_ok=True)
F.OUT = OUT  # F.save writes to the supplementary directory
save = F.save
CAL = S65 / 'calibration'
HR = R9 / 'hires_reproduction_2026-10-29'
B55 = R9 / 'B_sim_vs_real_2026-10-27_legacy_scale55'
S8 = R / 'sr08'
GSE = {'GSM6505133': 'GSE211895', 'GSM6505134': 'GSE211895', 'GSM6505135': 'GSE211895',
       'GSM9647219': 'GSE327056', 'GSM9647220': 'GSE327056', 'GSM9647221': 'GSE327056', 'GSM9647222': 'GSE327056'}
CAL_GSM = ['GSM9647219', 'GSM9647220', 'GSM9647221', 'GSM9647222']
CAL_COL = ['#1b9e77', '#7570b3', '#e7298a', '#66a61e']
TYPES = ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']
TCOL = {1: '#e8112d', 2: '#00c000', 3: '#1e90ff', 4: '#000000', 5: '#ff9f00'}
PROGS = ['EMT', 'hypoxia', 'proliferation', 'classical', 'basal', 'interferon', 'immune', 'epithelial']


def mosaic_contours(mosaic, pids):
    """Nuclear contours of a patch from the mosaic layout (patch-local coordinates, 0.25 µm/px)."""
    cells = json.loads(Path(json.loads((mosaic / 'cellvit_complete.json').read_text())['cells_path']).read_text())['cells']
    layout = pd.read_csv(mosaic / 'layout.csv'); ncol = int(np.ceil(np.sqrt(len(layout))))
    pos = {p: i for i, p in enumerate(layout.patch_id)}
    want = {pos[p]: p for p in pids}
    out = {p: [] for p in pids}
    for c in cells:
        x, y = c['centroid']; col = int((x - 128) // 640); row = int((y - 128) // 640); idx = row * ncol + col
        if idx in want:
            out[want[idx]].append((np.asarray(c['contour'], float) - [128 + col * 640, 128 + row * 640], c['type']))
    del cells
    return out


# ---------------------------------------------------------------- S1 blur calibration
def s1():
    curves = pd.read_csv(CAL / 'calibration_sample_curves.csv')
    opt = pd.read_csv(CAL / 'calibration_patch_optima.csv')
    cfg = json.loads((CAL / 'calibration.json').read_text()); lo, hi = cfg['sigma_um_range']
    fig, axs = plt.subplots(1, 4, figsize=(180 * MM, 52 * MM), gridspec_kw={'wspace': 0.62, 'width_ratios': [1.1, 1, 1, 0.9]})
    a = axs[0]
    a.axvspan(lo, hi, color=C['sr'], alpha=0.18, lw=0)
    a.axvspan(0.3, 1.2, color=C['old'], alpha=0.10, lw=0)
    for g, col in zip(CAL_GSM, CAL_COL):
        d = curves[curves.gsm == g].sort_values('sigma_um')
        a.plot(d.sigma_um, d.loss, '-o', ms=2, lw=0.9, color=col, label=g)
    a.set_xscale('log'); a.set_xticks([0.3, 1, 3, 10, 24]); a.set_xticklabels(['0.3', '1', '3', '10', '24'])
    a.set_xlabel('Gaussian σ (µm)'); a.set_ylabel('Mismatch to real hires (mean)')
    a.text(0.75, a.get_ylim()[1] * 0.97, 'SR08\nrange', color=C['old'], fontsize=5.5, ha='center', va='top')
    a.text(1.62, a.get_ylim()[1] * 0.97, 'calibrated', color=C['sr'], fontsize=5.5, ha='left', va='top')
    a.legend(frameon=False, fontsize=5.5, loc='lower left', bbox_to_anchor=(0.0, 0.42), handlelength=1.2); a.set_title('Loss curves,\nfour calibration pairs', loc='left'); tag(a, 'a')

    a = axs[1]
    grid = cfg['grid_um']; cnt = opt.sigma_um.value_counts().reindex(grid, fill_value=0)
    a.bar(range(len(grid)), cnt.values, color=[C['sr'] if lo <= s <= hi else '#bbbbbb' for s in grid], width=0.8)
    a.set_xticks(range(len(grid))); a.set_xticklabels([f'{s:g}' for s in grid], rotation=90, fontsize=5.5)
    a.set_xlim(-0.6, 11.6)
    a.set_xlabel('Per-field optimal σ (µm)'); a.set_ylabel('Fields')
    q = np.percentile(opt.sigma_um, [10, 50, 90])
    a.set_title(f'Per-field optima (n = {len(opt)})\npct 10/50/90: {q[0]:g}/{q[1]:g}/{q[2]:g} µm', loc='left'); tag(a, 'b')

    a = axs[2]
    for g, col in zip(CAL_GSM, CAL_COL):
        d = opt[opt.gsm == g]
        a.scatter(d.lapvar_real, d.lapvar_sim, s=5, color=col, lw=0, alpha=0.85)
    v = np.r_[opt.lapvar_real, opt.lapvar_sim]; lims = [v.min() * 0.8, v.max() * 1.25]
    a.plot(lims, lims, color='#888888', lw=0.6, ls='--'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lims); a.set_ylim(lims)
    r = np.corrcoef(np.log(opt.lapvar_real), np.log(opt.lapvar_sim))[0, 1]
    a.set_xlabel('Laplacian variance, real hires'); a.set_ylabel('Laplacian variance, simulated\n(at per-field optimum)')
    a.set_title(f'Sharpness at optimum\nPearson r (log) = {r:.2f}', loc='left'); tag(a, 'c')

    # d held-out check: simulated / real hires Laplacian sharpness ratio on GSE211895
    a = axs[3]
    lv = pd.read_csv(R / 'sr_paper_figures_2026-11-01/fig3c_laplacian_ratio.csv'); rng = np.random.default_rng(0)
    for k, (col, cc_) in enumerate([('v2', C['old']), ('cal', C['sr'])]):
        v = lv[col].to_numpy(); a.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=2, color=cc_, alpha=0.45, lw=0)
        q = np.percentile(v, [25, 50, 75]); a.plot([k, k], [q[0], q[2]], color='black', lw=1.3); a.scatter([k], [q[1]], color='white', edgecolor='black', s=12, zorder=4, linewidth=0.8)
        a.text(k, 2.25, f'{2 ** q[1]:.2f}×', ha='center', fontsize=6)
    a.axhline(0, color='#333333', lw=0.6, ls='--'); a.set_xticks([0, 1], ['SR08', 'Calibrated']); a.set_xlim(-0.6, 1.6); a.set_ylim(-0.7, 2.45)
    a.set_ylabel('log$_2$ sharpness, simulated / real'); a.set_title(f'Held-out check\nGSE211895, {len(lv)} fields', loc='left'); tag(a, 'd')
    save(fig, 'FigS1_blur_calibration')


# ---------------------------------------------------------------- S2 TCGA QC and training
def s2():
    qc = pd.read_csv(S8 / 'qc/tcga_qc_per_slide.csv')
    tiles = pd.read_csv(S8 / 'data/tiles.csv', usecols=['file', 'patient', 'site', 'split'])
    sp = pd.read_csv(S8 / 'data/split_sites.csv')
    used = set(tiles.file)
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 55 * MM), gridspec_kw={'wspace': 0.45, 'width_ratios': [1.1, 1, 1.2]})
    a = axs[0]
    for k, (q, col) in enumerate([(30, '#bbbbbb'), (70, C['native'])]):
        d = qc[qc.Q == q]
        x = k + np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
        u = d.file.isin(used).to_numpy()
        a.scatter(x[~u], d.sharpness[~u], s=4, color=col, lw=0, alpha=0.6)
        a.scatter(x[u], d.sharpness[u], s=6, color=C['sr'], lw=0)
        a.plot([k - 0.28, k + 0.28], [d.sharpness.median()] * 2, color='black', lw=1)
    a.set_xticks([0, 1]); a.set_xticklabels([f'JPEG Q30\n(n = {(qc.Q == 30).sum()})', f'JPEG Q70\n(n = {(qc.Q == 70).sum()})'])
    a.set_ylabel('Sharpness at 0.5 µm\n(Laplacian var / luma var)'); a.set_xlim(-0.5, 1.5)
    a.legend([Line2D([0], [0], marker='o', ls='', color=C['sr'], ms=3)], [f'used for training ({len(used)} slides)'], frameon=False, loc='upper left', fontsize=6)
    a.set_ylim(-0.1, 2.65); a.set_yticks([0, 0.5, 1, 1.5, 2])
    a.set_title('TCGA-PAAD compression quality', loc='left'); tag(a, 'a')

    a = axs[1]
    sp = sp.sort_values('slides', ascending=True)
    y = np.arange(len(sp)); left = np.zeros(len(sp))
    for k, col in [('train', C['native']), ('val', '#8c8c8c'), ('test', C['sr'])]:
        a.barh(y, sp[k], left=left, color=col, height=0.7, label=k); left += sp[k].to_numpy()
    a.set_yticks(y); a.set_yticklabels(sp.site); a.set_xlabel('Slides')
    npat = tiles.groupby('split').patient.nunique()
    a.legend(frameon=False, fontsize=6, loc='lower right', title=f"patients {npat['train']}/{npat['val']}/{npat['test']}", title_fontsize=6)
    a.set_title('Patient split by tissue source site', loc='left'); tag(a, 'b')

    a = axs[2]
    for name, path, col in [('SR08', S8 / 'models/x8/2026-10-24_run01/metrics.csv', C['old']),
                            ('calibrated SR', S65 / 'training/metrics.csv', C['sr'])]:
        m = pd.read_csv(path)
        a.plot(m.step / 1000, m.val_l1, '-o', ms=2, lw=1, color=col, label=f'{name}, validation')
        t = m.dropna(subset=['train_l1']); a.plot(t.step / 1000, t.train_l1, '--', lw=0.8, color=col, alpha=0.7, label=f'{name}, training')
    a.set_xlabel('Iteration (×1,000)'); a.set_ylabel('L1 loss'); a.legend(frameon=False, fontsize=6)
    a.set_title('Training curves, SwinIR-M ×8', loc='left'); tag(a, 'c')
    save(fig, 'FigS2_tcga_qc_training')


# ---------------------------------------------------------------- S3 thirteen downsamplers
def s3():
    d = pd.read_csv(HR / 'method_comparison_all.csv')
    order = d.groupby('method').psnr_db.median().sort_values().index
    hl = {'opencv_area': C['sr'], 'skimage_resize_rescale_aa_order0': C['old']}
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), sharey=True, gridspec_kw={'wspace': 0.08})
    for a, col, xl, logx in [(axs[0], 'psnr_db', 'PSNR vs real hires (dB)', False),
                             (axs[1], 'laplacian_variance_ratio', 'Laplacian variance ratio (generated / real)', True)]:
        for i, m in enumerate(order):
            v = d[d.method == m][col].to_numpy(); c = hl.get(m, '#999999')
            a.scatter(v, i + np.linspace(-0.2, 0.2, len(v)), s=6, color=c, lw=0, zorder=3)
            a.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color='black', lw=0.9, zorder=4)
        if logx:
            a.set_xscale('log'); a.axvline(1, color='#666666', lw=0.6, ls='--')
            a.set_xticks([0.5, 1, 2, 4, 8]); a.set_xticklabels(['0.5', '1', '2', '4', '8']); a.xaxis.set_minor_formatter(plt.NullFormatter())
        a.set_xlabel(xl); a.grid(axis='y', color='#eeeeee', lw=0.5)
    nice = {'opencv_area': 'OpenCV area', 'opencv_linear': 'OpenCV linear', 'opencv_cubic': 'OpenCV cubic', 'pil_box': 'PIL box',
            'pil_lanczos': 'PIL Lanczos', 'pil_bicubic': 'PIL bicubic', 'skimage_resize_rescale_aa_order0': 'scikit-image AA, order 0',
            'skimage_resize_rescale_aa_order1': 'scikit-image AA, order 1', 'skimage_resize_rescale_aa_order3': 'scikit-image AA, order 3',
            'gaussian_then_linear': 'Gaussian, then linear', 'opencv_area_fixed_sf': 'OpenCV area, fixed factor',
            'opencv_linear_fixed_sf': 'OpenCV linear, fixed factor', 'opencv_cubic_fixed_sf': 'OpenCV cubic, fixed factor'}
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([nice.get(m, m) for m in order], fontsize=6)
    axs[0].set_title('Fidelity to real hires (7 pairs; bar = median)', loc='left'); tag(axs[0], 'a')
    axs[1].set_title('Sharpness relative to real hires', loc='left'); tag(axs[1], 'b')
    axs[1].legend([Line2D([0], [0], marker='o', ls='', color=c, ms=3) for c in hl.values()],
                  ['OpenCV area ("box", upper bracket)', 'scikit-image AA, order 0 (lower bracket)'], frameon=False, fontsize=6, loc='upper right')
    save(fig, 'FigS3_downsamplers')


# ---------------------------------------------------------------- S4 Donor C native segmentation
def s4():
    sel = pd.read_csv(S65 / 'pipeline_figure/selected_spots.csv')
    from sr13_paper_figures_pub import micro
    fig = plt.figure(figsize=(180 * MM, 108 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.38, wspace=0.12, bottom=0.1)
    top = gs[0, :].subgridspec(1, 2, wspace=0.3)
    a = fig.add_subplot(top[0, 0])
    bins = np.logspace(np.log10(1), np.log10(300), 70)
    for g, col in zip(DONOR, ['#1b9e77', '#7570b3', C['sr']]):
        n = pd.read_csv(S65 / 'paired/samples' / g / 'nuclei_native.csv', usecols=['area_um2'])
        a.hist(n.area_um2, bins=bins, histtype='step', density=True, color=col, lw=1,
               label=f'{DONOR[g]}: {100 * (n.area_um2 < 8).mean():.1f}% < 8 µm²')
    a.axvline(8, color='#666666', ls='--', lw=0.6); a.set_xscale('log')
    a.set_xlabel('Native nuclear area (µm², all detections)'); a.set_ylabel('Density')
    a.legend(frameon=False, fontsize=6); a.set_title('Native nuclear area by donor', loc='left'); tag(a, 'a')
    a = fig.add_subplot(top[0, 1])
    for g, col in zip(DONOR, ['#1b9e77', '#7570b3', C['sr']]):
        s = pd.read_csv(S65 / 'paired/samples' / g / 'spot_features_native.csv')
        v = s['area_median'].dropna() if 'area_median' in s else s.filter(like='area').iloc[:, 0].dropna()
        a.hist(v, bins=np.linspace(0, 80, 60), histtype='step', density=True, color=col, lw=1, label=DONOR[g])
    a.set_xlabel('Spot median native nuclear area (µm²)'); a.set_ylabel('Density'); a.legend(frameon=False, fontsize=6)
    a.set_title('Spot-level consequence', loc='left'); tag(a, 'b')
    first = True
    for k, g in enumerate(['GSM6505133', 'GSM6505135']):
        pids = sel[sel.gsm == g].patch_id.tolist()[:2]
        cs = mosaic_contours(S65 / 'paired/samples' / g / 'mosaic/native', pids)
        for j, p in enumerate(pids):
            a = fig.add_subplot(gs[1, 2 * k + j])
            im = img(S65 / 'paired/samples' / g / 'patches/native' / f'{p}.png')[128:384, 128:384]
            micro(a, im, f'{DONOR[g]}, native scan', um_px=0.25, bar_um=10, label=first)
            for ct, t in cs[p]:
                ct = ct - 128
                if (ct[:, 0].max() < 0) or (ct[:, 1].max() < 0) or (ct[:, 0].min() > 256) or (ct[:, 1].min() > 256): continue
                ar = 0.0625 * 0.5 * abs(np.dot(ct[:, 0], np.roll(ct[:, 1], 1)) - np.dot(ct[:, 1], np.roll(ct[:, 0], 1)))
                a.plot(*np.vstack([ct, ct[:1]]).T, color='#00e5ff' if ar < 8 else '#ffe600', lw=0.6)
            a.set_xlim(0, 256); a.set_ylim(256, 0)
            if first:
                tag(a, 'c'); first = False
    fig.legend([Line2D([0], [0], color='#ffe600', lw=2), Line2D([0], [0], color='#00e5ff', lw=2)],
               ['CellViT contour ≥ 8 µm²', 'CellViT contour < 8 µm²'], loc='upper center', ncol=2, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.115))
    save(fig, 'FigS4_donorC_native_segmentation')


# ---------------------------------------------------------------- S5 stain statistics
def s5():
    dist = pd.read_csv(B55 / 'stain_distributions.csv')
    sim = pd.read_csv(B55 / 'stain_TCGA_similarity.csv')
    chans = [('H_OD', 'Haematoxylin OD'), ('E_OD', 'Eosin OD'), ('Lab_L', 'Lab L*'), ('Lab_a', 'Lab a*')]
    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.6, wspace=0.35)
    gcol = {'GSE211895': C['sr'], 'GSE327056': C['old']}
    for i, (ch, nm) in enumerate(chans):
        a = fig.add_subplot(gs[0, i])
        t = dist[(dist.source == 'train') & (dist.channel == ch)]
        a.fill_between(t.bin_center, t.probability, color='#cccccc', lw=0, label='TCGA training')
        for g in GSE:
            d = dist[(dist['sample'] == g) & (dist.source == 'native') & (dist.channel == ch)]
            a.plot(d.bin_center, d.probability, color=gcol[GSE[g]], lw=0.7, alpha=0.9)
        if ch.endswith('OD'): a.set_xlim(0, 0.8 if ch == 'H_OD' else 0.5)
        if ch == 'Lab_a': a.set_xlim(-20, 80)
        a.set_xlabel(nm); a.set_yticks([])
        if i == 0:
            a.set_ylabel('Pixel fraction'); tag(a, 'a')
            a.legend([plt.Rectangle((0, 0), 1, 1, color='#cccccc'), Line2D([0], [0], color=gcol['GSE211895']), Line2D([0], [0], color=gcol['GSE327056'])],
                     ['TCGA training', 'GSE211895 native', 'GSE327056 native'], frameon=False, fontsize=6, loc='upper right')
    a = fig.add_subplot(gs[1, :2])
    piv = sim.pivot(index='sample', columns='source', values='distance_to_TCGA').reindex(list(GSE))
    x = np.arange(len(piv))
    a.bar(x - 0.18, piv['native'], 0.36, color=C['native'], label='native scan')
    a.bar(x + 0.18, piv['true_hires'], 0.36, color='#9a9a9a', label='real hires')
    a.set_xticks(x); a.set_xticklabels(list(piv.index), fontsize=6, rotation=45, ha='right')
    a.text(1, -0.58, 'GSE211895', transform=a.get_xaxis_transform(), ha='center', fontsize=6, color=C['sr'])
    a.text(4.5, -0.58, 'GSE327056', transform=a.get_xaxis_transform(), ha='center', fontsize=6, color=C['old'])
    a.set_ylabel('Standardised stain\ndistance to TCGA'); a.legend(frameon=False, fontsize=6)
    a.set_title('Overall distance to the training stain distribution', loc='left'); tag(a, 'b')
    a = fig.add_subplot(gs[1, 2:])
    comp = ['H_OD_distance', 'E_OD_distance', 'Lab_L_distance', 'Lab_a_distance', 'Lab_b_distance']
    m = sim[sim.source == 'native'].set_index('sample').reindex(list(GSE))[comp].to_numpy()
    im = a.imshow(m.T, cmap='Greys', aspect='auto', vmin=0)
    a.set_yticks(range(len(comp))); a.set_yticklabels(['H OD', 'E OD', 'L*', 'a*', 'b*'])
    a.set_xticks(range(len(GSE))); a.set_xticklabels(list(GSE), rotation=45, ha='right', fontsize=6)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a.text(i, j, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=5.8, color='white' if m[i, j] > 0.6 * np.nanmax(m) else 'black')
    a.set_title('Per-channel distance, native scans', loc='left'); tag(a, 'c')
    save(fig, 'FigS5_stain_statistics')


# ---------------------------------------------------------------- S6 type overlays and confusion matrices
def s6():
    """Millimetre-fixed layout (178 × 190 mm): 3 × 3 images + type legend; bottom row b two confusion matrices, c type counts."""
    from sr13_paper_figures_pub import micro
    sel = pd.read_csv(S65 / 'pipeline_figure/selected_spots.csv')
    conf = pd.read_csv(CT / 'type_confusion.csv')
    WM, HM = 178, 190
    fig = plt.figure(figsize=(WM * MM, HM * MM))
    def AX(x, y, w, h): return fig.add_axes([x / WM, 1 - (y + h) / HM, w / WM, h / HM])
    def tg(x, y, t): fig.text(x / WM, 1 - y / HM, t, fontsize=9, fontweight='bold', ha='left', va='bottom')
    tg(1, 5, 'a')
    for i, g in enumerate(DONOR):
        p = sel[sel.gsm == g].patch_id.iloc[0]
        base = S65 / 'paired/samples' / g
        for j, m in enumerate(['native', 'sr', 'bicubic']):
            cs = mosaic_contours(base / 'mosaic' / m, [p])[p]
            a = AX(8 + j * 40, 7 + i * 43, 38, 38)
            micro(a, img(base / 'patches' / m / f'{p}.png'), f"{DONOR[g]} · {['Native', 'SR (calibrated)', 'Bicubic'][j]}", um_px=0.25, label=(i == 0 and j == 0))
            for ct, t in cs: a.plot(*np.vstack([ct, ct[:1]]).T, color=TCOL[t], lw=0.5)
            a.add_patch(plt.Circle((255.5, 255.5), 110, fill=False, color='white', lw=0.8, ls='--'))
            a.set_xlim(0, 512); a.set_ylim(512, 0)
    a = AX(130, 50, 40, 38); a.axis('off')
    shown = [1, 2, 3, 5]  # dead (4) does not occur in these fields
    a.legend([Line2D([0], [0], color=TCOL[t], lw=2) for t in shown] + [Line2D([0], [0], color='black', ls='--', lw=0.8)],
             [TYPES[t - 1].capitalize() for t in shown] + ['55 µm spot'], frameon=False, fontsize=6, loc='center left', title='CellViT type', title_fontsize=6.3)
    Yb, Hb = 146, 24
    tg(1, Yb - 7, 'b'); tg(133, Yb - 7, 'c')
    lab = ['neoplastic', 'epithelial', 'connective', 'inflammatory']
    for k, m in enumerate(['sr', 'bicubic']):
        d = conf[(conf.group == 'paired_real_hires') & (conf.method == m)].groupby(['native_type', 'method_type'])['count'].sum().unstack(fill_value=0)
        tot = d.sum(1)
        d = d.reindex(index=lab, columns=lab, fill_value=0)
        frac = d.div(d.sum(1), axis=0)
        a = AX(22 + k * 54, Yb, 32, Hb)
        a.imshow(frac.to_numpy(), cmap='Greys', vmin=0, vmax=1, aspect='auto')
        for r in range(frac.shape[0]):
            for c in range(frac.shape[1]):
                v = frac.iat[r, c]
                a.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white' if v > 0.55 else 'black')
        a.set_xticks(range(len(lab))); a.set_xticklabels([t.capitalize() for t in lab], rotation=45, ha='right', fontsize=6)
        a.set_yticks(range(len(lab))); a.set_yticklabels([t.capitalize() for t in lab] if k == 0 else [], fontsize=6)
        for r_, t in enumerate(lab): a.text(len(lab) - 0.4, r_, f'{100 * d.loc[t].sum() / tot[t]:.0f}%', va='center', ha='left', fontsize=6, color='#555555')
        a.text(len(lab) - 0.4, -0.6, 'matched', va='bottom', ha='left', fontsize=6, color='#555555')
        a.set_xlabel(f'{LAB[m]} label'); a.set_ylabel('Native label' if k == 0 else '')
        a.set_title(LAB[m], loc='left', fontsize=6.5)
    ax = AX(147, Yb, 28, Hb)
    cr = pd.read_csv(CT / 'composition_recovery.csv'); cr = cr[(cr.measure == 'n') & (cr.condition == 'real_hires')]
    for k, ct in enumerate(['connective', 'inflammatory', 'neoplastic']):
        z = cr[cr.cell_type == ct].pivot(index='sample', columns='method', values='rho')
        paired(ax, 2.5 * k, 2.5 * k + 1, z.bicubic, z.sr, C['bicubic'], C['sr'])
        ax.text(2.5 * k + 0.5, 1.0, f'{(z.sr > z.bicubic).sum()}/{len(z)}', ha='center', va='bottom', fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_xticks([0.5, 3, 5.5], ['Stromal', 'Immune', 'Tumour'], fontsize=6, rotation=30, ha='right', rotation_mode='anchor'); ax.set_xlim(-0.5, 6.5); ax.set_ylim(0, 0.72)
    ax.set_ylabel('Count ρ vs native'); ax.set_title('Type counts (7 pairs)', loc='left', fontsize=6.5, pad=10)
    fig.savefig(OUT / 'FigS6_celltype_overlays_confusion.pdf'); fig.savefig(OUT / 'FigS6_celltype_overlays_confusion.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / 'FigS6_celltype_overlays_confusion.png', dpi=300); plt.close(fig); print('saved FigS6_celltype_overlays_confusion', flush=True)


# ---------------------------------------------------------------- S7 tumour-nucleus morphology–expression
def s7():
    d = pd.read_csv(CT / 'neoplastic_morphology_expression.csv')
    feats = [('neo_area_median', 'area'), ('neo_area_cv', 'area CV'), ('neo_ecc_median', 'eccentricity')]
    rows = [('paired_real_hires', 'real_hires', 'Real hires pairs (n = 7)'), ('HEST_generated_hires', 'lower_aa0', 'HEST, lower bracket (n = 9)')]
    PL = {'EMT': 'EMT', 'hypoxia': 'Hypoxia', 'proliferation': 'Prolif.', 'classical': 'Classical', 'basal': 'Basal-like', 'interferon': 'IFN', 'immune': 'Immune', 'epithelial': 'Epithelial'}
    fig, axs = plt.subplots(2, 3, figsize=(180 * MM, 100 * MM), gridspec_kw={'hspace': 0.62, 'wspace': 0.06})
    for i, (grp, cond, rt) in enumerate(rows):
        for j, m in enumerate(['native', 'sr', 'bicubic']):
            a = axs[i, j]
            s = d[(d.group == grp) & (d.condition == cond) & (d.method == m)]
            mean = s.groupby(['feature', 'program']).partial_rho_umi_neocount.mean().unstack().reindex(index=[f for f, _ in feats], columns=PROGS)
            pos = (s.assign(p=s.partial_rho_umi_neocount > 0).groupby(['feature', 'program']).p.sum().unstack()
                   .reindex(index=[f for f, _ in feats], columns=PROGS))
            n = s['sample'].nunique()
            a.imshow(mean.to_numpy(), cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
            for r in range(mean.shape[0]):
                for c in range(mean.shape[1]):
                    a.text(c, r, f'{mean.iat[r, c]:+.2f}'.replace('0.', '.').replace('-', '−') + f'\n{int(pos.iat[r, c])}/{n}', ha='center', va='center', fontsize=5.8, linespacing=1.05)
            a.set_xticks(range(len(PROGS))); a.set_xticklabels([PL[p_] for p_ in PROGS], rotation=45, ha='right', fontsize=6)
            a.set_yticks(range(len(feats))); a.set_yticklabels([f'Tumour {l}' for _, l in feats] if j == 0 else [], fontsize=6)
            a.set_title(LAB[m], loc='left', fontsize=6.5)
            if j == 0: a.set_ylabel(rt.replace(' (', '\n('), fontsize=6)
            if j == 0: tag(a, 'ab'[i])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-0.3, 0.3))
    cb = fig.colorbar(sm, ax=axs, fraction=0.015, pad=0.01); cb.set_label('Mean partial ρ (cell = mean, positive/n)', fontsize=6)
    save(fig, 'FigS7_tumour_nucleus_associations')


# ---------------------------------------------------------------- Table S1
def md_table(df):
    fmt = lambda v: '' if pd.isna(v) else (f'{v:.3f}' if isinstance(v, float) else str(v))
    return '\n'.join(['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df.itertuples(index=False)])


def table():
    rel = lambda p: str(Path(p).relative_to(F.ROOT))
    # A: real hires pairs
    sa = pd.read_csv(S65 / 'scale_audit.csv')
    sa = sa[sa.gsm.isin(list(GSE))].set_index('gsm')
    ab = pd.read_csv(PA / 'model_ablation_long.csv')
    rec = pd.read_csv(R9 / 'association_recovery_2026-10-29/recovery_by_sample.csv')
    pr = pd.read_csv(CT / 'prediction_cv_r2.csv'); pr = pr[pr.feature_set == 'morph+types']
    cr = pd.read_csv(CT / 'composition_recovery.csv')
    rows = []
    for g in GSE:
        r = {'sample': g, 'GSE': GSE[g], 'role': 'held-out test' if GSE[g] == 'GSE211895' else 'blur calibration',
             'native_um_px': round(sa.at[g, 'native_mpp'], 4), 'hires_um_px': round(sa.at[g, 'hires_mpp'], 3),
             'neighbour_pitch_um': round(sa.at[g, 'neighbor_pitch_um'], 1), 'n_spots': int(sa.at[g, 'n_spots'])}
        for f in ['density', 'area_median', 'area_cv']:
            for m, k in [('bicubic', 'BC'), ('old_SR08', 'SR08'), ('calibrated_SR', 'SR')]:
                v = ab[(ab.gsm == g) & (ab.method == m) & (ab.feature == f)].rho
                r[f'rho_{f}_{k}'] = round(v.iloc[0], 3) if len(v) else np.nan
        for m, k in [('sr', 'SR'), ('bicubic', 'BC')]:
            r[f'matrix_r_{k}'] = round(rec[(rec.gsm == g) & (rec.method == m)].matrix_element_pearson_r.iloc[0], 3)
        for m, k in [('native', 'native'), ('sr', 'SR'), ('bicubic', 'BC')]:
            r[f'cv_R2_{k}'] = round(pr[(pr['sample'] == g) & (pr.method == m)].cv_r2.mean(), 3)
        rows.append(r)
    A = pd.DataFrame(rows)
    # B: HEST generated hires
    sm = pd.read_csv(HB / 'spot_metrics_by_sample.csv'); ar = pd.read_csv(HB / 'association_recovery_by_sample.csv')
    au = pd.read_csv(HB / 'sample_audit.csv').set_index('sample')
    prh = pr[pr.group == 'HEST_generated_hires']
    rows = []
    for s in au.index:
        for cond, cl in [('upper_area', 'upper (OpenCV area)'), ('lower_aa0', 'lower (skimage AA0)')]:
            r = {'sample': s, 'cancer': au.at[s, 'cancer'], 'organ': au.at[s, 'organ'], 'bracket': cl,
                 'native_um_px': round(au.at[s, 'native_mpp'], 4), 'n_spots': int(au.at[s, 'n_spots']),
                 'SR08_training_test_slide': bool(au.at[s, 'old_SR08_test'])}
            for f in ['density', 'area_median']:
                for m, k in [('bicubic', 'BC'), ('old_SR08', 'SR08'), ('calibrated_SR', 'SR')]:
                    v = sm[(sm['sample'] == s) & (sm.condition == cond) & (sm.method == m) & (sm.feature == f)].rho
                    r[f'rho_{f}_{k}'] = round(v.iloc[0], 3) if len(v) and pd.notna(v.iloc[0]) else np.nan
            for m, k in [('calibrated_SR', 'SR'), ('bicubic', 'BC')]:
                v = ar[(ar['sample'] == s) & (ar.condition == cond) & (ar.method == m)].matrix_element_pearson_r
                r[f'matrix_r_{k}'] = round(v.iloc[0], 3) if len(v) else np.nan
            for m, k in [('native', 'native'), ('sr', 'SR'), ('bicubic', 'BC')]:
                r[f'cv_R2_{k}'] = round(prh[(prh['sample'] == s) & (prh.condition == cond) & (prh.method == m)].cv_r2.mean(), 3)
            rows.append(r)
    B = pd.DataFrame(rows)
    # C: SR08 simulated evaluation (×8)
    ev = pd.read_csv(S8 / 'eval_x8/metrics_by_sample.csv')
    ev = ev[ev.method.isin(['bicubic', 'SwinIR'])].fillna({'native_mpp': -1, 'group': 'not stated'})
    Cc = ev.pivot_table(index=['sample', 'testset', 'stratum', 'group', 'native_mpp'], columns='method',
                        values=['f1', 'rho_density', 'rho_mean_area', 'FDR']).round(3)
    Cc.columns = [f'{a}_{"SR08" if b == "SwinIR" else "BC"}' for a, b in Cc.columns]
    Cc = Cc.reset_index()
    Cc['testset'] = Cc.testset.map(lambda s: 'HEST' if s.startswith('HEST') else 'Xenium')
    Cc['stratum'] = Cc.stratum.map({'\u4e3b\u9a8c\u8bc1\u96c6': 'primary', '\u6cdb\u5316\u6d4b\u8bd5\u96c6': 'generalisation', '\u8f85\u52a9\u9a8c\u8bc1\u96c6': 'Xenium auxiliary', 'primary': 'primary', 'generalisation': 'generalisation', 'Xenium auxiliary': 'Xenium auxiliary'})
    Cc['group'] = Cc.group.map({'FFPE': 'FFPE', '\u51b0\u51bb': 'frozen', '\u672a\u6807\u6ce8': 'not stated', 'frozen': 'frozen', 'not stated': 'not stated'})
    Cc['native_mpp'] = Cc.native_mpp.where(Cc.native_mpp > 0)
    Cc = Cc.sort_values(['testset', 'stratum', 'sample'])
    src = {
        'A': [S65 / 'scale_audit.csv', PA / 'model_ablation_long.csv', R9 / 'association_recovery_2026-10-29/recovery_by_sample.csv', CT / 'prediction_cv_r2.csv'],
        'B': [HB / 'sample_audit.csv', HB / 'spot_metrics_by_sample.csv', HB / 'association_recovery_by_sample.csv', CT / 'prediction_cv_r2.csv'],
        'C': [S8 / 'eval_x8/metrics_by_sample.csv']}
    titles = {'A': 'Part A. Real hires pairs (GSE211895, GSE327056). ρ = spot-level Spearman with native; area and area CV on spots with ≥5 native nuclei; matrix r = Pearson r between 7 × 7 morphology–program partial-ρ matrices; CV R² = spatial-block cross-validated ridge R², mean of 8 programs (morphology + type composition). Density/area/CV not computed for calibration pairs (they were used to fit blur).',
              'B': 'Part B. HEST slides with regenerated hires (upper and lower sharpness brackets).',
              'C': 'Part C. Simulated ×8 evaluation of SR08 (HEST native CellViT reference; Xenium DAPI reference). F1 = nucleus detection; FDR = fraction of predicted nuclei unmatched.'}
    md = ['# Supplementary Table S1. Per-sample results and source files', '']
    for k, df in [('A', A), ('B', B), ('C', Cc)]:
        df.to_csv(OUT / f'TableS1{k}.csv', index=False)
        md += [f'## {titles[k]}', '', md_table(df), '',
               'Source files: ' + '; '.join(f'`{rel(p)}`' for p in src[k]), '']
    (OUT / 'TableS1.md').write_text('\n'.join(md))
    print('saved TableS1', len(A), len(B), len(Cc))


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7', 'T']
    for w in which:
        table() if w == 'T' else globals()[f's{w}']()
