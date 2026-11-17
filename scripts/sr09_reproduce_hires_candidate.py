"""Tested approximate resamplers, not a recovered Space Ranger implementation.

Some local pairs can have high PSNR while retaining substantial texture and
pixel differences. Do not label generated outputs as measured or equivalent
real hires, and do not use this utility to bypass independent-test provenance.
"""
import numpy as np
VALIDATION_NOTE = 'Approximation only; consult results/sr09/hires_reproduction_2026-10-29/README.md.'

def reproduce_hires(native, scalef, *, method='skimage_aa0', target_shape=None):
    """Return uint8 RGB candidate hires with a recorded, explicit resampler.

    native is the exact RGB input array; scalef is tissue_hires_scalef.
    target_shape is optional (height, width), otherwise round(native*scalef).
    No fitted shift, color correction, or model inference is performed.
    The default is a characterized candidate, not a claim of exact reproduction.
    """
    if native.dtype != np.uint8 or native.ndim != 3 or native.shape[2] != 3:
        raise ValueError('Expected uint8 RGB native image.')
    if not 0 < float(scalef) <= 1:
        raise ValueError('Expected a downsampling scale in (0,1].')
    h,w=tuple(map(int,target_shape)) if target_shape is not None else tuple(int(round(v*float(scalef))) for v in native.shape[:2])
    if min(h,w)<1:raise ValueError('Output dimensions must be positive.')
    if method.startswith('skimage_aa'):
        from skimage.transform import resize
        order=int(method.removeprefix('skimage_aa'))
        if order not in (0,1,3):raise ValueError('Supported orders are 0,1,3.')
        out=np.empty((h,w,3),np.uint8)
        for c in range(3):
            plane=resize(native[...,c],(h,w),order=order,anti_aliasing=True,preserve_range=True,mode='reflect')
            out[...,c]=np.rint(np.clip(plane,0,255)).astype(np.uint8)
        return out
    import cv2
    cv2.setNumThreads(2)
    codes={'opencv_area':cv2.INTER_AREA,'opencv_linear':cv2.INTER_LINEAR,'opencv_cubic':cv2.INTER_CUBIC}
    if method.endswith('_fixed_sf'):
        code=codes[method.removesuffix('_fixed_sf')]
        out=cv2.resize(native,None,fx=float(scalef),fy=float(scalef),interpolation=code)
        if out.shape[:2]!=(h,w):raise ValueError('Fixed-SF output differs from requested target shape.')
        return out
    if method in codes:return cv2.resize(native,(w,h),interpolation=codes[method])
    raise ValueError(f'Unknown method: {method}')
