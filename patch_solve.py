import cv2

with open('/home/jacy/project/final_design/axispose/tools/extract_groundtruth.py', 'r') as f:
    code = f.read()

import re
old_pnp = """    ok, rvec, tvec, inliers = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4.0,
        confidence=0.99,
        iterationsCount=200
    )"""
new_pnp = """    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    inliers = None"""
code = code.replace(old_pnp, new_pnp)
code = re.sub(r'ok_refine, rvec_refine.*?if ok_refine:\s+rvec, tvec = rvec_refine, tvec_refine', '', code, flags=re.DOTALL)
with open('/home/jacy/project/final_design/axispose/tools/extract_groundtruth.py', 'w') as f:
    f.write(code)
