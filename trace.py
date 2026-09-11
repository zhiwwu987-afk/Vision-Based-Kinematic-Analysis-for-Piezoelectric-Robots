import os
import cv2
import numpy as np
import random
from ultralytics import YOLO


DEVICE = None
CONF_THRES = 0.25
TARGET_CLASS_ID = None

SHOW_WINDOW = False
SAVE_VIDEO = True

RANDOM_SEED = 42

REACQUIRE_EVERY_N = 1

SUPPORTED_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v',
                  '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.M4V')


def preprocess_gray(img, clahe=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return gray_eq


def polygons_to_mask(polygons, shape_hw):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygons is None or len(polygons) == 0:
        return mask
    cv2.fillPoly(mask, [np.array(p, dtype=np.int32) for p in polygons], 1)
    return mask


def centroid_from_mask(mask_bin):
    mask_u8 = (mask_bin > 0).astype(np.uint8)
    M = cv2.moments(mask_u8, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return np.array([[[cx, cy]]], dtype=np.float32)


def refine_point_with_ecc(prev_gray, next_gray, prev_pt, pred_pt, patch_half=15, ecc_iters=30, eps=1e-5):
    def get_patch(img, center, half):
        x, y = int(round(center[0])), int(round(center[1]))
        x1, y1 = x - half, y - half
        x2, y2 = x + half + 1, y + half + 1
        pad_left, pad_top = max(0, -x1), max(0, -y1)
        pad_right, pad_bottom = max(0, x2 - img.shape[1]), max(0, y2 - img.shape[0])
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(img.shape[1], x2), min(img.shape[0], y2)
        patch = img[y1c:y2c, x1c:x2c].copy()
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            patch = cv2.copyMakeBorder(
                patch, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE
            )
        return patch

    template = get_patch(prev_gray, prev_pt.ravel(), patch_half).astype(np.float32)
    current = get_patch(next_gray, pred_pt.ravel(), patch_half).astype(np.float32)

    if template.size == 0 or current.size == 0 or template.shape != current.shape:
        return pred_pt, False

    warp_mode = cv2.MOTION_TRANSLATION
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, eps)

    try:
        cc, warp = cv2.findTransformECC(template, current, warp, warp_mode, criteria, None, 1)
        dx, dy = warp[0, 2], warp[1, 2]
        refined = pred_pt.copy()
        refined[0, 0, 0] += dx
        refined[0, 0, 1] += dy
        return refined, True
    except cv2.error:
        return pred_pt, False


def get_instance_mask_from_results(res, target_class_id=None, ref_pt=None):
    if res.masks is None or res.masks.xy is None or len(res.masks.xy) == 0:
        return None, None

    H, W = res.orig_img.shape[:2]
    polys = res.masks.xy

    classes = None
    if res.boxes is not None and res.boxes.cls is not None:
        try:
            classes = res.boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            classes = np.array(res.boxes.cls, dtype=int)

    valid_idx = []
    for i, poly in enumerate(polys):
        if target_class_id is not None and classes is not None:
            if classes[i] != target_class_id:
                continue
        valid_idx.append(i)

    if not valid_idx:
        return None, None

    if ref_pt is None:
        idx = max(valid_idx, key=lambda i: abs(cv2.contourArea(np.array(polys[i], dtype=np.float32))))
    else:
        rx, ry = float(ref_pt[0]), float(ref_pt[1])

        def centroid_of_poly(poly):
            cnt = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return cx, cy
            arr = np.array(poly, dtype=np.float32)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())

        best_i, best_d = None, 1e18
        for i in valid_idx:
            cx, cy = centroid_of_poly(polys[i])
            d = (cx - rx) ** 2 + (cy - ry) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        idx = best_i

    mask_bin = polygons_to_mask([polys[idx]], (H, W))
    return mask_bin, idx


def reacquire_centroid(frame, model, ref_pt=None):
    try:
        results = model.predict(source=frame, imgsz=640, conf=CONF_THRES, verbose=False, device=DEVICE)
        res = results[0]
        mask_bin, _ = get_instance_mask_from_results(res, target_class_id=TARGET_CLASS_ID, ref_pt=ref_pt)
        if mask_bin is None or mask_bin.sum() == 0:
            return None, None
        p = centroid_from_mask(mask_bin)
        return p, mask_bin
    except Exception as e:
        print(f"[WARN] Relocation failed: {e}")
        return None, None


def process_one_video(video_path, out_video_path, out_txt_path, model):
    random.seed(RANDOM_SEED)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Can't open: {video_path}")
        return False

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print(f"[ERROR] Failed to read the first frame: {video_path}")
        cap.release()
        return False

    H, W = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    try:
        results = model.predict(source=first_frame, imgsz=640, conf=CONF_THRES, verbose=False, device=DEVICE)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {video_path}, {e}")
        cap.release()
        return False

    res = results[0]
    mask_bin, idx = get_instance_mask_from_results(res, target_class_id=TARGET_CLASS_ID)
    if mask_bin is None or mask_bin.sum() == 0:
        print(f"[WARN] No valid instance mask detected: {video_path}")
        try:
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write("frame,time_s,x,y,ecc_ok,fb_err\n")
                f.write("# no valid instance mask detected on the first frame\n")
        except Exception as e:
            print(f"[WARN] Failed to write to txt file: {out_txt_path}, {e}")
        cap.release()
        return False

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    prev_gray = preprocess_gray(first_frame, clahe)
    p0 = centroid_from_mask(mask_bin)
    if p0 is None:
        print(f"[WARN] The centroid cannot be calculated within the masked region.: {video_path}")
        try:
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write("frame,time_s,x,y,ecc_ok,fb_err\n")
                f.write("# failed to compute centroid on the first frame\n")
        except Exception as e:
            print(f"[WARN] Failed to write to txt file: {out_txt_path}, {e}")
        cap.release()
        return False

    start_x, start_y = int(round(p0[0, 0, 0])), int(round(p0[0, 0, 1]))
    print(f"  Centroid: ({start_x}, {start_y})")

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=0
    )
    fb_thresh = 10

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    trail_canvas = np.zeros_like(first_frame)
    trail_color = (0, 255, 255)
    trail_thickness = 2
    track_log = []
    frame_idx = 0
    track_log.append({
        "frame": frame_idx,
        "time_s": 0.0,
        "x": float(start_x),
        "y": float(start_y),
        "ecc_ok": False,
        "fb_err": None
    })

    prev_pt = p0.copy()
    prev_xy_int = (start_x, start_y)

    cv2.circle(trail_canvas, prev_xy_int, 2, trail_color, -1)

    vis0 = first_frame.copy()
    vis0 = cv2.addWeighted(vis0, 1.0, trail_canvas, 0.8, 0)
    cv2.circle(vis0, prev_xy_int, 6, (0, 255, 0), -1)
    if SHOW_WINDOW:
        cv2.imshow("Tracking with Trail", vis0)
        cv2.waitKey(1)
    if writer is not None:
        writer.write(vis0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1

        next_gray = preprocess_gray(frame, clahe)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pt, None, **lk_params)

        fb_err = None
        if p1 is not None and st is not None and st[0, 0] == 1:
            p0r, st_back, err_back = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, p1, None, **lk_params)
            if p0r is not None and st_back is not None and st_back[0, 0] == 1:
                fb_err = float(np.linalg.norm(prev_pt - p0r))

        need_reacq = (
            (frame_idx % REACQUIRE_EVERY_N == 0) or
            (p1 is None or st is None or st[0, 0] == 0) or
            (fb_err is not None and fb_err > fb_thresh)
        )

        ecc_ok = False
        current_pt = None

        if need_reacq:
            ref_for_reacq = p1 if (p1 is not None and st is not None and st[0, 0] == 1) else prev_pt
            ref_xy = ref_for_reacq.reshape(-1).tolist() if ref_for_reacq is not None else None

            p_re, _ = reacquire_centroid(frame, model, ref_pt=ref_xy)
            if p_re is not None:
                current_pt = p_re
                ecc_ok = False
            else:
                if p1 is not None and st is not None and st[0, 0] == 1:
                    refined_pt, ecc_ok = refine_point_with_ecc(prev_gray, next_gray, prev_pt, p1, patch_half=15)
                    current_pt = refined_pt if ecc_ok else p1
                else:
                    current_pt = prev_pt.copy()
        else:
            refined_pt, ecc_ok = refine_point_with_ecc(prev_gray, next_gray, prev_pt, p1, patch_half=15)
            current_pt = refined_pt if ecc_ok else p1

        x, y = current_pt.ravel()
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))
        cx, cy = int(round(x)), int(round(y))

        cv2.line(trail_canvas, prev_xy_int, (cx, cy), trail_color, trail_thickness, lineType=cv2.LINE_AA)
        cv2.circle(trail_canvas, (cx, cy), 1, trail_color, -1)

        vis = frame.copy()
        vis = cv2.addWeighted(vis, 1.0, trail_canvas, 0.8, 0)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
        if SHOW_WINDOW:
            cv2.imshow("Tracking with Trail", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if writer is not None:
            writer.write(vis)

        timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        track_log.append({
            "frame": frame_idx,
            "time_s": round(float(timestamp_s), 3),
            "x": float(cx),
            "y": float(cy),
            "ecc_ok": bool(ecc_ok),
            "fb_err": fb_err
        })

        prev_gray = next_gray
        prev_pt = np.array([[[x, y]]], dtype=np.float32)
        prev_xy_int = (cx, cy)

    cap.release()
    if writer is not None:
        writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    try:
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write("frame,time_s,x,y,ecc_ok,fb_err\n")
            for r in track_log:
                f.write(f"{r['frame']},{r['time_s']},{r['x']},{r['y']},{int(r['ecc_ok'])},{'' if r['fb_err'] is None else r['fb_err']}\n")
    except Exception as e:
        print(f"[WARN] Failed to write to txt file: {out_txt_path}, {e}")
    return True

def main(input_dir=None, output_dir=None, weights_path=None):
    global INPUT_DIR, OUTPUT_DIR, WEIGHTS_PATH
    INPUT_DIR = input_dir
    OUTPUT_DIR = output_dir
    WEIGHTS_PATH = weights_path

    if not os.path.isdir(INPUT_DIR):
        raise ValueError(f"[ERROR] Invalid input folder: {INPUT_DIR}")
    if not os.path.isdir(OUTPUT_DIR):
        raise ValueError(f"[ERROR] Invalid output folder: {OUTPUT_DIR}")
    if not os.path.isfile(WEIGHTS_PATH):
        raise ValueError(f"[ERROR] Invalid weights file: {WEIGHTS_PATH}")

    if not os.path.isdir(INPUT_DIR):
        raise NotADirectoryError(f"Invalid input folder: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Invalid weights: {WEIGHTS_PATH}")
    print("[INFO] Loading YOLO model")
    model = YOLO(WEIGHTS_PATH)

    video_files = [os.path.join(INPUT_DIR, f)
                   for f in os.listdir(INPUT_DIR)
                   if os.path.splitext(f)[1] in SUPPORTED_EXTS]
    if not video_files:
        print("[WARN] No video files found in the input folder.")
        return

    print(f"[INFO] Number of videos pending processing: {len(video_files)}")
    success_count, fail_count = 0, 0
    out_videos = []

    for i, video_path in enumerate(video_files, 1):
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_video_path = os.path.join(OUTPUT_DIR, f"{base}_tracked.mp4")
        out_txt_path = os.path.join(OUTPUT_DIR, f"{base}_track.txt")

        print(f"\n[{i}/{len(video_files)}] process: {video_path}")
        ok = process_one_video(video_path, out_video_path, out_txt_path, model)
        if ok:
            print(f"  Done: {out_video_path}")
            print(f"  Trajectory: {out_txt_path}")
            success_count += 1
            out_videos.append(out_video_path)
        else:
            print(f"  Fail: {video_path}")
            fail_count += 1
            out_videos.append(out_video_path)

    print(f"\n[SUMMARY] Success: {success_count}, Failure: {fail_count}, Total: {len(video_files)}")
    return out_videos
if __name__ == "__main__":
    # --- 请在这里修改为你自己的实际路径 ---
    INPUT_DIR = "C:/Users/Administrator/Desktop/input video"  # 存放待处理视频的文件夹
    OUTPUT_DIR = "C:/Users/Administrator/Desktop/output video"      # 存放处理结果的文件夹
    WEIGHTS_PATH = "C:/Users/Administrator/Desktop/2026UMcartrace/UMcartrace/best.pt" # YOLO模型权重文件的完整路径
    
    # 调用 main 函数并传入参数
    main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, weights_path=WEIGHTS_PATH)