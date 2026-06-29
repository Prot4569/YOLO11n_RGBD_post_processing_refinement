import cv2
import time
import ctypes
import numpy as np
import tensorrt as trt
import time
import threading
import socket
import struct

latest_frame = None
latest_boxes = []
latest_scores = []
latest_class_ids = []
latest_selected = []

running = True

from pal.products.qcar import QCarRealSense

# ============================================================
# LABELS
# ============================================================

CLASS_NAMES = [
    "camouflage_soldier",
    "weapon",
    "military_tank",
    "military_truck",
    "military_vehicle",
    "civilian",
    "soldier",
    "civilian_vehicle",
    "military_artillery",
    "trench",
    "military_aircraft",
    "military_warship"
]

# ============================================================
# CUDA RUNTIME
# ============================================================

cudart = ctypes.CDLL('libcudart.so')

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# ============================================================
# TENSORRT ENGINE
# ============================================================

ENGINE_PATH = "/home/nvidia/Reinaldo/yolo11n_military_fp16_v10.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

# print("\n=== ENGINE INFO ===")

# for i in range(engine.num_bindings):
#     print(
#         i,
#         engine.get_binding_name(i),
#         engine.get_binding_shape(i)
#     )

# print("===================\n")

context = engine.create_execution_context()

# ============================================================
# INPUT / OUTPUT SHAPES
# ============================================================

input_shape = (1, 3, 640, 640)

output_shape = (1, 16, 8400)

input_size = int(np.prod(input_shape) * np.float32().nbytes)

output_size = int(np.prod(output_shape) * np.float32().nbytes)

# ============================================================
# CUDA MEMORY
# ============================================================

d_input = ctypes.c_void_p()
d_output = ctypes.c_void_p()

cudart.cudaMalloc(ctypes.byref(d_input), input_size)
cudart.cudaMalloc(ctypes.byref(d_output), output_size)

# ============================================================
# HOST OUTPUT
# ============================================================

host_output = np.empty(output_shape, dtype=np.float32)

# ============================================================
# CAMERA
# ============================================================

import threading

latest_frame = None
latest_depth = None
running = True
def camera_thread(cam):

    global latest_frame
    global latest_depth
    global running

    while running:

        cam.read_RGB()

        latest_frame = cam.imageBufferRGB.copy()

        cam.read_depth()

        latest_depth = cam.imageBufferDepthPX.copy()

with QCarRealSense(mode='RGB&DEPTH') as cam:

    cam_thread = threading.Thread(
        target=camera_thread,
        args=(cam,)
    )

    cam_thread.daemon = True

    cam_thread.start()

    # ============================================================
    # SOCKET PARA O PC
    # ============================================================

    PC_IP = "192.168.2.10"
    PORT = 9999

    client = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    client.connect((PC_IP, PORT))

    print("Conectado ao PC")

    bindings = [
        int(d_input.value),
        int(d_output.value)
    ]

    while True:

        loop_start = time.time()

        # ====================================================
        # READ CAMERA
        # ====================================================

        if latest_frame is None:
            continue

        frame_yolo = latest_frame.copy()

        h, w = frame_yolo.shape[:2]

        t_cam = time.time()

        # ====================================================
        # PREPROCESS
        # ====================================================

        # t0 = time.time()

        img = cv2.resize(frame_yolo, (640,640))

        # t1 = time.time()

        img = img.astype(np.float32) / 255.0

        # t2 = time.time()

        img = np.transpose(img, (2,0,1))

        # t3 = time.time()

        img = np.expand_dims(img, axis=0)

        # t4 = time.time()

        img = np.ascontiguousarray(img)

        # t5 = time.time()

        # print(
        #     "RESIZE={}ms FLOAT={}ms TRANS={}ms EXPAND={}ms CONTIG={}ms".format(
        #         int(1000*(t1-t0)),
        #         int(1000*(t2-t1)),
        #         int(1000*(t3-t2)),
        #         int(1000*(t4-t3)),
        #         int(1000*(t5-t4))
        #     )
        # )

        t_pre = time.time()

        # ====================================================
        # COPY INPUT TO GPU
        # ====================================================

        cudart.cudaMemcpy(
            d_input,
            img.ctypes.data,
            input_size,
            cudaMemcpyHostToDevice
        )

        # ====================================================
        # TRT INFERENCE
        # ====================================================

        t_inf_i = time.time()
        context.execute_v2(bindings)
        t_inf_f = time.time()

        # ====================================================
        # COPY OUTPUT TO CPU
        # ====================================================

        cudart.cudaMemcpy(
            host_output.ctypes.data,
            d_output,
            output_size,
            cudaMemcpyDeviceToHost
        )

        t_inf = time.time()

        # ====================================================
        # POSTPROCESS: Score NMS
        # ====================================================

        USE_RGBD = True
        MIN_DEPTH = 300
        MAX_DEPTH = 3000
        ROI_PERCENT = 0.05
        CONF = 0.35
        IOU_LIM = 0.5
        RAT_LIM = 0.15
        STD_LIM = 15

        output = host_output[0]

        boxes = []
        scores = []
        class_ids = []

        distances = []
        depth_stds = []
        valid_ratios = []

        scores_matrix = output[4:, :]

        max_scores = np.max(scores_matrix, axis=0)

        valid = np.where(max_scores > CONF)[0]

        # print("candidatos =", len(valid))

        for i in valid:

            class_scores = scores_matrix[:, i]

            class_id = np.argmax(class_scores)

            confidence = class_scores[class_id]

            x = output[0, i]
            y = output[1, i]
            bw = output[2, i]
            bh = output[3, i]

            x1 = int((x - bw / 2) * w / 640)
            y1 = int((y - bh / 2) * h / 640)

            width = int(bw * w / 640)
            height = int(bh * h / 640)

            # -------------------------
            # PROFUNDIDADE RGB-D
            # -------------------------

            distance_m = -1
            depth_quality = 1.0
            depth_std = -1
            valid_ratio = 0.0   

            if USE_RGBD and latest_depth is not None:

                cx = x1 + width//2
                cy = y1 + height//2

                w_roi = max(2, int(ROI_PERCENT*width))
                h_roi = max(2, int(ROI_PERCENT*height))

                roi = latest_depth[
                    max(0,cy-h_roi):min(latest_depth.shape[0],cy+h_roi),
                    max(0,cx-w_roi):min(latest_depth.shape[1],cx+w_roi),
                    0
                ]

                valid_depth = roi[
                    (roi>MIN_DEPTH)&
                    (roi<MAX_DEPTH)
                ]

                if len(valid_depth)>0:

                    distance_m = np.median(valid_depth)/1000

                    valid_ratio = len(valid_depth)/roi.size

                    depth_std = np.std(valid_depth)

                    # penalização pelo percentual de pixels validos

                    if valid_ratio < RAT_LIM:

                        depth_quality *= 0.5

                    # penalização pela qualidade da profundidade

                    if depth_std > STD_LIM:

                        depth_quality *= 0.5
                else:
                    depth_quality *= 0.5

            # ==========================
            # RGB-D Confidence Refinement
            # ==========================

            if USE_RGBD:
            
                confidence *= depth_quality

            if confidence < CONF:
                continue
            
            boxes.append([x1, y1, width, height])

            class_ids.append(class_id)
            
            scores.append(float(confidence))

            distances.append(distance_m)
            
            depth_stds.append(depth_std)
            
            valid_ratios.append(valid_ratio)

        # ====================================================
        # POSTPROCESS: IoU NMS
        # ====================================================

        indices = np.argsort(scores)[::-1]

        selected = []

        for i in indices:

            keep = True

            box1 = boxes[i]

            x1, y1, w1, h1 = box1

            area1 = w1 * h1

            for j in selected:

                box2 = boxes[j]

                x2, y2, w2, h2 = box2

                xx1 = max(x1, x2)
                yy1 = max(y1, y2)

                xx2 = min(x1 + w1, x2 + w2)
                yy2 = min(y1 + h1, y2 + h2)

                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)

                inter = inter_w * inter_h

                area2 = w2 * h2

                union = area1 + area2 - inter

                iou = inter / union if union > 0 else 0

                if iou > IOU_LIM:
                    keep = False
                    break

            if keep:
                selected.append(i)

        latest_boxes = boxes
        latest_scores = scores
        latest_class_ids = class_ids
        latest_selected = selected

        latest_distances = distances
        latest_depth_stds = depth_stds
        latest_valid_ratios = valid_ratios

        t_post = time.time()

        if len(latest_selected) > 0:

            print("\n========== DETECTIONS ==========")

            for i in latest_selected:

                print(
                    f"{CLASS_NAMES[latest_class_ids[i]]}"
                    f" conf={latest_scores[i]:.3f}"
                )
                if USE_RGBD:
                    print(
                        f" dist={latest_distances[i]:.2f}"
                        f" std={latest_depth_stds[i]:.1f}"
                        f" ratio={100*latest_valid_ratios[i]:.1f}%"
                )

            print("===============================\n")

        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        display_frame = latest_frame.copy()

        for i in latest_selected:

            x, y, bw, bh = latest_boxes[i]

            class_id = latest_class_ids[i]

            conf = latest_scores[i]

            label = CLASS_NAMES[class_id]

            # -------------------------
            # TEXTO
            # -------------------------

            if distance_m > 0:

                text = "{} {:.2f} {:.1f}m".format(
                    label,
                    conf,
                    distance_m
                )

            else:

                text = "{} {:.2f}".format(
                    label,
                    conf
                )

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + bw, y + bh),
                (0,255,0),
                2
            )

            cv2.putText(
                display_frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )
        
        t_draw = time.time()

        # ====================================================
        # FPS OF DETECTION
        # ====================================================

        fps_det = 1 / (t_post - t_cam)

        cv2.putText(
            display_frame,
            "FPS_DETECTION: {:.2f}".format(fps_det),
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # ====================================================
        # SHOW
        # ====================================================

        # stream_frame = cv2.resize(
        #     display_frame,
        #     (640,480)
        # )
        
        ret, jpg = cv2.imencode(
            '.jpg',
            display_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not ret:
            continue
        
        data = jpg.tobytes()

        client.sendall(
            struct.pack(">L", len(data))
        )

        client.sendall(data)
        
        t_send = time.time()

        # ====================================================
        # TOTAL FPS
        # ====================================================

        fps_tot = 1 / (t_send - loop_start)
        fps_yolo = 1 / (t_inf_f - t_inf_i)
    
        print("TOTAL FPS:", round(fps_tot,2))
        print("YOLO FPS:", round(fps_yolo,2))
        # print("DETECTION FPS:", round(fps_det,2))

        # print(
        #     "CAM={:.1f} PRE={:.1f} INF={:.1f} POST={:.1f} DRAW={:.1f} SEND={:.1f}".format(
        #         1000*(t_cam-loop_start),
        #         1000*(t_pre-t_cam),
        #         1000*(t_inf_f-t_inf_i),
        #         1000*(t_post-t_inf),
        #         1000*(t_draw-t_post),
        #         1000*(t_send-t_draw)
        #     )
        # )

# ============================================================
# CLEANUP
# ============================================================

cudart.cudaFree(d_input)
cudart.cudaFree(d_output)

cv2.destroyAllWindows()