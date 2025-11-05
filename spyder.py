import cv2
import numpy as np
import gradio as gr
from typing import Dict, Tuple

def histogram_equalization(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    pdf = hist / hist.sum()
    cdf = pdf.cumsum()
    sk = np.round(cdf * 255).astype(np.uint8)
    equalized = cv2.LUT(image, sk)
    return equalized

def manual_CLAHE(image, tile_size=(8, 8), clip_limit=2.0):
    L = 256
    img_h, img_w = image.shape
    tile_h, tile_w = tile_size
    n_tiles_h = img_h // tile_h
    n_tiles_w = img_w // tile_w
    output = np.zeros_like(image, dtype=np.uint8)
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            tile = image[y0:y1, x0:x1]
            hist = cv2.calcHist([tile], [0], None, [256], [0, 256]).flatten()
            max_clip = clip_limit * tile.size / L
            excess = hist - max_clip
            excess[excess < 0] = 0
            hist = np.minimum(hist, max_clip)
            redistribute = excess.sum() / L
            hist += redistribute
            pdf = hist / hist.sum()
            cdf = pdf.cumsum()
            sk = np.round(cdf * (L - 1)).astype(np.uint8)
            output[y0:y1, x0:x1] = cv2.LUT(tile, sk)
    return output

def gaussian_kernel(sigma):
    size = int(5 * sigma) | 1
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_convolution(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(img_h):
        for j in range(img_w):
            s = 0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    s += padded[i + ki, j + kj] * kernel[ki, kj]
            if s < 0:
                s = 0
            elif s > 255:
                s = 255
            output[i, j] = s
    return output.astype(np.uint8)

def manual_otsu_threshold(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    total = img.size
    n_i = hist
    P_i = n_i / total
    m_g = 0.0
    for i in range(256):
        m_g += i* P_i[i]
    max_sigma2_B = 0
    threshold = 0
    for i in range(256):
        P1 = 0.0
        for j in range(i + 1):
            P1 += P_i[j]
        if P1 == 0 or P1 == 1:
            continue
        m = 0.0
        for j in range(i + 1):
            m += j * P_i[j]
        sigma2_B = ((m_g * P1 - m)**2) / (P1 * (1 - P1))
        if sigma2_B > max_sigma2_B:
            max_sigma2_B = sigma2_B
            threshold = i
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[img > threshold] = 255
    return threshold, binary

def get_count_nuclei(binary_img):
    binary = np.zeros_like(binary_img, dtype=int)
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i, j] > 0:
                binary[i, j] = 1
    rows, cols = binary.shape
    labels = np.zeros_like(binary, dtype=int)
    current_label = 1
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    def nei_label(r, c, label):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if (0 <= x < rows) and (0 <= y < cols):
                if binary[x, y] == 1 and labels[x, y] == 0:
                    labels[x, y] = label
                    for dx, dy in neighbors:
                        stack.append((x + dx, y + dy))
    for i in range(rows):
        for j in range(cols):
            if binary[i, j] == 1 and labels[i, j] == 0:
                nei_label(i, j, current_label)
                current_label += 1
    count = current_label - 1
    return labels, count

def builtin_otsu_threshold(img):
    threshold, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold, binary

import numpy as np

def get_labels_and_centroids(binary_img):
    labels, count = get_count_nuclei(binary_img)

    centers_cc = []
    centers_bbox = []
    bboxes = []

    for label_id in range(1, count + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue

        cx_cc = float(xs.mean())
        cy_cc = float(ys.mean())

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        cx_bbox = (xmin + xmax) / 2.0
        cy_bbox = (ymin + ymax) / 2.0

        centers_cc.append((cx_cc, cy_cc))
        centers_bbox.append((int(round(cx_bbox)), int(round(cy_bbox))))
        bboxes.append((xmin, ymin, xmax, ymax))

    return labels, count, centers_bbox, bboxes, centers_cc


def img_area(bin_img):
    return np.count_nonzero(bin_img)

def img_perimeter(border_img):
    return np.count_nonzero(border_img)

def find_max_d(bin_img):
    min_x = min_y = 1e9
    max_x = max_y = 0
    h, w = bin_img.shape
    for x in range(h):
        for y in range(w):
            if bin_img[x, y] > 0:
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)
    return (max_x - min_x, max_y - min_y)

def calc_descriptors(binary_img, i):
    se = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_img, se, iterations=1)
    border_img = binary_img - eroded
    area = img_area(binary_img)
    perimeter = img_perimeter(border_img)
    max_d_tup = find_max_d(binary_img)
    max_d = max(max_d_tup)
    compactness = (perimeter ** 2) / area
    form_factor = (4 * np.pi * area) / (perimeter ** 2)
    roundness = (4 * area) / (np.pi * max_d ** 2)
    return form_factor, roundness, compactness

def compute_region_descriptors(labels_mat, closing_mask, bboxes):
    descriptors = []
    for idx in range(1, len(bboxes) + 1):
        region = np.zeros_like(closing_mask, dtype=np.uint8)
        region[labels_mat == idx] = 255
        se = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(region, se, iterations=1)
        border = region - eroded
        area = img_area(region)
        perimeter = img_perimeter(border)
        max_d_tup = find_max_d(region)
        max_d = int(max(max_d_tup))
        ff, rd, cp = calc_descriptors(region, idx)
        descriptors.append({
            "label": idx,
            "area": int(area),
            "perimeter": int(perimeter),
            "max_diameter": int(max_d),
            "form_factor": float(ff),
            "roundness": float(rd),
            "compactness": float(cp)
        })
    return descriptors

def process_nuclei_image(input_image, sigma=1.0, equalization_method='manual_clahe', threshold_method='manual_otsu'):
    if input_image is None:
        return None
    results = {}
    img_color = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    cells = img_color[:, :, 0]
    results['original'] = np.stack([cells]*3, axis=-1)
    manual_he = histogram_equalization(cells)
    results['manual_he'] = np.stack([manual_he]*3, axis=-1)
    manual_clahe_img = manual_CLAHE(cells, tile_size=(8, 8), clip_limit=2.0)
    results['manual_clahe'] = np.stack([manual_clahe_img]*3, axis=-1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    opencv_clahe_img = clahe.apply(cells)
    results['opencv_clahe'] = np.stack([opencv_clahe_img]*3, axis=-1)
    results['_manual_he_raw'] = manual_he
    results['_manual_clahe_raw'] = manual_clahe_img
    results['_opencv_clahe_raw'] = opencv_clahe_img
    if equalization_method == 'manual_he':
        equalized_cells = manual_he
    elif equalization_method == 'manual_clahe':
        equalized_cells = manual_clahe_img
    else:
        equalized_cells = opencv_clahe_img
    results['selected_equalization'] = equalization_method
    kernel = gaussian_kernel(sigma)
    blurred_cells = apply_convolution(equalized_cells, kernel)
    results['blurred'] = np.stack([blurred_cells]*3, axis=-1)
    results['_blurred_raw'] = blurred_cells
    manual_thresh_val, manual_thresh_img = manual_otsu_threshold(blurred_cells)
    results['threshold_manual_otsu'] = np.stack([manual_thresh_img]*3, axis=-1)
    results['threshold_manual_otsu_value'] = manual_thresh_val
    builtin_thresh_val, builtin_thresh_img = builtin_otsu_threshold(blurred_cells)
    results['threshold_builtin_otsu'] = np.stack([builtin_thresh_img]*3, axis=-1)
    results['threshold_builtin_otsu_value'] = builtin_thresh_val
    if threshold_method == 'manual_otsu':
        thresh_binary = manual_thresh_img
        thresh_value = manual_thresh_val
    else:
        thresh_binary = builtin_thresh_img
        thresh_value = builtin_thresh_val
    results['threshold'] = np.stack([thresh_binary]*3, axis=-1)
    results['otsu_thresh'] = thresh_value
    results['selected_threshold'] = threshold_method
    se = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh_binary, se, iterations=3)
    results['eroded'] = np.stack([eroded]*3, axis=-1)
    opening = cv2.dilate(eroded, se, iterations=3)
    results['opening'] = np.stack([opening]*3, axis=-1)
    dilated = cv2.dilate(opening, se, iterations=3)
    results['dilated'] = np.stack([dilated]*3, axis=-1)
    closing = cv2.erode(dilated, se, iterations=3)
    results['closing'] = np.stack([closing]*3, axis=-1)
    eroded_boundary = cv2.erode(closing, se, iterations=1)
    boundaries = closing - eroded_boundary
    results['boundaries'] = np.stack([boundaries]*3, axis=-1)
    output_image = img_rgb.copy()
    output_image[boundaries > 0] = [255, 0, 0]
    labels_mat, count, centers_bbox, bboxes, centers_cc = get_labels_and_centroids(closing)
    results['labels_matrix'] = labels_mat
    results['bboxes'] = bboxes
    results['centroids_cc'] = centers_cc
    results['centroids'] = centers_bbox
    results['count'] = count
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness_text = 1
    thickness_outline = 3
    measure_thickness = max(thickness_text, thickness_outline)
    H, W = output_image.shape[:2]
    for idx, (cx, cy) in enumerate(centers_bbox, start=1):
        text = str(idx)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness_outline)
        org = (int(cx - tw * 0.5), int(cy + th * 0.5))  
        cv2.putText(output_image, text, org, font, font_scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
        cv2.putText(output_image, text, org, font, font_scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

    desc = compute_region_descriptors(labels_mat, closing, bboxes)
    results['descriptors'] = desc
    results['final'] = output_image
    return results

def get_step_info(step_num, equalization_variant='manual_clahe', threshold_variant='manual_otsu'):
    steps = [
        ('original', 'Original Image (Blue Channel)', 'Extracted blue channel from the input image'),
        (equalization_variant, 'Histogram Equalization', 'Enhanced contrast using histogram equalization'),
        ('blurred', 'Gaussian Blur', 'Applied Gaussian blur for noise reduction'),
        ('threshold', 'Otsu Thresholding', 'Binary thresholding using Otsu\'s method'),
        ('eroded', 'Erosion', 'Morphological erosion (3 iterations)'),
        ('opening', 'Opening', 'Morphological opening (dilation after erosion)'),
        ('dilated', 'Dilation', 'Morphological dilation (3 iterations)'),
        ('closing', 'Closing', 'Morphological closing (erosion after dilation)'),
        ('boundaries', 'Boundary Detection', 'Extracted nuclei boundaries'),
        ('final', 'Final Output', 'Boundaries + numeric labels overlaid on original image')
    ]
    return steps[step_num]

def format_descriptors(desc):
    if not desc:
        return ""
    lines = []
    for d in desc:
        lines.append(
            f"- #{d['label']}: area={d['area']}, perimeter={d['perimeter']}, maxD={d['max_diameter']}, "
            f"compactness={d['compactness']:.3f}, form_factor={d['form_factor']:.3f}, roundness={d['roundness']:.3f}"
        )
    return "\n".join(lines)

def display_output(mode, results_state, current_step, equalization_variant, threshold_variant, sigma):
    if results_state is None:
        return None, "Please upload an image first.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    if mode == "Display Final Output":
        count = results_state.get('count', 0)
        eq_method = results_state.get('selected_equalization', 'manual_clahe')
        thresh_method = results_state.get('selected_threshold', 'manual_otsu')
        desc_text = format_descriptors(results_state.get('descriptors', []))
        info_text = f"**Nuclei Count: {count}**\n\n*Using: {eq_method.replace('_', ' ').title()} + {thresh_method.replace('_', ' ').title()}*\n\n**Region Descriptors**\n{desc_text}"
        return (
            results_state['final'],
            info_text,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    else:
        img_key, title, description = get_step_info(current_step, equalization_variant, threshold_variant)
        output_img = results_state[img_key]
        info_text = f"### Step {current_step + 1}/10: {title}\n{description}"
        if current_step == 1:
            method_name = equalization_variant.replace('_', ' ').title()
            info_text += f"\n\n**Current Method: {method_name}**"
        if current_step == 2:
            eq_method = results_state.get('selected_equalization', 'manual_clahe')
            info_text += f"\n\n**Sigma (σ): {sigma}**"
            info_text += f"\n*Using: {eq_method.replace('_', ' ').title()}*"
        if current_step == 3:
            thresh_method = threshold_variant.replace('_', ' ').title()
            info_text += f"\n\n**Current Method: {thresh_method}**"
            info_text += f"\n**Threshold Value: {results_state['otsu_thresh']}**"
        if current_step == 9:
            info_text += f"\n\n**Nuclei Count: {results_state['count']}**"
            eq_method = results_state.get('selected_equalization', 'manual_clahe')
            thresh_method = results_state.get('selected_threshold', 'manual_otsu')
            info_text += f"\n*Pipeline: {eq_method.replace('_', ' ').title()} → {thresh_method.replace('_', ' ').title()}*"
            desc_text = format_descriptors(results_state.get('descriptors', []))
            info_text += f"\n\n**Region Descriptors**\n{desc_text}"
        show_eq_buttons = (current_step == 1)
        show_thresh_buttons = (current_step == 3)
        return (
            output_img,
            info_text,
            gr.update(visible=True),
            gr.update(visible=True, interactive=(current_step > 0)),
            gr.update(visible=True, interactive=(current_step < 9)),
            gr.update(visible=show_eq_buttons),
            gr.update(visible=show_thresh_buttons)
        )

def process_and_update(input_image, mode, sigma, equalization_method='manual_clahe', threshold_method='manual_otsu'):
    if input_image is None:
        return None, None, "Please upload an image.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    results = process_nuclei_image(input_image, sigma, equalization_method, threshold_method)
    current_step = 0
    equalization_variant = equalization_method
    threshold_variant = threshold_method
    if mode == "Display Final Output":
        count = results.get('count', 0)
        eq_method = results.get('selected_equalization', 'manual_clahe')
        thresh_method = results.get('selected_threshold', 'manual_otsu')
        desc_text = format_descriptors(results.get('descriptors', []))
        info_text = f"**Nuclei Count: {count}**\n\n*Using: {eq_method.replace('_', ' ').title()} + {thresh_method.replace('_', ' ').title()}*\n\n**Region Descriptors**\n{desc_text}"
        return (
            results,
            results['final'],
            info_text,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    else:
        img_key, title, description = get_step_info(current_step, equalization_variant, threshold_variant)
        info_text = f"### Step {current_step + 1}/10: {title}\n{description}"
        return (
            results,
            results[img_key],
            info_text,
            gr.update(visible=True),
            gr.update(visible=True, interactive=False),
            gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def next_step(results_state, current_step, equalization_variant, threshold_variant, sigma):
    if results_state is None:
        return current_step, None, "Please upload an image first.", gr.update(), gr.update(), gr.update(), gr.update()
    new_step = min(current_step + 1, 9)
    img_key, title, description = get_step_info(new_step, equalization_variant, threshold_variant)
    output_img = results_state[img_key]
    info_text = f"### Step {new_step + 1}/10: {title}\n{description}"
    if new_step == 1:
        method_name = equalization_variant.replace('_', ' ').title()
        info_text += f"\n\n**Current Method: {method_name}**"
    if new_step == 2:
        eq_method = results_state.get('selected_equalization', 'manual_clahe')
        info_text += f"\n\n**Sigma (σ): {sigma}**"
        info_text += f"\n*Using: {eq_method.replace('_', ' ').title()}*"
    if new_step == 3:
        thresh_method = threshold_variant.replace('_', ' ').title()
        info_text += f"\n\n**Current Method: {thresh_method}**"
        info_text += f"\n**Threshold Value: {results_state['otsu_thresh']}**"
    if new_step == 9:
        info_text += f"\n\n**Nuclei Count: {results_state['count']}**"
        eq_method = results_state.get('selected_equalization', 'manual_clahe')
        thresh_method = results_state.get('selected_threshold', 'manual_otsu')
        info_text += f"\n*Pipeline: {eq_method.replace('_', ' ').title()} → {thresh_method.replace('_', ' ').title()}*"
        desc_text = format_descriptors(results_state.get('descriptors', []))
        info_text += f"\n\n**Region Descriptors**\n{desc_text}"
    show_eq_buttons = (new_step == 1)
    show_thresh_buttons = (new_step == 3)
    return (
        new_step,
        output_img,
        info_text,
        gr.update(interactive=(new_step > 0)),
        gr.update(interactive=(new_step < 9)),
        gr.update(visible=show_eq_buttons),
        gr.update(visible=show_thresh_buttons)
    )

def prev_step(results_state, current_step, equalization_variant, threshold_variant, sigma):
    if results_state is None:
        return current_step, None, "Please upload an image first.", gr.update(), gr.update(), gr.update(), gr.update()
    new_step = max(current_step - 1, 0)
    img_key, title, description = get_step_info(new_step, equalization_variant, threshold_variant)
    output_img = results_state[img_key]
    info_text = f"### Step {new_step + 1}/10: {title}\n{description}"
    if new_step == 1:
        method_name = equalization_variant.replace('_', ' ').title()
        info_text += f"\n\n**Current Method: {method_name}**"
    if new_step == 2:
        eq_method = results_state.get('selected_equalization', 'manual_clahe')
        info_text += f"\n\n**Sigma (σ): {sigma}**"
        info_text += f"\n*Using: {eq_method.replace('_', ' ').title()}*"
    if new_step == 3:
        thresh_method = threshold_variant.replace('_', ' ').title()
        info_text += f"\n\n**Current Method: {thresh_method}**"
        info_text += f"\n**Threshold Value: {results_state['otsu_thresh']}**"
    if new_step == 9:
        info_text += f"\n\n**Nuclei Count: {results_state['count']}**"
        eq_method = results_state.get('selected_equalization', 'manual_clahe')
        thresh_method = results_state.get('selected_threshold', 'manual_otsu')
        info_text += f"\n*Pipeline: {eq_method.replace('_', ' ').title()} → {thresh_method.replace('_', ' ').title()}*"
        desc_text = format_descriptors(results_state.get('descriptors', []))
        info_text += f"\n\n**Region Descriptors**\n{desc_text}"
    show_eq_buttons = (new_step == 1)
    show_thresh_buttons = (new_step == 3)
    return (
        new_step,
        output_img,
        info_text,
        gr.update(interactive=(new_step > 0)),
        gr.update(interactive=(new_step < 9)),
        gr.update(visible=show_eq_buttons),
        gr.update(visible=show_thresh_buttons)
    )

def change_equalization_variant(input_image, results_state, variant, current_step, threshold_variant, sigma):
    if results_state is None or current_step != 1:
        return variant, results_state, None, "Please upload an image first.", gr.update(), gr.update(), gr.update(), gr.update()
    new_results = process_nuclei_image(input_image, sigma, variant, threshold_variant)
    variant_names = {'manual_he': 'Manual Histogram Equalization','manual_clahe': 'Manual CLAHE','opencv_clahe': 'OpenCV Built-in CLAHE'}
    img_key, title, description = get_step_info(current_step, variant, threshold_variant)
    output_img = new_results[variant]
    info_text = f"### Step {current_step + 1}/10: {title}\n{description}\n\n**Current Method: {variant_names[variant]}**\n\n*All subsequent steps updated with this choice*"
    he_variant = "primary" if variant == "manual_he" else "secondary"
    clahe_variant = "primary" if variant == "manual_clahe" else "secondary"
    opencv_variant = "primary" if variant == "opencv_clahe" else "secondary"
    return (
        variant,
        new_results,
        output_img,
        info_text,
        gr.update(visible=True),
        gr.update(variant=he_variant),
        gr.update(variant=clahe_variant),
        gr.update(variant=opencv_variant)
    )

def change_threshold_variant(input_image, results_state, variant, current_step, equalization_variant, sigma):
    if results_state is None or current_step != 3:
        return variant, results_state, None, "Please upload an image first.", gr.update(), gr.update(), gr.update()
    new_results = process_nuclei_image(input_image, sigma, equalization_variant, variant)
    variant_names = {'manual_otsu': 'Manual Otsu Thresholding','builtin_otsu': 'OpenCV Built-in Otsu'}
    img_key, title, description = get_step_info(current_step, equalization_variant, variant)
    if variant == 'manual_otsu':
        output_img = new_results['threshold_manual_otsu']
        thresh_value = new_results['threshold_manual_otsu_value']
    else:
        output_img = new_results['threshold_builtin_otsu']
        thresh_value = new_results['threshold_builtin_otsu_value']
    info_text = f"### Step {current_step + 1}/10: {title}\n{description}\n\n**Current Method: {variant_names[variant]}**\n**Threshold Value: {thresh_value}**\n\n*All subsequent steps updated with this choice*"
    manual_variant = "primary" if variant == "manual_otsu" else "secondary"
    builtin_variant = "primary" if variant == "builtin_otsu" else "secondary"
    return (
        variant,
        new_results,
        output_img,
        info_text,
        gr.update(visible=True),
        gr.update(variant=manual_variant),
        gr.update(variant=builtin_variant)
    )

with gr.Blocks(title="Nuclei Segmentation & Counter", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Osteosarcoma Cell Nucleus Segementation
        """
    )
    results_state = gr.State(None)
    current_step_state = gr.State(0)
    equalization_variant_state = gr.State('manual_clahe')
    threshold_variant_state = gr.State('manual_otsu')
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="numpy", height=450)
            mode_selector = gr.Radio(choices=["Display Final Output", "Show Step by Step"], value="Display Final Output", label="Display Mode", info="Choose how to view the results")
            sigma_input = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Gaussian Blur Sigma (σ)", info="Controls the amount of blurring")
            process_btn = gr.Button("Process Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output", height=450)
            info_text = gr.Markdown("**Awaiting image upload...**")
            with gr.Row(visible=False) as step_controls:
                prev_btn = gr.Button("Previous", interactive=False)
                next_btn = gr.Button("Next ", interactive=True)
            with gr.Row(visible=False) as equalization_buttons:
                gr.Markdown("**Choose Equalization Method:**")
            with gr.Row(visible=False) as equalization_buttons_row:
                manual_he_btn = gr.Button("Manual HE", size="sm", variant="secondary")
                manual_clahe_btn = gr.Button("Manual CLAHE", size="sm", variant="primary")
                opencv_clahe_btn = gr.Button("OpenCV CLAHE", size="sm", variant="secondary")
            with gr.Row(visible=False) as threshold_buttons:
                gr.Markdown("**Choose Thresholding Method:**")
            with gr.Row(visible=False) as threshold_buttons_row:
                manual_otsu_btn = gr.Button("Manual Otsu", size="sm", variant="primary")
                builtin_otsu_btn = gr.Button("Built-in Otsu", size="sm", variant="secondary")
    gr.Markdown(
        """
        ---
        ### Processing Pipeline:
        1. **Original Image** - Extract blue channel  
        2. **Histogram Equalization** - Enhance contrast (3 variants available)  
        3. **Gaussian Blur** - Reduce noise  
        4. **Otsu Thresholding** - Binary segmentation  
        5. **Erosion** - Remove small objects  
        6. **Opening** - Smooth object boundaries  
        7. **Dilation** - Expand objects  
        8. **Closing** - Fill small holes  
        9. **Boundary Detection** - Extract cell boundaries  
        10. **Final Output** - Overlay boundaries and numbered labels; count nuclei
        """
    )
    process_btn.click(
        fn=process_and_update,
        inputs=[input_image, mode_selector, sigma_input, equalization_variant_state, threshold_variant_state],
        outputs=[results_state, output_image, info_text, step_controls, prev_btn, next_btn, equalization_buttons_row, threshold_buttons_row]
    ).then(fn=lambda: 0, outputs=current_step_state)
    input_image.change(
        fn=process_and_update,
        inputs=[input_image, mode_selector, sigma_input, equalization_variant_state, threshold_variant_state],
        outputs=[results_state, output_image, info_text, step_controls, prev_btn, next_btn, equalization_buttons_row, threshold_buttons_row]
    ).then(fn=lambda: 0, outputs=current_step_state)
    mode_selector.change(
        fn=display_output,
        inputs=[mode_selector, results_state, current_step_state, equalization_variant_state, threshold_variant_state, sigma_input],
        outputs=[output_image, info_text, step_controls, prev_btn, next_btn, equalization_buttons_row, threshold_buttons_row]
    )
    def handle_sigma_change(input_image, mode, sigma, eq_var, thresh_var, results_state):
        if results_state is None:
            return results_state, None, "Please upload and process an image first.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        return process_and_update(input_image, mode, sigma, eq_var, thresh_var)
    sigma_input.change(
        fn=handle_sigma_change,
        inputs=[input_image, mode_selector, sigma_input, equalization_variant_state, threshold_variant_state, results_state],
        outputs=[results_state, output_image, info_text, step_controls, prev_btn, next_btn, equalization_buttons_row, threshold_buttons_row]
    ).then(fn=lambda: 0, outputs=current_step_state)
    next_btn.click(
        fn=next_step,
        inputs=[results_state, current_step_state, equalization_variant_state, threshold_variant_state, sigma_input],
        outputs=[current_step_state, output_image, info_text, prev_btn, next_btn, equalization_buttons_row, threshold_buttons_row]
    )
    prev_btn.click(
        fn=prev_step,
        inputs=[results_state, current_step_state, equalization_variant_state, threshold_variant_state, sigma_input],
        outputs=[current_step_state, output_image, info_text, prev_btn, next_btn, equalization_buttons_row, threshold_buttons_row]
    )
    manual_he_btn.click(
        fn=change_equalization_variant,
        inputs=[input_image, results_state, gr.State('manual_he'), current_step_state, threshold_variant_state, sigma_input],
        outputs=[equalization_variant_state, results_state, output_image, info_text, equalization_buttons_row, manual_he_btn, manual_clahe_btn, opencv_clahe_btn]
    )
    manual_clahe_btn.click(
        fn=change_equalization_variant,
        inputs=[input_image, results_state, gr.State('manual_clahe'), current_step_state, threshold_variant_state, sigma_input],
        outputs=[equalization_variant_state, results_state, output_image, info_text, equalization_buttons_row, manual_he_btn, manual_clahe_btn, opencv_clahe_btn]
    )
    opencv_clahe_btn.click(
        fn=change_equalization_variant,
        inputs=[input_image, results_state, gr.State('opencv_clahe'), current_step_state, threshold_variant_state, sigma_input],
        outputs=[equalization_variant_state, results_state, output_image, info_text, equalization_buttons_row, manual_he_btn, manual_clahe_btn, opencv_clahe_btn]
    )
    manual_otsu_btn.click(
        fn=change_threshold_variant,
        inputs=[input_image, results_state, gr.State('manual_otsu'), current_step_state, equalization_variant_state, sigma_input],
        outputs=[threshold_variant_state, results_state, output_image, info_text, threshold_buttons_row, manual_otsu_btn, builtin_otsu_btn]
    )
    builtin_otsu_btn.click(
        fn=change_threshold_variant,
        inputs=[input_image, results_state, gr.State('builtin_otsu'), current_step_state, equalization_variant_state, sigma_input],
        outputs=[threshold_variant_state, results_state, output_image, info_text, threshold_buttons_row, manual_otsu_btn, builtin_otsu_btn]
    )

if __name__ == "__main__":
    demo.launch(share=False)
