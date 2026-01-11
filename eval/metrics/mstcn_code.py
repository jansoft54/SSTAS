# taken from https://github.com/yabufarha/ms-tcn/blob/c1f537b18772564433445d63948b80a096a3529f/eval.py

import numpy as np


def get_labels_start_end_time(frame_wise_labels, ignored_classes=[-100]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in ignored_classes:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in ignored_classes:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in ignored_classes:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in ignored_classes:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, ignored_classes=[-100]):
    P, _, _ = get_labels_start_end_time(recognized, ignored_classes)
    Y, _, _ = get_labels_start_end_time(ground_truth, ignored_classes)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap,  ignored_classes=[-100]):
    p_label, p_start, p_end = get_labels_start_end_time(
        recognized, ignored_classes)
    y_label, y_start, y_end = get_labels_start_end_time(
        ground_truth, ignored_classes)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(
            p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union) * \
            ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def get_labels_start_end_time_(frame_wise_labels, ignore_ids):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]

    # Init first segment
    if frame_wise_labels[0] not in ignore_ids:
        labels.append(frame_wise_labels[0])
        starts.append(0)

    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in ignore_ids:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in ignore_ids:
                ends.append(i)
            last_label = frame_wise_labels[i]

    if last_label not in ignore_ids:
        ends.append(len(frame_wise_labels))

    return labels, starts, ends


def f_score_per_class(recognized, ground_truth, overlap, num_classes, ignored_classes):
    """
    Berechnet TP, FP, FN pro Klasse.
    Args:
        recognized: Liste von Prediction Frames
        ground_truth: Liste von GT Frames
        overlap: IoU Threshold (z.B. 0.5)
        num_classes: Totale Anzahl Klassen (z.B. 19 oder 25)
        ignored_classes: Liste von IDs (z.B. [-100])
    Returns:
        tp, fp, fn: Jeweils ein Numpy Array der Länge num_classes
    """
    p_label, p_start, p_end = get_labels_start_end_time_(
        recognized, ignored_classes)
    y_label, y_start, y_end = get_labels_start_end_time_(
        ground_truth, ignored_classes)

    # Arrays initialisieren (für JEDE Klasse)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    # Wir müssen über ALLE Klassen iterieren, die im Video vorkommen könnten
    # Um Zeit zu sparen, nehmen wir die Union aus Pred und GT Klassen
    relevant_classes = set(np.unique(p_label)) | set(np.unique(y_label))

    for c in relevant_classes:
        c = int(c)
        if c >= num_classes:
            continue  # Schutz gegen Out of Bounds

        # Segmente dieser spezifischen Klasse filtern
        # Indizes der Segmente, die Klasse c haben
        idx_p_cls = [i for i, x in enumerate(p_label) if x == c]
        idx_y_cls = [i for i, x in enumerate(y_label) if x == c]

        hits = np.zeros(len(idx_y_cls))
        tp_c = 0
        fp_c = 0

        # Predictions gegen GT prüfen
        for i_p in idx_p_cls:
            # Pred Segment Zeit
            p_s = p_start[i_p]
            p_e = p_end[i_p]

            best_iou = 0
            best_idx = -1

            # Suche bestes GT Segment Match
            for idx_local, i_y in enumerate(idx_y_cls):
                y_s = y_start[i_y]
                y_e = y_end[i_y]

                # Intersection
                inter = min(p_e, y_e) - max(p_s, y_s)
                # Union
                union = max(p_e, y_e) - min(p_s, y_s)

                if union > 0:
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx_local

            if best_iou >= overlap and best_idx != -1 and not hits[best_idx]:
                tp_c += 1
                hits[best_idx] = 1
            else:
                fp_c += 1

        fn_c = len(idx_y_cls) - sum(hits)

        # Ins Array schreiben
        tp[c] = tp_c
        fp[c] = fp_c
        fn[c] = fn_c

    return tp, fp, fn
