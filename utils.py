import os
import uuid
import numpy as np
from datetime import datetime
from sklearn.preprocessing import normalize


def orientation(pos):
    u = normalize(X=pos[1:, :] - pos[:-1, :], norm='l2', axis=1)
    u1 = u[1:, :]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            value = getattr(args, arg)
            if isinstance(value, list):
                list_str = ", ".join(str(item) for item in value)
                formatted_value = f"[{list_str}]"
            else:
                formatted_value = str(value)
            info = f"{arg:<30} : {formatted_value:>35}"
            print(info)
            f.write(info + '\n')
        f.write(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def create_run_directory(args, base_output_dir="output"):
    os.makedirs(base_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]
    run_identifier = f"run_{timestamp}_{random_id}_{args.model_name}_{args.model_idx}"

    run_dir = os.path.join(base_output_dir, run_identifier)
    os.makedirs(run_dir, exist_ok=True)

    log_dir = run_dir
    weights_dir = os.path.join(run_dir, "weights")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"run dir: {run_dir}")
    print(f"log dir: {log_dir}")
    print(f"weights dir: {weights_dir}")

    return run_dir, log_dir, weights_dir


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label * pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum / pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum / label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2. * precision * recall) / max(precision + recall, 1e-8)
        f_max = max(f, f_max)

    return f_max
