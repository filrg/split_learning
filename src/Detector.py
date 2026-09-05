import numpy as np
import torch
from collections import defaultdict, deque
import random

EPS = 1e-8

def _prepare_z(z: torch.Tensor):
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    if z.ndim < 2:
        raise ValueError("z must have ndim>=2")
    if z.ndim > 2:
        z = z.mean(dim=tuple(range(2, z.ndim)))
    return z

def _energy_distance(X: np.ndarray, Y: np.ndarray):
    if len(X) == 0 or len(Y) == 0: return 0.0
    def mpd(A, B): return np.mean(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)))
    Exy, Exx, Eyy = mpd(X, Y), mpd(X, X), mpd(Y, Y)
    return float(2 * Exy - Exx - Eyy)

# ---- NEW: shape score = variance share of PC1 (bigger => flatter/elliptic)
def _shape_share(Z: np.ndarray):
    if len(Z) < 2:
        return 0.0
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # covariance (K is small); add EPS on diag for stability
    C = (Zc.T @ Zc) / max(1, len(Zc) - 1) + EPS * np.eye(Zc.shape[1])
    vals = np.linalg.eigvalsh(C)  # ascending
    s = float(vals[-1] / (np.sum(vals) + EPS))
    return s

class ZPerLabelRuleDetectorOnline:
    """
    Online, per-label detector (logits-only).
    Decision:
        Stage-1: (energy_ratio >= thr_energy) AND (centroid_ratio >= thr_centroid)
        Stage-2: shape confirm:
                 shape_ratio >= thr_shape  OR  shape_diff >= shape_diff_min
    """

    def __init__(self,
                 n_classes: int,
                 per_client_limit: int = 100,
                 store_per_batch: int = 64,
                 baseline_cap: int = 512,
                 min_client_points: int = 64,
                 min_baseline_points: int = 128,
                 thr_energy: float = 2.5,
                 thr_centroid: float = 2.3,
                 # --- NEW: shape thresholds ---
                 thr_shape: float = 1.2,          # ratio threshold for shape_share
                 shape_diff_min: float = 0.10,    # absolute diff threshold
                 consecutive: int = 2,
                 seed: int = 42):
        self.C = int(n_classes)
        self.LIMIT = int(per_client_limit)
        self.STORE = int(store_per_batch)
        self.CAP = int(baseline_cap)
        self.MIN_X = int(min_client_points)
        self.MIN_Y = int(min_baseline_points)

        self.thE = float(thr_energy)
        self.thC = float(thr_centroid)
        self.thShape = float(thr_shape)
        self.minShapeDiff = float(shape_diff_min)
        self.consecutive = int(consecutive)

        random.seed(seed); np.random.seed(seed)

        self.buf = defaultdict(lambda: deque(maxlen=self.LIMIT))
        self.proto_sum = defaultdict(lambda: np.zeros(self.C, dtype=np.float64))
        self.proto_cnt = defaultdict(int)
        self.latest_metrics = defaultdict(dict)
        self.flags = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.consecutive)))

    def _baseline_Y(self, label: int, exclude_client: str):
        pool = []
        for (cid, lab), dq in self.buf.items():
            if lab != label or cid == exclude_client or len(dq) == 0:
                continue
            arr = list(dq)
            random.shuffle(arr)
            take = min(max(1, self.CAP // 8), len(arr))
            pool.extend(arr[:take])
        if len(pool) > self.CAP:
            random.shuffle(pool)
            pool = pool[:self.CAP]
        return np.asarray(pool, dtype=np.float64)

    def _proto(self, label: int):
        if self.proto_cnt[label] >= 50:
            return self.proto_sum[label] / max(1, self.proto_cnt[label])
        return None

    def _median_others(self, label: int, client_id: str):
        others = [v for cid, v in self.latest_metrics[label].items() if cid != client_id]
        if len(others) >= 3:
            medE = np.median([o["energy"] for o in others])
            medC = np.median([o["cshift"] for o in others])
            # you could also store shape of others later if muốn ratio-LOO cho shape
            return medE, medC
        return None, None

    def update_batch(self, *, z: torch.Tensor, y: torch.Tensor, client_id: str):
        assert isinstance(client_id, str)
        z = _prepare_z(z)  # (B,C)
        Zb = z.detach().cpu().numpy()
        y = y.detach().cpu().numpy().reshape(-1).astype(np.int64)
        assert Zb.shape[0] == y.shape[0]

        per_label = {}
        any_flag = False
        labels = np.unique(y).tolist()

        for l in labels:
            idx = np.where(y == l)[0]
            if idx.size == 0:
                continue

            # limit samples added from this batch
            take = idx.copy()
            np.random.shuffle(take)
            take = take[:min(self.STORE, len(take))]
            Zi = Zb[take]

            key = (client_id, l)
            for row in Zi:
                self.buf[key].append(row)

            X = np.asarray(list(self.buf[key]), dtype=np.float64)
            Yb = self._baseline_Y(l, exclude_client=client_id)

            if len(X) < self.MIN_X or len(Yb) < self.MIN_Y:
                proto = self._proto(l)
                if proto is None:
                    self.proto_sum[l] += Zi.sum(0)
                    self.proto_cnt[l] += len(Zi)
                per_label[l] = {
                    "present_in_batch": True,
                    "stored_client": int(len(X)),
                    "baseline": int(len(Yb)),
                    "status": "insufficient",
                    "suspicious_now": False,
                    "suspicious_consecutive": False,
                }
                continue

            # --- metrics ---
            energy = _energy_distance(X, Yb)
            proto = self._proto(l)
            if proto is None:
                proto = Yb.mean(0)
            cshift = float(np.linalg.norm(X.mean(0) - proto))

            self.latest_metrics[l][client_id] = dict(energy=energy, cshift=cshift)

            medE, medC = self._median_others(l, client_id)
            if medE is None or medE <= 0:
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(Yb))
                H1, H2 = Yb[perm[:len(Yb) // 2]], Yb[perm[len(Yb) // 2:]]
                medE = _energy_distance(H1, H2) + EPS
                medC = float(np.linalg.norm(H1.mean(0) - H2.mean(0))) + EPS

            re = energy / (medE + EPS)
            rc = cshift / (medC + EPS)

            # ---------- Stage 1: ratio gate ----------
            ratio_pass = (re >= self.thE) and (rc >= self.thC)

            # ---------- Stage 2: shape confirm (ellipticity) ----------
            shape_ratio = 1.0
            shape_diff  = 0.0
            shape_confirm = False
            if ratio_pass:
                sx = _shape_share(X)      # PC1 share of client
                sy = _shape_share(Yb)     # PC1 share of baseline
                shape_ratio = sx / (sy + EPS)
                shape_diff  = sx - sy
                shape_confirm = (shape_ratio >= self.thShape) or (shape_diff >= self.minShapeDiff)

            suspicious_now = bool(ratio_pass and shape_confirm)
            bypass = bool(ratio_pass and not shape_confirm)

            # consecutive (chỉ cộng khi đã xác nhận)
            self.flags[client_id][l].append(1 if suspicious_now else 0)
            consec_ok = (len(self.flags[client_id][l]) >= self.consecutive and
                         sum(list(self.flags[client_id][l])[-self.consecutive:]) == self.consecutive)
            suspicious_consec = bool(consec_ok)
            any_flag = any_flag or suspicious_consec

            # cập nhật prototype nếu KHÔNG xác nhận (benign hoặc bypass)
            if not suspicious_now:
                self.proto_sum[l] += Zi.sum(0)
                self.proto_cnt[l] += len(Zi)

            per_label[l] = {
                "present_in_batch": True,
                "stored_client": int(len(X)),
                "baseline": int(len(Yb)),
                "energy": energy, "centroid_shift": cshift,
                "energy_ratio": re, "centroid_ratio": rc,
                "shape_ratio": shape_ratio, "shape_diff": shape_diff,
                "shape_thresholds": {"thr_shape": self.thShape, "shape_diff_min": self.minShapeDiff},
                "ratio_pass": bool(ratio_pass),
                "shape_confirm": bool(shape_confirm),
                "bypass_due_to_shape": bool(bypass),
                "suspicious_now": suspicious_now,
                "recent_flags": list(self.flags[client_id][l])[-self.consecutive:],
                "suspicious_consecutive": suspicious_consec,
                "status": "ok"
            }

        summary = {
            "client_id": client_id,
            "any_label_flagged": any_flag,
            "flagged_labels": [l for l, info in per_label.items() if info.get("suspicious_consecutive", False)],
            "consecutive": self.consecutive,
            "thresholds": {"energy_ratio": self.thE, "centroid_ratio": self.thC,
                           "thr_shape": self.thShape, "shape_diff_min": self.minShapeDiff}
        }
        return summary, per_label
