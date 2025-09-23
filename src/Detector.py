# z_rule_detector_online_no_mmd.py
import numpy as np
import torch
from collections import defaultdict, deque
import random

EPS = 1e-8


# ---------- helpers ----------
def _prepare_z(z: torch.Tensor):
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    if z.ndim < 2:
        raise ValueError("z must have ndim>=2")
    if z.ndim > 2:  # pool không gian → (B, C)
        z = z.mean(dim=tuple(range(2, z.ndim)))
    return z


def _energy_distance(X: np.ndarray, Y: np.ndarray):
    if len(X) == 0 or len(Y) == 0: return 0.0

    def mpd(A, B): return np.mean(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)))

    Exy, Exx, Eyy = mpd(X, Y), mpd(X, X), mpd(Y, Y)
    return float(2 * Exy - Exx - Eyy)


def _label_margin(Z: np.ndarray, l: int):
    top_other = np.max(Z[:, np.arange(Z.shape[1]) != l], axis=1)
    return Z[:, l] - top_other


# ---------- detector ----------
class ZPerLabelRuleDetectorOnline:
    """
    Online, per-label detector từ logits z (không đọc file).
    Logic quyết định:
        suspicious_now = (energy_ratio >= thr_energy)
                         AND (centroid_ratio >= thr_centroid)

    - energy_ratio = energy / median_energy_others
    - centroid_ratio = centroid_shift / median_centroid_others
    """

    def __init__(self,
                 n_classes: int,
                 per_client_limit: int = 100,  # tối đa mẫu X cho 1 (client,label)
                 store_per_batch: int = 64,  # số điểm lấy từ batch để lưu
                 baseline_cap: int = 512,  # tối đa mẫu baseline Y
                 min_client_points: int = 64,  # tối thiểu để quyết
                 min_baseline_points: int = 128,
                 thr_energy: float = 2.5,  # ngưỡng tỉ lệ energy
                 thr_centroid: float = 2.3,  # ngưỡng tỉ lệ centroid_shift
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
        self.consecutive = int(consecutive)

        random.seed(seed)
        np.random.seed(seed)

        # Bộ đệm logits: (client,label) -> deque[np.ndarray(C,)]
        self.buf = defaultdict(lambda: deque(maxlen=self.LIMIT))
        # Prototype (mean) theo nhãn l, chỉ cập nhật bằng batch KHÔNG bị cờ
        self.proto_sum = defaultdict(lambda: np.zeros(self.C, dtype=np.float64))
        self.proto_cnt = defaultdict(int)
        # Lưu metric gần nhất của từng client/nhãn (để tính median others)
        self.latest_metrics = defaultdict(dict)  # label -> {client_id: dict(energy, cshift)}
        # Cờ liên tiếp
        self.flags = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.consecutive)))

    # ---- baseline hiện tại của nhãn l (loại client này) ----
    def _baseline_Y(self, label: int, exclude_client: str):
        pool = []
        for (cid, lab), dq in self.buf.items():
            if lab != label or cid == exclude_client or len(dq) == 0: continue
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
            return medE, medC
        return None, None

    def update_batch(self, *, z: torch.Tensor, y: torch.Tensor, client_id: str):
        """
        Trả về:
          summary: {"client_id", "any_label_flagged", "flagged_labels", "thresholds", ...}
          per_label: dict[label] -> metrics, ratios, flags cho NHÃN CÓ MẶT trong batch
        """
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
            if idx.size == 0: continue

            # Lấy <= STORE điểm từ batch để đưa vào X của client
            take = idx.copy()
            np.random.shuffle(take)
            take = take[:min(self.STORE, len(take))]
            Zi = Zb[take]  # (n,C)

            # Cập nhật buffer (client,label)
            key = (client_id, l)
            for row in Zi:
                self.buf[key].append(row)

            # Chuẩn bị X (client) và Y (baseline)
            X = np.asarray(list(self.buf[key]), dtype=np.float64)
            Yb = self._baseline_Y(l, exclude_client=client_id)

            # Nếu chưa đủ dữ liệu → chưa quyết
            if len(X) < self.MIN_X or len(Yb) < self.MIN_Y:
                # vẫn cập nhật prototype để ấm máy
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
                # proxy: dùng mean baseline khi chưa đủ ấm máy
                proto = Yb.mean(0)
            cshift = float(np.linalg.norm(X.mean(0) - proto))

            # Lưu metric để tính median others về sau
            self.latest_metrics[l][client_id] = dict(energy=energy, cshift=cshift)

            # --- ratios vs median others (fallback nếu chưa đủ) ---
            medE, medC = self._median_others(l, client_id)
            if medE is None or medE <= 0:
                # fallback: tách đôi baseline để ước lượng "benign median"
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(Yb))
                H1, H2 = Yb[perm[:len(Yb) // 2]], Yb[perm[len(Yb) // 2:]]
                medE = _energy_distance(H1, H2) + EPS
                medC = float(np.linalg.norm(H1.mean(0) - H2.mean(0))) + EPS

            re = energy / (medE + EPS)
            rc = cshift / (medC + EPS)

            # --- quyết định theo rule mới ---
            suspicious_now = (re >= self.thE) and (rc >= self.thC)

            # consecutive
            self.flags[client_id][l].append(1 if suspicious_now else 0)
            consec_ok = (len(self.flags[client_id][l]) >= self.consecutive and
                         sum(list(self.flags[client_id][l])[-self.consecutive:]) == self.consecutive)
            suspicious_consec = bool(consec_ok)
            any_flag = any_flag or suspicious_consec

            # chỉ cập nhật prototype bằng dữ liệu KHÔNG bị cờ
            if not suspicious_now:
                self.proto_sum[l] += Zi.sum(0)
                self.proto_cnt[l] += len(Zi)

            per_label[l] = {
                "present_in_batch": True,
                "stored_client": int(len(X)),
                "baseline": int(len(Yb)),
                "energy": energy, "centroid_shift": cshift,
                "energy_ratio": re, "centroid_ratio": rc,
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
            "thresholds": {"energy_ratio": self.thE, "centroid_ratio": self.thC}
        }
        return summary, per_label
