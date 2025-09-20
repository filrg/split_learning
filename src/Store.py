import os, json, time, uuid
from typing import Optional, Dict, List
import torch


# ====== Helpers ======
def _now_iso():
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_torch_save(path: str, obj: dict):
    tmp = path + ".tmp_" + uuid.uuid4().hex
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on most OS


def _prepare_z(z: torch.Tensor, reduce_spatial: str = "mean"):
    """
    Chuẩn hoá logits về (B, C).
    - Nếu z có dạng (B, C, H, W, ...) thì gộp các chiều không gian theo 'mean' hoặc 'max'.
    """
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    z = z.detach()
    if z.ndim < 2:
        raise ValueError(f"z.ndim must be >=2 (got {z.ndim})")

    if z.ndim > 2:
        # gộp mọi chiều sau dim=1
        reduce_dims = tuple(range(2, z.ndim))
        if reduce_spatial == "mean":
            z = z.mean(dim=reduce_dims)
        elif reduce_spatial == "max":
            z = z.amax(dim=reduce_dims)
        else:
            raise ValueError("reduce_spatial must be 'mean' or 'max'")
    # giờ z: (B, C)
    return z


# ====== ZLogger (Writer) ======
class ZLogger:
    """
    Lưu logits z theo từng client và từng nhãn thành các shard .pt:
    - Thư mục gốc: root/
        └─ client_id/
           └─ label_<l>/
               └─ <ts>_e{epoch}_s{step}_n{n}_C{C}.pt   (tensor z: [n, C])
    - Manifest: root/manifest.jsonl (mỗi dòng là một shard)
    """

    def __init__(self, root: str, reduce_spatial: str = "mean"):
        self.root = os.path.abspath(root)
        self.reduce_spatial = reduce_spatial
        _ensure_dir(self.root)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        # Nếu manifest chưa có, tạo file trống
        if not os.path.exists(self.manifest_path):
            open(self.manifest_path, "a").close()

    @torch.no_grad()
    def log_batch(
            self,
            *,
            client_id: str,
            z: torch.Tensor,  # (B, C) hoặc (B, C, H, W, ...)
            y: torch.Tensor,  # (B,)
            epoch: Optional[int] = None,
            step: Optional[int] = None,
            extra: Optional[Dict] = None,
    ):
        """
        Ghi một batch vào nhiều shard theo từng nhãn.
        Trả về danh sách đường dẫn shard đã ghi.
        """
        if not isinstance(client_id, str):
            raise TypeError("client_id must be a string")

        z = _prepare_z(z, self.reduce_spatial)  # (B, C)
        y = y.detach().reshape(-1).to(torch.long)  # (B,)
        if z.size(0) != y.numel():
            raise ValueError(f"B mismatch: z[{z.size(0)}] vs y[{y.numel()}]")

        z = z.cpu()  # lưu CPU để nhẹ VRAM
        y = y.cpu()
        paths = []
        B, C = z.shape
        ts = _now_iso()

        # nhóm theo nhãn
        labels = torch.unique(y).tolist()
        for l in labels:
            idx = (y == l).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            Zi = z.index_select(dim=0, index=idx)  # (n, C)
            n = Zi.size(0)

            # đường dẫn shard
            client_dir = os.path.join(self.root, client_id, f"label_{l}")
            _ensure_dir(client_dir)
            fname = f"{ts}_e{epoch if epoch is not None else 'NA'}_s{step if step is not None else 'NA'}_n{n}_C{C}.pt"
            fpath = os.path.join(client_dir, fname)

            # lưu tensor
            _safe_torch_save(fpath, {"z": Zi})

            # ghi manifest
            rec = {
                "ts": ts,
                "client_id": client_id,
                "label": int(l),
                "path": os.path.relpath(fpath, self.root),
                "n": int(n),
                "C": int(C),
                "epoch": None if epoch is None else int(epoch),
                "step": None if step is None else int(step),
                "extra": extra or {},
            }
            with open(self.manifest_path, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

            paths.append(fpath)

        return paths


# ====== ZReader (Reader/Analyzer) ======
class ZReader:
    """
    Đọc manifest và cung cấp API đọc logits theo điều kiện.
    Có thể stream theo shard để tránh tràn RAM.
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

    def load_manifest(self):
        rows = []
        with open(self.manifest_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def query(self, *, client_id: Optional[str] = None, label: Optional[int] = None):
        rows = self.load_manifest()
        out = []
        for r in rows:
            if client_id is not None and r["client_id"] != client_id:
                continue
            if label is not None and int(r["label"]) != int(label):
                continue
            out.append(r)
        return out

    def iter_shards(self, *, client_id: Optional[str] = None, label: Optional[int] = None):
        """
        Trả về generator: (meta_dict, z_tensor [n, C])
        """
        for rec in self.query(client_id=client_id, label=label):
            fpath = os.path.join(self.root, rec["path"])
            obj = torch.load(fpath, map_location="cpu")
            z = obj["z"]  # (n, C)
            yield rec, z

    def load_all(self, *, client_id: Optional[str] = None, label: Optional[int] = None,
                 max_rows: Optional[int] = None):
        """
        Nối tất cả z thỏa điều kiện thành một tensor [N, C].
        Cẩn thận RAM nếu N lớn!
        """
        zs = []
        total = 0
        for _, z in self.iter_shards(client_id=client_id, label=label):
            if max_rows is not None and total >= max_rows:
                break
            if max_rows is not None and total + z.size(0) > max_rows:
                zs.append(z[: max_rows - total])
                total = max_rows
                break
            zs.append(z)
            total += z.size(0)
        if not zs:
            return torch.empty(0, 0)
        return torch.cat(zs, dim=0)

    def quick_stats(self, *, client_id: Optional[str] = None, label: Optional[int] = None):
        """
        Tính số shard, tổng N, và mean logits (nếu đủ nhỏ để gom).
        """
        metas = self.query(client_id=client_id, label=label)
        num_shards = len(metas)
        N = sum(m["n"] for m in metas)
        # Mean logits ước lượng theo shard (tránh load hết)
        mean = None
        total = 0
        for _, z in self.iter_shards(client_id=client_id, label=label):
            if mean is None:
                mean = z.float().sum(dim=0)
            else:
                mean += z.float().sum(dim=0)
            total += z.size(0)
        if mean is not None and total > 0:
            mean = (mean / total).tolist()
        return {"num_shards": num_shards, "total_rows": N, "mean_logits": mean}
