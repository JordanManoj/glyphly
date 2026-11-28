"""Train glyphly's script identifier: a line image -> which writing system it is in.

Not shipped; run by maintainers in an environment with torch and onnx:

    python tools/train_script_id.py          -> bench/data/script_id/script_id.onnx

Data: bench/data/script_id/lines/<script>/*.png (tools/script_id_data.py), plus lines cut
from the development half of the real-scan benchmark (bench/data/real, `split == "dev"`),
labelled by the page's language. The test half is never used here; `evaluate` measures
on it.
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data" / "script_id"
REAL = ROOT / "bench" / "data" / "real"
HEIGHT = 32
WINDOW = 192  # training crop width
# language of a real-scan page -> script class
LANG_SCRIPT = {
    "en": "latin", "fr": "latin", "de": "latin", "es": "latin", "it": "latin", "pt": "latin",
    "nl": "latin", "pl": "latin", "sv": "latin", "vi": "latin", "la": "latin",
    "ru": "cyrillic", "uk": "cyrillic", "be": "cyrillic", "sr": "cyrillic", "el": "greek",
    "ar": "arabic", "fa": "arabic", "he": "hebrew", "hi": "devanagari", "mr": "devanagari",
    "sa": "devanagari", "bn": "bengali", "as": "bengali", "or": "odia", "pa": "gurmukhi",
    "gu": "gujarati", "ta": "tamil", "te": "telugu", "kn": "kannada", "ml": "malayalam",
    "zh": "han", "ko": "hangul", "th": "thai",
}  # fmt: skip


def to_input(img: Image.Image) -> np.ndarray:
    """Grayscale, HEIGHT pixels high, [0, 1] with ink dark -> (1, H, W) float32."""
    g = img.convert("L")
    w = max(8, round(g.width * HEIGHT / max(1, g.height)))
    g = g.resize((min(w, 1024), HEIGHT), Image.Resampling.BILINEAR)
    return (np.asarray(g, dtype=np.float32) / 255.0)[None]


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("L")
    if rng.random() < 0.3:  # low resolution scan
        f = rng.uniform(0.35, 0.8)
        small = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))))
        img = small.resize(img.size)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-3, 3), expand=True, fillcolor=255)
    a = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.5:  # faint ink, grey paper
        lo, hi = rng.uniform(0, 90), rng.uniform(170, 255)
        a = lo + a / 255.0 * (hi - lo)
    if rng.random() < 0.4:
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 18), a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(25, 70))
        img = Image.open(io.BytesIO(buf.getvalue()))
    return img


def window(x: np.ndarray, rng: random.Random | None) -> np.ndarray:
    """A WINDOW-wide piece of the line (random in training), white-padded if shorter."""
    w = x.shape[2]
    if w >= WINDOW:
        s = rng.randrange(0, w - WINDOW + 1) if rng else (w - WINDOW) // 2
        return x[:, :, s : s + WINDOW]
    out = np.ones((1, HEIGHT, WINDOW), dtype=np.float32)
    out[:, :, :w] = x
    return out


class ScriptNet(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()

        def block(i: int, o: int, pool: tuple[int, int]) -> list[nn.Module]:
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool)]

        self.features = nn.Sequential(
            *block(1, 32, (2, 2)), *block(32, 64, (2, 2)), *block(64, 128, (2, 2)),
            *block(128, 160, (2, 1)),
        )  # fmt: skip
        self.head = nn.Linear(160, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(1.0 - x)  # ink as signal, paper as zero
        return self.head(f.mean(dim=(2, 3)))  # any line width


def real_lines(split: str) -> list[tuple[Image.Image, str]]:
    """Lines from real-scan pages of one split, labelled by the page's script."""
    sys.path.insert(0, str(ROOT / "src"))
    from glyphly import Reader
    from glyphly.detect import detect
    from glyphly.engine import cut, to_rgb

    reader = Reader(download=False)
    out = []
    for index in sorted(REAL.glob("*/index.jsonl")):
        for line in index.read_text("utf-8").splitlines():
            row = json.loads(line)
            script = LANG_SCRIPT.get(row.get("lang", ""))
            if row.get("rejected") or row.get("split") != split or not script:
                continue
            rgb = to_rgb(Image.open(index.parent / f"{row['id']}.jpg"))
            boxes = sorted(
                detect(reader._det, rgb, max_side=reader.max_side), key=lambda b: -b.length
            )
            for b in boxes[:12]:  # the longest lines: mostly body text in the page's script
                crop = cut(rgb, b)
                if crop.shape[1] >= 4 * crop.shape[0]:
                    out.append((Image.fromarray(np.ascontiguousarray(crop)), script))
    return out


def main() -> None:
    torch.manual_seed(0)
    rng = random.Random(0)
    classes = sorted(p.name for p in (DATA / "lines").iterdir() if p.is_dir())
    train: list[tuple[Image.Image, int]] = []
    val: list[tuple[Image.Image, int]] = []
    for c, script in enumerate(classes):
        for path in sorted((DATA / "lines" / script).glob("*.png")):
            page = int(path.stem[1:4])
            (val if page % 5 == 0 else train).append((Image.open(path).copy(), c))
    real = [(img, classes.index(s)) for img, s in real_lines("dev") if s in classes]
    train += real * 2  # real scans count double: they are what the model will see
    print(f"{len(classes)} classes, {len(train)} train lines ({len(real)} real), {len(val)} val")

    model = ScriptNet(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 14
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 2e-3, total_steps=epochs * (len(train) // 64 + 1)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    ckpt = DATA / "script_id.ckpt"  # saved after every epoch, so a stopped run resumes
    start = 0
    if ckpt.exists():
        state = torch.load(ckpt, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["epoch"]
        rng.seed(start)
        print(f"resuming after epoch {start}")
    for epoch in range(start, epochs):
        model.train()
        rng.shuffle(train)
        total = 0.0
        for b in range(0, len(train), 64):
            batch = train[b : b + 64]
            x = torch.from_numpy(
                np.stack([window(to_input(augment(i, rng)), rng) for i, _ in batch])
            )
            y = torch.tensor([c for _, c in batch])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(batch)
        model.eval()
        with torch.no_grad():
            right = sum(
                int(model(torch.from_numpy(to_input(i)[None])).argmax()) == c for i, c in val
            )
        acc = right / len(val)
        print(
            f"epoch {epoch + 1}: loss {total / len(train):.3f}, val accuracy {acc:.3f}", flush=True
        )
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(),
             "sched": sched.state_dict(), "epoch": epoch + 1},
            ckpt,
        )  # fmt: skip

    out = DATA / "script_id.onnx"
    model.eval()
    torch.onnx.export(
        model, (torch.ones(2, 1, HEIGHT, 256),), str(out), dynamo=False, opset_version=17,
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch", 3: "width"}, "logits": {0: "batch"}},
    )  # fmt: skip
    import onnx

    m = onnx.load(str(out))
    meta = m.metadata_props.add()
    meta.key, meta.value = (
        "glyphly",
        json.dumps({"kind": "script_id", "classes": classes, "height": HEIGHT}),
    )
    onnx.save(m, str(out))
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB)")


def evaluate() -> None:
    """Accuracy on lines of the real-scan TEST pages (never trained on), per script."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(DATA / "script_id.onnx"), providers=["CPUExecutionProvider"])
    classes = json.loads(sess.get_modelmeta().custom_metadata_map["glyphly"])["classes"]
    per: dict[str, list[int]] = {}
    for img, script in real_lines("test"):
        logits = sess.run(None, {"x": to_input(img)[None]})[0][0]
        per.setdefault(script, []).append(int(classes[int(logits.argmax())] == script))
    for script, hits in sorted(per.items()):
        print(f"{script:11s} {sum(hits) / len(hits):6.1%}  ({len(hits)} lines)")
    allhits = [h for hs in per.values() for h in hs]
    print(f"{'ALL':11s} {sum(allhits) / len(allhits):6.1%}  ({len(allhits)} lines)")


if __name__ == "__main__":
    evaluate() if sys.argv[1:] == ["evaluate"] else main()
