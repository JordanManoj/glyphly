"""Convert an IndicPhotoOCR PARSeq checkpoint (Apache-2.0 weights) to ONNX for glyphly.

Not part of the package; run once per language by maintainers. Needs, in a separate
environment: torch, timm, pytorch-lightning, nltk, pyyaml, onnx, onnxruntime, and the
IndicPhotoOCR source (MIT) unpacked next to this file as IndicPhotoOCR-main/; the
checkpoint from https://github.com/anikde/STocr/releases/tag/V2.0.0 goes in ckpt/.

    python tools/convert_parseq.py malayalam   -> out/malayalam_parseq_int8.onnx

The int8 file is what glyphly downloads; its SHA-256 is pinned in glyphly/models.py.
Rebuilding gives the same fingerprint.

The checkpoint is loaded with torch's weights-only unpickler. Before that, the pickle's
globals are listed without executing anything and must be exactly the expected set
(tensor storage + an LR-scheduler reference), so a tampered file is refused.
"""

import io
import json
import pickletools
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.optim.swa_utils import SWALR

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "IndicPhotoOCR-main"))
from IndicPhotoOCR.utils.strhub.models.parseq.system import PARSeq  # noqa: E402

ALLOWED = {
    "collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage",
    "torch LongStorage", "__builtin__ getattr", "torch.optim.swa_utils SWALR",
}  # fmt: skip


def check_globals(path: Path) -> None:
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    ops = list(pickletools.genops(io.BytesIO(z.read(name))))
    found = {str(a) for o, a, _ in ops if o.name == "GLOBAL"}
    if any(o.name in ("STACK_GLOBAL", "INST", "OBJ") for o, _, _ in ops) or not found <= ALLOWED:
        raise SystemExit(f"{path.name}: unexpected pickle globals {sorted(found - ALLOWED)}")
    # getattr may only fetch SWALR._cosine_anneal / _linear_anneal (scheduler state)
    strings = [a for o, a, _ in ops if o.name in ("BINUNICODE", "SHORT_BINUNICODE")]
    if "__builtin__ getattr" in found and not ({"_cosine_anneal", "_linear_anneal"} & set(strings)):
        raise SystemExit(f"{path.name}: getattr used for something unexpected")


class Wrapped(torch.nn.Module):
    def __init__(self, system: PARSeq) -> None:
        super().__init__()
        self.system = system

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        m = self.system.model
        # max_length given -> no data-dependent early stop, so the loop traces cleanly
        return m.forward(self.system.tokenizer, images, m.max_label_length).softmax(-1)


def bool_where_to_logic(model: onnx.ModelProto) -> int:
    """ONNX Runtime has no Where for bool tensors: rewrite Where(c, x, y) on bools as
    Or(And(c, x), And(Not(c), y))."""
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]
    }
    types.update({i.name: i.data_type for i in model.graph.initializer})
    nodes, n = [], 0
    for node in model.graph.node:
        if node.op_type == "Where" and types.get(node.input[1]) == onnx.TensorProto.BOOL:
            c, x, y = node.input
            k = f"{node.name}_logic"
            nodes += [
                onnx.helper.make_node("And", [c, x], [f"{k}_a"], name=f"{k}_and1"),
                onnx.helper.make_node("Not", [c], [f"{k}_n"], name=f"{k}_not"),
                onnx.helper.make_node("And", [f"{k}_n", y], [f"{k}_b"], name=f"{k}_and2"),
                onnx.helper.make_node(
                    "Or", [f"{k}_a", f"{k}_b"], list(node.output), name=f"{k}_or"
                ),
            ]
            n += 1
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return n


def main(lang: str) -> None:
    ckpt = HERE / "ckpt" / f"{lang}.ckpt"
    check_globals(ckpt)
    with torch.serialization.safe_globals([getattr, SWALR]):
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    system = PARSeq(**ck["hyper_parameters"])
    system.load_state_dict(ck["state_dict"])
    system.eval()
    del ck
    H, W = system.hparams.img_size
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"{lang}_parseq.onnx"
    x = np.random.default_rng(0).uniform(-1, 1, (4, 3, H, W)).astype(np.float32)
    with torch.no_grad():
        ref = Wrapped(system)(torch.from_numpy(x)).numpy()
    dummy = torch.zeros(2, 3, H, W)
    with torch.no_grad():
        torch.onnx.export(
            Wrapped(system), (dummy,), str(path), dynamo=False, opset_version=17,
            input_names=["x"], output_names=["probs"],
            dynamic_axes={"x": {0: "batch"}, "probs": {0: "batch"}},
        )  # fmt: skip
    # characters: index 0 = EOS, then the charset (strhub Tokenizer: [E] + charset + [B], [P])
    itos = list(system.tokenizer._itos)
    model = onnx.load(str(path))
    print("bool Where nodes rewritten:", bool_where_to_logic(model))
    meta = model.metadata_props.add()
    meta.key, meta.value = "glyphly", json.dumps(
        {"kind": "parseq", "img_size": [H, W], "itos": itos, "eos": system.tokenizer.eos_id,
         "source": "IndicPhotoOCR (Bhashini, IIT Jodhpur) via anikde/STocr V2.0.0, Apache-2.0"},
        ensure_ascii=False,
    )  # fmt: skip
    onnx.save(model, str(path))
    q = out / f"{lang}_parseq_int8.onnx"
    quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
    m2 = onnx.load(str(q))
    m2.metadata_props.extend(model.metadata_props)
    onnx.save(m2, str(q))

    # check: the ONNX files give the same predictions as torch did before export
    for p in (path, q):
        got = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"x": x})[
            0
        ]
        same = (got.argmax(-1) == ref.argmax(-1)).mean()
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB, argmax agreement {same:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
