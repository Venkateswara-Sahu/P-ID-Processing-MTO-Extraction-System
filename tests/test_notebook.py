"""
The Colab notebook is a deliverable, so it needs guarding.

It must:
  * contain no API key,
  * embed the repository's tools byte-for-byte (so it runs without the
    repo being pushed),
  * use the training arguments that fix the measured defects,
  * never fall back to the training set for validation.
"""
import ast
import base64
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "01_yolo_training.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NB.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_sources(notebook):
    out = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            out.append((i, "".join(cell["source"])))
    return out


def test_the_notebook_exists_and_has_cells(notebook):
    assert len(notebook["cells"]) > 15
    assert notebook["nbformat"] == 4


def test_the_notebook_contains_no_api_key(notebook):
    """The old notebook had a live Roboflow key in plaintext."""
    raw = NB.read_text(encoding="utf-8")
    # Never hardcode a literal key here: that would re-publish the secret.
    assert "api_key=\"" not in raw
    assert "api_key='" not in raw
    # It must read the key from secrets or the environment instead.
    assert "ROBOFLOW_API_KEY" in raw


def test_every_code_cell_parses(code_sources):
    for index, src in code_sources:
        if src.lstrip().startswith("!") or "!" == src.strip()[:1]:
            continue  # Colab shell magic, not Python
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"cell {index} does not parse: {exc}\n{src[:300]}")


def test_the_tools_are_embedded_and_match_the_repository(notebook):
    """
    The notebook must run in Colab without the repo being pushed, so it
    embeds tools/*.py. They must be byte-identical, or the notebook
    silently runs a different tool than the one in the repo.
    """
    found = None
    for cell in notebook["cells"]:
        src = "".join(cell["source"])
        if "TOOLS_B64" in src and "write_bytes" in src:
            found = src
            break
    assert found is not None, "the notebook does not embed any tools"

    namespace: dict = {}
    exec(found, namespace)
    embedded = namespace["TOOLS_B64"]
    assert embedded, "no tools embedded"

    for name, blob in embedded.items():
        repo_file = ROOT / "tools" / name
        assert repo_file.exists(), f"{name} is embedded but not in tools/"
        decoded = base64.b64decode(blob)
        assert decoded == repo_file.read_bytes(), (
            f"the embedded {name} differs from tools/{name} - "
            "re-run python tools/build_notebook.py"
        )


def test_the_training_arguments_fix_the_measured_defects(code_sources):
    """
    These are the specific defects the audit found. If a default slips
    back in, the training silently reproduces the old problems.
    """
    train_src = ""
    constants: dict[str, object] = {}
    for _, src in code_sources:
        # Collect the module-level constants the train() call refers to.
        for m in re.finditer(r"^([A-Z_]{2,})\s*=\s*(.+)$", src, re.M):
            try:
                constants[m.group(1)] = ast.literal_eval(m.group(2).strip())
            except (ValueError, SyntaxError):
                pass
        if "model.train(" in src:
            train_src = src
    assert train_src, "no training cell found"

    def arg(name):
        m = re.search(rf"{name}\s*=\s*([^,\n]+)", train_src)
        if not m:
            return None
        value = m.group(1).strip()
        if value in constants:
            return constants[value]
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    # erasing=0.4 DELETES 4% of every image, which deletes symbols.
    assert arg("erasing") == 0.0, (
        "erasing must be 0.0 - the default 0.4 erases symbols"
    )
    # mosaic blurs fine line work; close it for the last epochs.
    assert arg("close_mosaic") is not None, "close_mosaic is missing"
    assert int(arg("close_mosaic")) >= 10
    # a long enough schedule for a 229-class problem
    assert int(arg("epochs")) >= 200
    assert int(arg("patience")) >= 30
    # resolution: 44 different source image sizes
    assert int(arg("imgsz")) >= 1024
    # reproducibility
    assert arg("seed") is not None
    # randaugment is for natural images
    assert arg("auto_augment") is None


def test_the_validation_never_falls_back_to_the_training_set(code_sources):
    """
    The old notebook set `val = train_path` when no val split was
    found, which meant the reported mAP was measured on training data.
    """
    for _, src in code_sources:
        assert "val = train_path" not in src
        assert "val_path or train_path" not in src
        assert "data_config['val'] = val_path or train_path" not in src


def test_the_checkpoint_gets_its_training_record_re_attached(code_sources):
    """The shipped best.pt had no args, no epoch and no metrics."""
    joined = "\n".join(s for _, s in code_sources)
    assert "train_args" in joined
    assert "class_health" in joined
    assert "torch.save" in joined


def test_the_per_class_evaluation_is_present(code_sources):
    joined = "\n".join(s for _, s in code_sources)
    assert "ap_class_index" in joined
    assert "never evaluated" in joined
    assert "map50" in joined


def test_the_notebook_reports_how_many_classes_were_evaluated(code_sources):
    """
    The whole point of the rewrite: the old notebook printed an mAP that
    hid 76 unevaluated classes.
    """
    joined = "\n".join(s for _, s in code_sources)
    assert "classes EVALUATED" in joined
    assert "classes never evaluated" in joined
