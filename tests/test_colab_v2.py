"""Validate the notebook without Colab, downloads, credentials or real training."""
import ast
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "COLAB_QUERY2LABEL_V2.ipynb").read_text(encoding="utf-8"))


class ColabV2Tests(unittest.TestCase):
    def test_cells_parse_and_expensive_operations_default_disabled(self):
        self.assertEqual(NOTEBOOK["nbformat"], 4)
        defaults = {}
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith("RUN_"):
                            defaults[target.id] = node.value.value
        self.assertEqual(defaults, {name: False for name in (
            "RUN_DEPTH_PREPARATION", "RUN_TRAINING", "RUN_CALIBRATION", "RUN_TEST", "RUN_INFERENCE")})

    def extraction_functions(self, destination):
        source = next("".join(cell["source"]) for cell in NOTEBOOK["cells"]
                      if cell["cell_type"] == "code" and "def extract_zip_safe" in "".join(cell["source"]))
        tree = ast.parse(source)
        module = ast.Module(body=[node for node in tree.body
                                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
                            type_ignores=[])
        namespace = {"Path": Path, "zipfile": zipfile, "shutil": shutil, "DATA_PARENT": destination}
        exec(compile(module, "notebook_zip_helpers", "exec"), namespace)
        return namespace

    def test_windows_zip_names_and_root_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            destination = directory / "extracted"
            archive = directory / "windows.zip"
            files = ("metadata/dish_metadata_cafe1.csv", "metadata/ingredients_metadata.csv",
                     "dish_ids/splits/rgb_train_ids.txt", "imagery/realsense_overhead/dish_1/rgb.png")
            with zipfile.ZipFile(archive, "w") as target:
                for relative in files:
                    target.writestr(("nutrition5k_dataset/" + relative).replace("/", "\\"), "example")
            namespace = self.extraction_functions(destination)
            namespace["extract_zip_safe"](archive, destination)
            self.assertEqual(namespace["dataset_roots"](), [destination / "nutrition5k_dataset"])
            self.assertTrue((destination / "nutrition5k_dataset" / files[0]).is_file())

    def test_zip_traversal_rejected_before_any_files_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            destination = directory / "extracted"
            archive = directory / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("safe.txt", "safe")
                target.writestr("..\\outside.txt", "unsafe")
            namespace = self.extraction_functions(destination)
            with self.assertRaises(ValueError):
                namespace["extract_zip_safe"](archive, destination)
            self.assertFalse((directory / "outside.txt").exists())
            self.assertFalse((destination / "safe.txt").exists())


if __name__ == "__main__":
    unittest.main()
