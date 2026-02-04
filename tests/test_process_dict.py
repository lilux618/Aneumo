import importlib.util
from pathlib import Path
import unittest


def load_process_dict_module():
    module_path = Path(__file__).resolve().parents[1] / "cfd_opt_deeponet" / "utils" / "process_dict.py"
    spec = importlib.util.spec_from_file_location("process_dict", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProcessStateDict(unittest.TestCase):
    def test_removes_orig_mod_prefix(self):
        module = load_process_dict_module()
        state_dict = {
            "_orig_mod.layer.weight": 1,
            "_orig_mod.layer.bias": 2,
            "layer.running_mean": 3,
        }

        processed = module.process_state_dict(state_dict)

        self.assertEqual(
            processed,
            {
                "layer.weight": 1,
                "layer.bias": 2,
                "layer.running_mean": 3,
            },
        )

    def test_does_not_mutate_original(self):
        module = load_process_dict_module()
        state_dict = {"_orig_mod.layer.weight": 1}

        processed = module.process_state_dict(state_dict)

        self.assertIn("_orig_mod.layer.weight", state_dict)
        self.assertNotIn("_orig_mod.layer.weight", processed)


if __name__ == "__main__":
    unittest.main()
