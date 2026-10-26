"""CPU mock checks for Face Boost model lifetime; no model files or GPU needed."""

import ast
from contextlib import nullcontext, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union
import unittest
from unittest.mock import Mock, MagicMock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), path, "exec"), namespace)


class FaceBoostReuseTests(unittest.TestCase):
    def setUp(self):
        self.crop = np.zeros((2, 2, 3), dtype=np.uint8)
        self.restored = np.full_like(self.crop, 80)
        self.tensor = Mock()
        self.tensor.unsqueeze.return_value.to.return_value = self.tensor
        self.output = object()
        self.model = MagicMock(return_value=[self.output])
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.session = Mock()
        self.session.get_inputs.return_value = [SimpleNamespace(name="input"), SimpleNamespace(name="weight")]
        self.session.run.return_value = [[self.output]]
        self.ort = Mock()
        self.ort.InferenceSession.return_value = self.session
        self.torch = SimpleNamespace(no_grad=nullcontext, load=Mock(return_value={"params_ema": {}}),
                                     cuda=SimpleNamespace(empty_cache=Mock()))
        self.utils = SimpleNamespace(load_torch_file=Mock(return_value={}))
        self.loading = SimpleNamespace(load_state_dict=Mock(return_value=self.model))
        self.constructor = Mock(return_value=self.model)
        self.env = dict(np=np, sys=__import__("sys"), torch=self.torch, onnxruntime=self.ort,
                        cv2=SimpleNamespace(INTER_CUBIC=2, INTER_LINEAR=1, INTER_NEAREST=0,
                                            INTER_LANCZOS4=4, resize=lambda image, size, interpolation: image),
                        normalize=Mock(), img2tensor=Mock(return_value=self.tensor),
                        tensor2img=Mock(return_value=self.restored),
                        prepare_cropped_face=Mock(return_value="prepared"),
                        normalize_cropped_face=Mock(return_value=self.restored),
                        folder_paths=SimpleNamespace(get_full_path=lambda category, name: "/models/" + name),
                        model_management=SimpleNamespace(get_torch_device=lambda: "cpu"),
                        logger=Mock(), providers=["CPUExecutionProvider"],
                        comfy=SimpleNamespace(utils=self.utils), model_loading=self.loading,
                        ARCH_REGISTRY=SimpleNamespace(get=Mock(return_value=self.constructor)))
        load_functions("scripts/r_faceboost/restorer.py", {"_load_restorer", "get_restored_face"}, self.env)
        self.restore = self.env["get_restored_face"]

    def call(self, model="gfpgan.pth", cache=None):
        return self.restore(self.crop, model, 0.5, 0.7, model_cache=cache)

    def load_swappers(self):
        face = SimpleNamespace(bbox=[0, 0, 1, 1])
        swap_model = Mock()
        swap_model.get.side_effect = lambda image, target, source, **kwargs: (
            (self.crop.copy(), np.ones((2, 3))) if kwargs.get("paste_back") is False else self.crop.copy())
        self.boost = Mock(wraps=self.restore)
        env = dict(Union=Union, List=List, Image=SimpleNamespace(Image=object, fromarray=lambda image: image),
                   Face=object, np=np, os=__import__("os"), logger=Mock(),
                   cv2=SimpleNamespace(COLOR_RGB2BGR=1, COLOR_BGR2RGB=2, cvtColor=lambda image, code: image),
                   SOURCE_FACES=None, SOURCE_IMAGE_HASH=None, TARGET_FACES=None, TARGET_IMAGE_HASH=None,
                   TARGET_FACES_LIST=[], TARGET_IMAGE_LIST_HASH=[],
                   get_image_md5hash=lambda image: "hash", analyze_faces=lambda image: [face, face],
                   get_face_single=lambda image, faces, face_index=0, **kwargs: (face, 0, face_index),
                   getFaceSwapModel=lambda path: swap_model, insightface_path="/models", reswapper_path="/models",
                   hyperswap_path="/models", state=SimpleNamespace(interrupted=False),
                   model_management=SimpleNamespace(processing_interrupted=lambda: False),
                   progress_bar=lambda count: Mock(), progress_bar_reset=Mock(),
                   restorer=SimpleNamespace(get_restored_face=self.boost),
                   swapper=SimpleNamespace(in_swap=lambda image, crop, matrix: image))
        load_functions("scripts/reactor_swapper.py", {"swap_face", "swap_face_many"}, env)
        return env

    def test_one_load_for_faces_and_batch_then_fresh_load_next_execution(self):
        for name in ("swap_face", "swap_face_many"):
            with self.subTest(caller=name):
                self.loading.load_state_dict.reset_mock()
                env = self.load_swappers()
                images = self.crop if name == "swap_face" else [self.crop, self.crop, self.crop]
                args = dict(model="inswapper.onnx", face_model=object(), faces_index=[0, 1],
                            face_boost_enabled=True, face_restore_model="gfpgan.pth")
                env[name](None, images, **args)
                calls_per_execution = 2 if name == "swap_face" else 6
                self.assertEqual(self.boost.call_count, calls_per_execution)
                self.loading.load_state_dict.assert_called_once()
                first_cache = self.boost.call_args.kwargs["model_cache"]
                env[name](None, images, **args)
                self.assertEqual(self.loading.load_state_dict.call_count, 2)
                self.assertIsNot(first_cache, self.boost.call_args.kwargs["model_cache"])

    def test_each_backend_reuses_model_and_keeps_inference_arguments(self):
        for model in ("gfpgan.pth", "codeformer.pth", "gpen.onnx"):
            with self.subTest(model=model):
                cache = {}
                self.call(model, cache)
                result, scale = self.call(model, cache)
                self.assertEqual(len(cache), 1)
                np.testing.assert_array_equal(result, np.full_like(self.crop, 40))
                self.assertEqual(scale, 256)
        self.utils.load_torch_file.assert_called_once_with("/models/gfpgan.pth", safe_load=True)
        self.torch.load.assert_called_once_with("/models/codeformer.pth")
        self.constructor.assert_called_once()
        self.model.assert_called_with(self.tensor, w=0.7)
        self.ort.InferenceSession.assert_called_once_with("/models/gpen.onnx", providers=["CPUExecutionProvider"])
        self.assertEqual(self.session.run.call_count, 2)
        inputs = self.session.run.call_args.args[1]
        self.assertEqual(inputs["input"], "prepared")
        np.testing.assert_array_equal(inputs["weight"], np.array([1], dtype=np.double))

    def test_load_failure_falls_back_and_retries(self):
        self.loading.load_state_dict.side_effect = [RuntimeError("load failed"), self.model]
        cache = {}
        with redirect_stderr(StringIO()):
            result, scale = self.call(cache=cache)
        self.assertEqual(cache, {})
        self.env["tensor2img"].assert_called_with(self.tensor, rgb2bgr=True, min_max=(-1, 1))
        self.call(cache=cache)
        self.assertEqual(self.loading.load_state_dict.call_count, 2)
        self.assertEqual(len(cache), 1)

    def test_inference_failure_uses_existing_fallback(self):
        self.model.side_effect = RuntimeError("inference failed")
        with redirect_stderr(StringIO()):
            result, scale = self.call(cache={})
        self.env["tensor2img"].assert_called_once_with(self.tensor, rgb2bgr=True, min_max=(-1, 1))
        np.testing.assert_array_equal(result, np.full_like(self.crop, 40))

    def test_omitted_cache_and_changed_configuration_do_not_reuse(self):
        self.call()
        self.call()
        self.assertEqual(self.loading.load_state_dict.call_count, 2)
        cache = {}
        self.call(cache=cache)
        self.call("other.pth", cache)
        self.env["providers"] = ["OtherExecutionProvider"]
        self.call(cache=cache)
        self.env["model_management"].get_torch_device = lambda: "other-device"
        self.call(cache=cache)
        self.assertEqual(self.loading.load_state_dict.call_count, 6)

    def test_disabled_boost_and_no_model_do_not_load(self):
        env = self.load_swappers()
        for name in ("swap_face", "swap_face_many"):
            images = self.crop if name == "swap_face" else [self.crop]
            env[name](None, images, model=None)
            env[name](None, images, model="inswapper.onnx", face_model=object(), face_boost_enabled=False)
        self.boost.assert_not_called()
        self.loading.load_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
