"""Keep each DDP process's dynamic Python-module cache private."""
import os
import runpy
import tempfile

with tempfile.TemporaryDirectory(prefix="tibo-hf-modules-") as module_cache:
    os.environ["HF_MODULES_CACHE"] = module_cache
    runpy.run_module("soft_prompt_distillation", run_name="__main__")
