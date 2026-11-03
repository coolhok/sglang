"""Regression tests for multimodal inputs at the V4.1 model boundary."""

import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.runner.eager_runner as eager
import sglang.srt.models.deepseek_v4 as dsv4
from sglang.srt.layers.cp.base import disable_cp_strategy, init_cp_strategy
from sglang.srt.layers.cp.utils import prepare_cp_forward
from sglang.srt.layers.engram import engram_gate
from sglang.srt.managers.req_types import Modality, MultimodalDataItem
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepseekV41PrefillCPMultimodal(CustomTestCase):
    def test_continuation_image_embeddings_reach_the_correct_rank(self):
        """A cached prefix covers the first image row, so the shard uses row 2."""
        image_id = 5
        image_rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 2)],
            precomputed_embeddings=image_rows,
        )
        item.set_hash(12345)
        scheduler_ids = torch.tensor([item.pad_value, item.pad_value, 9, 10])
        batch = SimpleNamespace(
            input_ids=scheduler_ids.clone(),
            positions=torch.arange(1, 5),
            forward_mode=ForwardMode.EXTEND,
            mm_inputs=[MultimodalInputs(mm_items=[item])],
            extend_prefix_lens_cpu=[1],
            extend_seq_lens_cpu=[4],
            attn_cp_metadata=None,
        )
        embedding = torch.nn.Embedding(16, 2)

        def model_body(ids, positions, forward_batch, input_embeds):
            torch.testing.assert_close(ids, torch.tensor([image_id, 10]))
            torch.testing.assert_close(positions, torch.tensor([2, 4]))
            torch.testing.assert_close(
                forward_batch.input_ids_global,
                torch.tensor([image_id, 9, image_id, 10]),
            )
            torch.testing.assert_close(
                input_embeds, torch.stack([image_rows[2], embedding.weight[10]])
            )
            return input_embeds

        model = SimpleNamespace(
            vision=object(),
            config=SimpleNamespace(image_token_id=image_id),
            get_input_embeddings=lambda: embedding,
            get_image_feature=lambda _: [image_rows],
            model=model_body,
            pp_group=SimpleNamespace(is_last_rank=True),
            lm_head=None,
            logits_processor=lambda ids, *_: ids,
        )
        model.prepare_language_model_inputs = MethodType(
            dsv4.DeepseekV4ForCausalLM.prepare_language_model_inputs, model
        )
        model._prepare_mm_embeddings = MethodType(
            dsv4.DeepseekV4ForCausalLM._prepare_mm_embeddings, model
        )
        runner = eager.EagerRunner.__new__(eager.EagerRunner)
        runner.model_runner = SimpleNamespace(model=model)
        with (
            disable_cp_strategy(),
            get_parallel().override(
                attn_cp_rank=1, attn_cp_size=2, enable_cp_tp_group_sharing=False
            ),
            patch("torch.cuda.current_stream", return_value=None),
            patch.object(
                eager, "cp_gather_after_forward", return_value=torch.empty(4, 2)
            ),
            torch.no_grad(),
        ):
            init_cp_strategy(
                enable_prefill_cp=True, cp_size=2, cp_strategy="interleave"
            )
            prepare_cp_forward(batch)
            logits_ids = runner._execute_extend_cp(batch, {})
        torch.testing.assert_close(
            logits_ids, torch.tensor([image_id, image_id, 9, 10])
        )
        torch.testing.assert_close(batch.input_ids, scheduler_ids)


class TestDeepseekV41BCGMultimodal(CustomTestCase):
    def test_replay_uses_prepared_image_ids_without_mutating_scheduler_ids(self):
        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                scheduler_ids = torch.tensor([7, dsv4.MM_PAD_SHIFT_VALUE + 1, 9, 0])
                static_ids = scheduler_ids.clone()
                image_id = 42
                layer = SimpleNamespace(forward=None)
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                runner._is_full_backend = False
                runner.is_dsv41_model = True
                runner._qwen_bcg_hc_sidechannel = False
                runner._use_draft_input_embeds = False
                runner.layer_model = layer
                runner.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
                runner._prefill_forward_context = lambda *a, **kw: nullcontext()
                runner.backend = SimpleNamespace(
                    replay=lambda *a, **kw: static_ids.clone()
                )

                def outer_forward(ids, positions, batch):
                    model = SimpleNamespace(
                        vision=object(), config=SimpleNamespace(image_token_id=image_id)
                    )
                    mapped, _ = (
                        dsv4.DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, batch
                        )
                    )
                    if keyword:
                        return layer.forward(
                            input_ids=mapped, positions=positions, forward_batch=batch
                        )
                    return layer.forward(mapped, positions, batch)

                runner.model_runner = SimpleNamespace(
                    pp_group=SimpleNamespace(is_first_rank=True),
                    model=SimpleNamespace(forward=outer_forward),
                )
                live = SimpleNamespace(input_ids=scheduler_ids, mm_input_embeds=None)
                static = SimpleNamespace(
                    input_ids=static_ids,
                    positions=torch.arange(4),
                    mm_input_embeds=None,
                    mm_inputs=None,
                    forward_mode=ForwardMode.EXTEND,
                )
                actual = runner._execute_body_capture(live, static, 4, 3, None)
                torch.testing.assert_close(actual, torch.tensor([7, image_id, 9, 0]))
                self.assertEqual(scheduler_ids[1].item(), dsv4.MM_PAD_SHIFT_VALUE + 1)

    def test_image_engram_mask_does_not_depend_on_capture_batch_mm_metadata(self):
        image_id = 42
        ids = torch.tensor([7, image_id, 9])
        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        kv = torch.ones(3, 4)
        weight = torch.ones(1, 2)

        class Engram:
            layer_hash_index = 0

            def __call__(self, residual, *args, input_ids, **kwargs):
                return engram_gate(
                    residual, kv, weight, weight, 1e-6, 1e-6, input_ids, image_id
                )

        layer = SimpleNamespace(
            engram=Engram(),
            hc_cfg=None,
            forward_hc_pre_from_prev=lambda **kw: kw["state"],
        )
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=1),
            engram_hasher=lambda *a: torch.zeros((3, 1), dtype=torch.long),
            late_layer_start=None,
            decoder_replay_tail=None,
            start_layer=0,
            end_layer=1,
            layers=[layer],
            config=SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=1, image_token_id=image_id
            ),
        )
        expected = engram_gate(hidden, kv, weight, weight, 1e-6, 1e-6)
        self.assertFalse(torch.equal(expected[1], hidden[1]))
        expected[1] = hidden[1]
        for contains_mm in (False, True):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND, contains_mm_inputs=lambda: contains_mm
            )
            with (
                patch.object(dsv4, "is_cp_active", return_value=False),
                patch.object(dsv4, "is_in_breakable_cuda_graph", return_value=False),
                patch.object(
                    dsv4, "get_forward", return_value=SimpleNamespace(sp_active=False)
                ),
                patch.object(dsv4, "check_cuda_graph_backend", return_value=True),
                patch.object(dsv4, "_is_hip", False),
            ):
                actual, _, _ = dsv4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                    model, torch.arange(3), hidden, batch, ids, ids, False, []
                )
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
