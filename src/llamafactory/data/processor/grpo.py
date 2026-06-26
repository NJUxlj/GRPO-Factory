# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ..data_utils import Role
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class GRPODatasetProcessor(DatasetProcessor):
    r"""Build prompt-only inputs and preserve reference answers for reward scoring."""

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> list[int]:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        input_ids, _ = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        input_ids, _ = self.template.mm_plugin.process_token_ids(
            input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        source_len, _ = infer_seqlen(len(input_ids), 0, self.data_args.cutoff_len)
        return input_ids[:source_len]

    @staticmethod
    def _extract_ground_truth(response: list[dict[str, str]]) -> str:
        if len(response) == 0:
            return ""
        return response[0].get("content", "")

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids = self._encode_data_example(
                prompt=examples["_prompt"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["ground_truth"].append(self._extract_ground_truth(examples["_response"][i]))
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("ground_truth:\n{}".format(example["ground_truth"]))
