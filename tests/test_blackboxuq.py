# Copyright 2025 CVS Health and/or one of its affiliates
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

import pytest
import json
from uqlm.scorers import BlackBoxUQ
from uqlm.utils.response_generator import LLM

datafile_path = "tests/data/scorers/blackbox_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]
    
PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]
    
mock_object = LLM(model_name="openai:gpt-4o-mini", temperature=1.0)

@pytest.mark.asyncio
async def test_bbuq(monkeypatch):
    uqe = BlackBoxUQ(
        llm=mock_object,
        scorers=['noncontradiction', 'exact_match', 'semantic_negentropy'],
    )
    
    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = [None] * 5
        return MOCKED_RESPONSES
    
    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [[None] * 5] * 5
        return MOCKED_SAMPLED_RESPONSES
    
    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)
    
    results = await uqe.generate_and_score(
        prompts=PROMPTS, num_responses=5,
    )

    assert all(
        [
            results.data["exact_match"][i] == pytest.approx(data["exact_match"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["noncontradiction"][i]
            == pytest.approx(data["noncontradiction"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["semantic_negentropy"][i]
            == pytest.approx(data["semantic_negentropy"][i])
            for i in range(len(PROMPTS))
        ]
    )
    
    assert results.metadata == metadata
