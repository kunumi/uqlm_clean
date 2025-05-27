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

import asyncio

import json
from uqlm.judges import LLMJudge
from uqlm.utils import ResponseGenerator
from uqlm.utils.response_generator import LLM


async def main():
    # This notebook generate results based on these input & using "exai-gpt-35-turbo-16k" model
    prompts = [
        "Which part of the human body produces insulin?",
        "What color are the two stars on the national flag of Syria",
        "How many 'm's are there in the word strawberry",
    ]

    original_llm = LLM(model_name="openai:gpt-4o-mini", logprobs=True)
    
    rg = ResponseGenerator(llm=original_llm, max_calls_per_min=250)
    generations = await rg.generate_responses(prompts=prompts, count=1)
    responses = generations["data"]["response"]

    judge = LLMJudge(llm=original_llm, max_calls_per_min=250)

    judge_result = await judge.judge_responses(prompts=prompts, responses=responses)

    extract_answer = judge._extract_answers(responses=judge_result["judge_responses"])

    store_results = {
    "prompts": prompts,
    "responses": responses,
    "judge_result": judge_result,
    "extract_answer": extract_answer,
    }
    results_file = "llmjudge_results_file.json"
    with open(results_file, "w") as f:
        json.dump(store_results, f)

if __name__ == '__main__':
    asyncio.run(main())