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

from uqlm.utils.dataloader import load_example_dataset
from uqlm.scorers import WhiteBoxUQ
from uqlm.utils.response_generator import LLM


async def main():
    # svamp dataset to be used as a prod dataset
    svamp = (
        load_example_dataset("svamp")
        .rename(columns={"question_concat": "question", "Answer": "answer"})[
            ["question", "answer"]
        ]
        .tail(5)
    )

    # Define prompts
    MATH_INSTRUCTION = (
        "When you solve this math problem only return the answer with no additional text.\n"
    )
    prompts = [MATH_INSTRUCTION + prompt for prompt in svamp.question]

    # This will be our main LLM for generation
    gpt = LLM(model_name="openai:gpt-4o-mini", logprobs=True)

    wbuq = WhiteBoxUQ(llm=gpt)

    results = await wbuq.generate_and_score(prompts=prompts)

    results_file = "whitebox_results_file.json"
    with open(results_file, "w") as f:
        json.dump(results.to_dict(), f)

if __name__ == '__main__':
    asyncio.run(main())
 