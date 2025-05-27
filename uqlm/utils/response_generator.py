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
import itertools
import time
from typing import Any, Dict, List, Optional, Tuple, Generator

import aisuite as ai

class LLM:

    def __init__(self, model_name: str, temperature: float = 0.7, logprobs: bool = False, provider_configs: dict = {}):
        self.model_name = model_name
        self.temperature = temperature
        self.client = ai.Client(provider_configs)
        self.n = 1  # Default value for n parameter
        self.logprobs = logprobs  # Default value for logprobs
    
    async def agenerate(self, messages: List[List[Dict[str, str]]]) -> Any:
        """
        Generates responses asynchronously using the aisuite client.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            logprobs=self.logprobs,
            n=self.n
        )
        return response


class ResponseGenerator:
    def __init__(
        self,
        llm: LLM = None,
        max_calls_per_min: Optional[int] = None,
        use_n_param: bool = False
    ) -> None:
        """
        Class for generating data from a provided set of prompts

        Parameters
        ----------
        llm : `LLM`, default=None
            A llm `LLM`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Used to control rate limiting

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `LLM`. Not compatible with all
            `LLM` classes. If used, it speeds up the generation process substantially when count > 1.
        """
        self.llm = llm
        self.use_n_param = use_n_param
        self.max_calls_per_min = max_calls_per_min

    async def generate_responses(
        self,
        prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
        count: int = 1,
    ) -> Dict[str, Any]:
        """
        Generates evaluation dataset from a provided set of prompts. For each prompt,
        `self.count` responses are generated.

        Parameters
        ----------
        prompts : list of strings
            List of prompts from which LLM responses will be generated

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        count : int, default=1
            Specifies number of responses to generate for each prompt.

        Returns
        -------
        dict
            A dictionary with two keys: 'data' and 'metadata'.

            'data' : dict
                A dictionary containing the prompts and responses.

                'prompt' : list
                    A list of prompts.
                'response' : list
                    A list of responses corresponding to the prompts.

            'metadata' : dict
                A dictionary containing metadata about the generation process.

                'temperature' : float
                    The temperature parameter used in the generation process.
                'count' : int
                    The count of prompts used in the generation process.
                'system_prompt' : str
                    The system prompt used for generating responses
        """
        assert isinstance(self.llm, LLM), """
            llm must be an instance of uqlm.utils.response_generator_aisuite.LLM
        """
        assert all(isinstance(prompt, str) for prompt in prompts), (
            "If using custom prompts, please ensure `prompts` is of type list[str]"
        )
        print(f"Generating {count} responses per prompt...")
        if self.llm.temperature == 0:
            assert count == 1, "temperature must be greater than 0 if count > 1"
        self._update_count(count)

        self.system_message = system_prompt if system_prompt else "You are a helpful assistant."

        generations, duplicated_prompts = await self._generate_in_batches(
            prompts=prompts
        )

        responses = generations["responses"]
        logprobs = generations["logprobs"]

        print("Responses successfully generated!")
        return {
            "data": {
                "prompt": self._enforce_strings(duplicated_prompts),
                "response": self._enforce_strings(responses),
            },
            "metadata": {
                "system_prompt": system_prompt,
                "temperature": self.llm.temperature,
                "count": self.count,
                "logprobs": logprobs,
            },
        }

    def _create_tasks(
        self,
        prompts: List[str],
    ) -> Tuple[List[Any], List[str]]:
        """
        Creates a list of async tasks and returns duplicated prompt list
        with each prompt duplicated `count` times
        """
        duplicated_prompts = [
            prompt for prompt, i in itertools.product(prompts, range(self.count))
        ]
        if self.use_n_param:
            tasks = [
                self._async_api_call(prompt=prompt, count=self.count)
                for prompt in prompts
            ]
        else:
            tasks = [
                self._async_api_call(prompt=prompt, count=1)
                for prompt in duplicated_prompts
            ]
        return tasks, duplicated_prompts

    def _update_count(self, count: int) -> None:
        """Updates self.count parameter and self.llm as necessary"""
        self.count = count
        if self.use_n_param:
            self.llm.n = count

    async def _generate_in_batches(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Executes async IO in batches to avoid rate limit error"""
        batch_size = (
            len(prompts)
            if not self.max_calls_per_min
            else self.max_calls_per_min // self.count
        )
        prompts_partition = self._split(prompts, batch_size)

        duplicated_prompts = []
        generations = {"responses": [], "logprobs": []}
        for prompt_batch in prompts_partition:
            start = time.time()
            # generate responses for current batch
            tasks, duplicated_batch_prompts = self._create_tasks(prompt_batch)
            generations_batch = await asyncio.gather(*tasks)
            responses_batch, logprobs_batch = [], []
            for g in generations_batch:
                responses_batch.extend(g["responses"])
                logprobs_batch.extend(g["logprobs"])

            # extend lists to include current batch
            duplicated_prompts.extend(duplicated_batch_prompts)
            generations["responses"].extend(responses_batch)
            generations["logprobs"].extend(logprobs_batch)
            stop = time.time()

            # pause if needed
            if (stop - start < 60) and (batch_size < len(prompts)):
                time.sleep(61 - stop + start)

        return generations, duplicated_prompts

    async def _async_api_call(self, prompt: str, count: int = 1) -> List[Any]:
        """Generates responses asynchronously using an RunnableSequence object"""

        messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        logprobs = [None] * count
        result = await self.llm.agenerate(messages)
        if hasattr(self.llm, "logprobs"):
            if self.llm.logprobs:
                if hasattr(result.choices[0], "logprobs"):
                    logprobs = [
                        result.choices[i].logprobs.content for i in range(count)
                    ]

        return {
            "logprobs": logprobs,
            "responses": [result.choices[i].message.content for i in range(count)],
        }

    @staticmethod
    def _enforce_strings(texts: List[Any]) -> List[str]:
        """Enforce that all outputs are strings"""
        return [str(r) for r in texts]

    @staticmethod
    def _split(list_a: List[str], chunk_size: int) -> Generator[List[str], None, None]:
        """Partitions list"""
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i : i + chunk_size]