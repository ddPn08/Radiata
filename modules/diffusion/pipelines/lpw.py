import re
from typing import *

import torch
from transformers import CLIPTokenizer

from modules.logger import logger

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt(prompt):
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(prompt):
        prompt = m.group(0)
        weight = m.group(1)

        if prompt.startswith("\\"):
            res.append([prompt[1:], 1.0])
        elif prompt == "(":
            round_brackets.append(len(res))
        elif prompt == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif prompt == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif prompt == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([prompt, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def pad_tokens_and_weights(
    tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77
):
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = (
        max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    )
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][
                        j
                        * (chunk_length - 2) : min(
                            len(weights[i]), (j + 1) * (chunk_length - 2)
                        )
                    ]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


class LongPromptWeightingPipeline:
    def __init__(
        self,
        text_encoder,
        tokenizer: CLIPTokenizer,
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = torch.device("cuda")

    def get_unweighted_text_embeddings(
        self,
        text_input: torch.Tensor,
        chunk_length: int,
        no_boseos_middle: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        When the length of tokens is a multiple of the capacity of the text encoder,
        it should be split into chunks and sent to the text encoder individually.
        """
        max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
        if max_embeddings_multiples > 1:
            text_embeddings = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = text_input[
                    :, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2
                ].clone()

                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = text_input[0, 0]
                text_input_chunk[:, -1] = text_input[0, -1]
                text_embedding = self.text_encoder(text_input_chunk)[0]

                if no_boseos_middle:
                    if i == 0:
                        # discard the ending token
                        text_embedding = text_embedding[:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        # discard the starting token
                        text_embedding = text_embedding[:, 1:]
                    else:
                        # discard both starting and ending tokens
                        text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat(text_embeddings, axis=1)
        else:
            text_embeddings = self.text_encoder(text_input)[0]
        return text_embeddings

    def get_prompts_with_weights(self, prompt: List[str], max_length: int):
        tokens = []
        weights = []
        truncated = False
        for text in prompt:
            texts_and_weights = parse_prompt(text)
            text_token = []
            text_weight = []
            for word, weight in texts_and_weights:
                token = self.tokenizer(
                    word,
                ).input_ids[1:-1]

                text_token += token
                text_weight += [weight] * len(token)
                if len(text_token) > max_length:
                    truncated = True
                    break

            if len(text_token) > max_length:
                truncated = True
                text_token = text_token[:max_length]
                text_weight = text_weight[:max_length]
            tokens.append(text_token)
            weights.append(text_weight)
        if truncated:
            logger.warning(
                "Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples"
            )

        return tokens, weights

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
        num_images_per_prompt: Optional[int] = 1,
        max_embeddings_multiples: Optional[int] = 3,
        **kwargs,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        batch_size = len(prompt)

        no_boseos_middle = False

        max_length = (
            self.tokenizer.model_max_length - 2
        ) * max_embeddings_multiples + 2

        prompt_tokens, prompt_weights = self.get_prompts_with_weights(
            prompt, max_length - 2
        )
        uncond_tokens, uncond_weights = self.get_prompts_with_weights(
            negative_prompt, max_length - 2
        )

        max_length = max(max_length, max([len(token) for token in uncond_tokens]))
        max_embeddings_multiples = min(
            max_embeddings_multiples,
            (max_length - 1) // (self.tokenizer.model_max_length - 2) + 1,
        )
        max_embeddings_multiples = max(1, max_embeddings_multiples)
        max_length = (
            self.tokenizer.model_max_length - 2
        ) * max_embeddings_multiples + 2

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id

        prompt_tokens, prompt_weights = pad_tokens_and_weights(
            prompt_tokens,
            prompt_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=self.tokenizer.model_max_length,
        )
        text_input_ids = torch.tensor(
            prompt_tokens, dtype=torch.int32, device=self.device
        )

        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=self.tokenizer.model_max_length,
        )
        uncond_input_ids = torch.tensor(
            uncond_tokens, dtype=torch.int32, device=self.device
        )

        text_embeddings = self.get_unweighted_text_embeddings(
            text_input_ids,
            self.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        seq_len = text_embeddings.shape[1]
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        uncond_embeddings = self.get_unweighted_text_embeddings(
            uncond_input_ids,
            self.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        prompt_weights = torch.tensor(
            prompt_weights, dtype=text_embeddings.dtype, device=self.device
        )
        uncond_weights = torch.tensor(
            uncond_weights,
            dtype=text_embeddings.dtype,
            device=self.device,
        )

        previous_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        )
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        )
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

        previous_mean = (
            uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
        )
        uncond_embeddings *= uncond_weights.unsqueeze(-1)
        current_mean = (
            uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
        )
        uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
