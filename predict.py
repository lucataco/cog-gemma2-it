# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
import subprocess
from threading import Thread
from transformers import GemmaTokenizerFast, AutoModelForCausalLM
from transformers.generation.streamers import TextIteratorStreamer


# MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-2-9b-bf16/model.tar"
MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-2-9b-it-bf16/model.tar"
MODEL_CACHE = "checkpoints"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            MODEL_CACHE,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        self.model.config.sliding_window = 4096
        self.model.eval()
    
    def predict(
        self,
        prompt: str = Input(
            description="Prompt to send to the model.",
            default="Write me a poem about Machine Learning.",
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1, le=2048, default=1024,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.1,
            le=4.0,
            default=0.6,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.05,
            le=1.0,
            default=0.9,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=1,
            le=1000,
            default=50,
        ),
        repetition_penalty: float = Input(
            description="A parameter that controls how repetitive text can be. Lower means more repetitive, while higher means less repetitive. Set to 1.0 to disable.",
            ge=0.0,
            default=1.2,
        )
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        chat = [{ "role": "user", "content": prompt }]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **input_ids.to(self.model.device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=True,
        )
        with torch.inference_mode():
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()
            for new_token in streamer:
                yield new_token
            t.join()
