"""
Class-based evaluation subjects.

The harness evaluates *subjects*: anything that can respond to a prompt.
Two concrete kinds are supported:

- **Models** — HuggingFace causal LMs wrapped by :class:`HFModelGenerator`,
  legacy models + tokenizer managers wrapped by :class:`LegacyModelGenerator`,
  plus HuggingFace sequence classifiers wrapped by :class:`HFClassifierPredictor`.
- **Agents** — agent/chat backends wrapped by :class:`AgentGenerator` or
  :class:`CallableGenerator`.

Every subject derives from :class:`ResponseGenerator` (or
:class:`LabelPredictor` for classification) so the harness takes a single,
class-based interface regardless of whether a model or an agent is under test.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, cast

import torch

try:
    import transformers as _transformers  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _transformers = None  # type: ignore[assignment,misc]

_TRANSFORMERS_OK = _transformers is not None


def _accepts_kwarg(fn: Callable[..., Any], kwarg: str) -> bool:
    """Best-effort check that a callable can receive a keyword argument."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    for param in signature.parameters.values():
        if param.name == kwarg and param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.KEYWORD_ONLY,
        ):
            return True
    return any(
        param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        for param in signature.parameters.values()
    )


class ResponseGenerator(ABC):
    """
    Anything that can produce a text response to a prompt.

    Implement ``generate(prompt, max_new_tokens)``. Subclasses represent
    models and agents uniformly so the harness can evaluate either.
    """

    name: str = "subject"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Return a text response for the given prompt."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> Optional[int]:
        """Return an exact token count for ``text``, or None to estimate."""
        return None

    def cache_key(self) -> Dict[str, Any]:
        """
        Identify this subject for response caching.

        Everything that changes what :meth:`generate` returns belongs here.
        A subject that omits a decoding parameter will serve another
        configuration's responses from cache, so subclasses carrying their own
        settings must extend this rather than rely on the name alone.
        """
        return {"subject": self.name, "type": type(self).__name__}


class LegacyModelGenerator(ResponseGenerator):
    """
    ResponseGenerator backed by a legacy model + tokenizer manager.

    The tokenizer manager exposes ``encode``/``decode`` with ``add_bos`` /
    ``add_eos`` kwargs rather than the HuggingFace ``__call__`` interface, so
    this subject lets the harness route legacy models through the same unified
    subject path as :class:`HFModelGenerator` and agents.
    """

    def __init__(
        self,
        model,
        tokenizer_manager,
        name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer_manager = tokenizer_manager
        self.name = name or str(getattr(model, "_name_or_path", None) or "legacy_model")
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        input_ids = self.tokenizer_manager.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        generated = self.model.generate(
            input_tensor,
            max_length=len(input_ids) + max_new_tokens,
        )
        new_token_ids = generated[0][len(input_ids) :].tolist()
        return str(self.tokenizer_manager.decode(new_token_ids))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer_manager.encode(text, add_bos=False, add_eos=False))


class HFModelGenerator(ResponseGenerator):
    """ResponseGenerator backed by a HuggingFace causal LM + tokenizer."""

    def __init__(
        self,
        model,
        tokenizer,
        name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if tokenizer.pad_token is None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except AttributeError:
                pass
        self.model = model
        self.tokenizer = tokenizer
        model_name = getattr(model, "_name_or_path", None) or getattr(
            getattr(model, "config", None), "name_or_path", None
        )
        self.name = name or model_name or "hf_model"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(self.device)
        self.model.eval()
        self.model.to(self._torch_device)

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._torch_device)
        else:
            inputs = {key: value.to(self._torch_device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return str(self.tokenizer.decode(generated_ids, skip_special_tokens=True))


class CallableGenerator(ResponseGenerator):
    """ResponseGenerator backed by a plain ``fn(prompt, max_new_tokens) -> str``."""

    def __init__(
        self,
        fn: Callable[[str, int], str],
        name: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.name = name or str(getattr(fn, "__name__", "callable"))

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        fn = cast(Callable[..., Any], self.fn)
        if _accepts_kwarg(fn, "max_new_tokens"):
            return str(fn(prompt, max_new_tokens=max_new_tokens))
        return str(fn(prompt))


class AgentGenerator(ResponseGenerator):
    """
    ResponseGenerator backed by an agent object.

    The agent may expose one of ``respond``, ``chat``, ``invoke``, ``run``,
    ``generate``, or be callable itself. ``max_new_tokens`` is passed through
    only when the selected callable accepts it.
    """

    _METHOD_CANDIDATES = ("respond", "chat", "invoke", "run", "generate")

    def __init__(
        self,
        agent: Any,
        name: Optional[str] = None,
        method: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.name = name or str(getattr(agent, "name", None) or type(agent).__name__)
        self.method = method or self._detect_method()

    def _detect_method(self) -> Optional[str]:
        if callable(self.agent):
            return None
        for candidate in self._METHOD_CANDIDATES:
            fn = getattr(self.agent, candidate, None)
            if callable(fn):
                return candidate
        raise TypeError(
            f"Agent {type(self.agent).__name__} has no callable respond/chat/invoke/run/"
            f"generate method and is not itself callable"
        )

    def _resolve_callable(self) -> Callable[..., Any]:
        if self.method is None:
            return cast(Callable[..., Any], self.agent)
        return cast(Callable[..., Any], getattr(self.agent, self.method))

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        fn = self._resolve_callable()
        if _accepts_kwarg(fn, "max_new_tokens"):
            return str(fn(prompt, max_new_tokens=max_new_tokens))
        return str(fn(prompt))


class LabelPredictor(ABC):
    """
    Anything that maps texts to class labels.

    ``predict(texts)`` must return one prediction per text, where each
    prediction is an int label, a ``(label, confidence)`` tuple, or a dict
    with ``label``/``confidence`` keys.
    """

    name: str = "classifier"

    @abstractmethod
    def predict(self, texts: List[str]) -> List[Any]:
        """Return predictions for a list of texts."""
        raise NotImplementedError


class CallablePredictor(LabelPredictor):
    """LabelPredictor backed by a plain ``fn(texts) -> List[Any]``."""

    def __init__(
        self,
        fn: Callable[[List[str]], List[Any]],
        name: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.name = name or str(getattr(fn, "__name__", "classifier"))

    def predict(self, texts: List[str]) -> List[Any]:
        return list(self.fn(texts))


class HFClassifierPredictor(LabelPredictor):
    """LabelPredictor backed by a HuggingFace sequence classification model."""

    def __init__(
        self,
        model_path: str,
        name: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        if not _TRANSFORMERS_OK:
            raise ImportError("HFClassifierPredictor requires transformers")
        assert _transformers is not None
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = name or model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(self.device)
        self.tokenizer = _transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = _transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self._torch_device)

    def predict(self, texts: List[str]) -> List[Any]:
        predictions: List[Any] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._torch_device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu().tolist()
                confs = torch.max(probs, dim=-1)[0].cpu().tolist()
            predictions.extend((int(pred), float(conf)) for pred, conf in zip(preds, confs))
        return predictions


class ApiGenerator(ResponseGenerator):
    """
    ResponseGenerator backed by a hosted chat-completion API.

    ``chat(messages, max_tokens, temperature)`` returns a plain string (or an
    object whose string form is the response). Subclasses implement
    :meth:`chat` and optionally :meth:`chat_kwargs` for provider-specific
    parameters. The provider client is imported lazily so the package stays
    importable without the provider SDK installed.
    """

    _provider_import: tuple  # (module, attr) resolved in __init__

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.client = self._build_client()

    def _build_client(self) -> Any:
        raise NotImplementedError

    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> Any:
        raise NotImplementedError

    def cache_key(self) -> Dict[str, Any]:
        key = super().cache_key()
        key.update(
            {
                "model": self.model,
                "temperature": self.temperature,
                "base_url": self.base_url,
            }
        )
        return key

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, max_tokens=max_new_tokens)
        if isinstance(response, str):
            return response.strip()
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()
        return str(response).strip()


class OpenAIGenerator(ApiGenerator):
    """ResponseGenerator backed by the OpenAI chat completions API."""

    _provider_import = ("openai", "OpenAI")

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenAIGenerator requires the 'openai' package; install it to use this subject"
            ) from exc
        kwargs: Dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> Any:
        kwargs = self.chat_kwargs(max_tokens=max_tokens)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return completion.choices[0].message.content

    def chat_kwargs(self, max_tokens: int) -> Dict[str, Any]:
        return {"max_tokens": max_tokens, "temperature": self.temperature}


class AnthropicGenerator(ApiGenerator):
    """ResponseGenerator backed by the Anthropic Messages API."""

    _provider_import = ("anthropic", "Anthropic")

    def _build_client(self) -> Any:
        try:
            from anthropic import Anthropic  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AnthropicGenerator requires the 'anthropic' package; install it to use this subject"
            ) from exc
        kwargs: Dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return Anthropic(**kwargs)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> Any:
        kwargs = self.chat_kwargs(max_tokens=max_tokens)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    def chat_kwargs(self, max_tokens: int) -> Dict[str, Any]:
        return {"temperature": self.temperature}


class OllamaGenerator(ApiGenerator):
    """ResponseGenerator backed by a local Ollama chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            name=name or model,
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    def _build_client(self) -> Any:
        try:
            import ollama  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OllamaGenerator requires the 'ollama' package; install it to use this subject"
            ) from exc
        client = ollama.Client(host=self.base_url)
        return client

    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> Any:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"num_predict": max_tokens, "temperature": self.temperature},
        )
        return response["message"]["content"]


__all__ = [
    "AgentGenerator",
    "AnthropicGenerator",
    "ApiGenerator",
    "CallableGenerator",
    "CallablePredictor",
    "HFClassifierPredictor",
    "HFModelGenerator",
    "LabelPredictor",
    "LegacyModelGenerator",
    "OllamaGenerator",
    "OpenAIGenerator",
    "ResponseGenerator",
]
