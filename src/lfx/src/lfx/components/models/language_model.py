from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from lfx.base.models.anthropic_constants import ANTHROPIC_MODELS
from lfx.base.models.google_generative_ai_constants import GOOGLE_GENERATIVE_AI_MODELS
from lfx.base.models.model import LCModelComponent
from lfx.base.models.openai_constants import OPENAI_CHAT_MODEL_NAMES, OPENAI_REASONING_MODEL_NAMES
from lfx.components.azure.azure_openai import AzureChatOpenAIComponent
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import BoolInput, MessageTextInput
from lfx.io import DropdownInput, MessageInput, MultilineInput, SecretStrInput, SliderInput
from lfx.schema.dotdict import dotdict


class LanguageModelComponent(LCModelComponent):
    display_name = "Language Model"
    description = "Runs a language model given a specified provider."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    priority = 0  # Set priority to 0 to make it appear first

    def set_attributes(self, params: dict) -> None:
        """Set component attributes with provider-specific defaults for api_key."""
        # Set default api_key value based on provider if not provided
        if "provider" in params and "api_key" not in params:
            provider = params["provider"]
            if provider == "OpenAI":
                params["api_key"] = "OPENAI_API_KEY"
            elif provider == "Azure OpenAI":
                params["api_key"] = "AZURE_OPENAI_API_KEY"
            elif provider == "Anthropic":
                params["api_key"] = "ANTHROPIC_API_KEY"
            elif provider == "Google":
                params["api_key"] = "GOOGLE_API_KEY"

        # Call parent set_attributes
        super().set_attributes(params)

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["OpenAI", "Azure OpenAI", "Anthropic", "Google"],
            value="OpenAI",
            info="Select the model provider",
            real_time_refresh=True,
            options_metadata=[
                {"icon": "OpenAI"},
                {"icon": "Azure"},
                {"icon": "Anthropic"},
                {"icon": "GoogleGenerativeAI"}
            ],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_CHAT_MODEL_NAMES[0],
            info="Select the model to use",
            real_time_refresh=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Model Provider API key",
            required=False,
            show=True,
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="azure_endpoint",
            display_name="Azure Endpoint",
            info="Your Azure endpoint, including the resource. Example: `https://example-resource.azure.openai.com/`",
            required=False,
            show=False,  # Will be shown only when Azure OpenAI is selected
        ),
        DropdownInput(
            name="api_version",
            display_name="API Version",
            options=sorted(AzureChatOpenAIComponent.AZURE_OPENAI_API_VERSIONS, reverse=True),
            value=next(
                (
                    version
                    for version in sorted(AzureChatOpenAIComponent.AZURE_OPENAI_API_VERSIONS, reverse=True)
                    if not version.endswith("-preview")
                ),
                AzureChatOpenAIComponent.AZURE_OPENAI_API_VERSIONS[0],
            ),
            info="Azure OpenAI API version",
            show=False,  # Will be shown only when Azure OpenAI is selected
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        model_name = self.model_name
        temperature = self.temperature
        stream = self.stream

        # Auto-resolve api_key if missing and provider is set
        if not self.api_key and provider:
            if provider == "OpenAI":
                self.api_key = "OPENAI_API_KEY"
            elif provider == "Azure OpenAI":
                self.api_key = "AZURE_OPENAI_API_KEY"
            elif provider == "Anthropic":
                self.api_key = "ANTHROPIC_API_KEY"
            elif provider == "Google":
                self.api_key = "GOOGLE_API_KEY"

            # Resolve environment variable
            if self.api_key and isinstance(self.api_key, str):
                import os
                env_value = os.getenv(self.api_key)
                if env_value:
                    self.api_key = env_value
                    self.log(f"Resolved environment variable {self.api_key} for {self.display_name} API key", "INFO")

        if provider == "OpenAI":
            if not self.api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)

            if model_name in OPENAI_REASONING_MODEL_NAMES:
                # reasoning models do not support temperature (yet)
                temperature = None

            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=stream,
                openai_api_key=self.api_key,
            )
        if provider == "Azure OpenAI":
            if not self.api_key:
                msg = "Azure OpenAI API key is required when using Azure OpenAI provider"
                raise ValueError(msg)
            if not self.azure_endpoint:
                msg = "Azure endpoint is required when using Azure OpenAI provider"
                raise ValueError(msg)

            return AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_deployment=model_name,
                api_version=self.api_version,
                temperature=temperature,
                streaming=stream,
                api_key=self.api_key,
            )
        if provider == "Anthropic":
            if not self.api_key:
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                anthropic_api_key=self.api_key,
            )
        if provider == "Google":
            if not self.api_key:
                msg = "Google API key is required when using Google provider"
                raise ValueError(msg)
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                google_api_key=self.api_key,
            )
        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "provider":
            if field_value == "OpenAI":
                build_config["model_name"]["options"] = OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_CHAT_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "OpenAI API Key"
                build_config["azure_endpoint"]["show"] = False
                build_config["api_version"]["show"] = False
            elif field_value == "Azure OpenAI":
                # Azure OpenAI supports the same models as OpenAI
                build_config["model_name"]["options"] = OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_CHAT_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "Azure OpenAI API Key"
                build_config["azure_endpoint"]["show"] = True
                build_config["api_version"]["show"] = True
            elif field_value == "Anthropic":
                build_config["model_name"]["options"] = ANTHROPIC_MODELS
                build_config["model_name"]["value"] = ANTHROPIC_MODELS[0]
                build_config["api_key"]["display_name"] = "Anthropic API Key"
                build_config["azure_endpoint"]["show"] = False
                build_config["api_version"]["show"] = False
            elif field_value == "Google":
                build_config["model_name"]["options"] = GOOGLE_GENERATIVE_AI_MODELS
                build_config["model_name"]["value"] = GOOGLE_GENERATIVE_AI_MODELS[0]
                build_config["api_key"]["display_name"] = "Google API Key"
                build_config["azure_endpoint"]["show"] = False
                build_config["api_version"]["show"] = False
        elif field_name == "model_name" and field_value.startswith("o1") and self.provider == "OpenAI":
            # Hide system_message for o1 models - currently unsupported
            if "system_message" in build_config:
                build_config["system_message"]["show"] = False
        elif field_name == "model_name" and not field_value.startswith("o1") and "system_message" in build_config:
            build_config["system_message"]["show"] = True
        return build_config
