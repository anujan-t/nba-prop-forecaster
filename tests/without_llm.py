"""Unit tests never reach a real language model. Importing this removes the language-model settings (GEMINI_API_KEY,
COURTSIDE_LLM_*, ...) from the test process, so a key set for the app, e.g. as a Windows user environment variable,
can't switch Coach to Gemini during the tests. Import it before backend.app, which builds an app when imported. The
opt-in live test (test_gemini_live) reads the key itself."""
import os

LLM_SETTINGS = ("GEMINI_API_KEY", "COURTSIDE_GEMINI_MODEL", "OPENAI_API_KEY")


def llm_settings() -> list[str]:
    return [name for name in os.environ if name.upper() in LLM_SETTINGS or name.upper().startswith("COURTSIDE_LLM_")]


for _name in llm_settings():
    del os.environ[_name]
