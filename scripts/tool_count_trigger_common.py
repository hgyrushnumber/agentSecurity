#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility shim: re-exports shared utilities from agents.common.

Kept so existing scripts can keep "from tool_count_trigger_common import ..."
unchanged. New code should import from agents.common directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.common import *  # noqa: F401,F403
from agents.common import __all__  # noqa: F401
from agents.common.tokenizer_utils import (  # noqa: F401
    apply_chat_template_ids,
    apply_chat_template_text,
    build_messages,
    choose_precision,
    load_tokenizer,
    model_input_device,
    normalize_token_ids,
)
