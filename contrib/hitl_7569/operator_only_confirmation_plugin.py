# Copyright 2026 Google LLC
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

"""Stop-gap for google/adk-python#7569 until #7134 lands.

Rewrites every tool confirmation answer that arrives on an invocation whose
``user_id`` is not an allowlisted operator to ``confirmed=False``, before the
runner stores the message. The rejected answer still resolves the pending
call, so the agent returns the normal rejection result instead of stalling,
which is what the reverted 9e9eaa6 got wrong.

Covers both places a confirmation answer enters the framework:

* the LlmAgent flow: a function response named ``adk_request_confirmation``;
* workflow tool and function nodes: a function response named
  ``adk_request_input`` whose id is ``wf_tool_confirmation:<node_path>``.

Over A2A on current main, ``user_id`` is the authenticated user name when the
A2A server authenticated the caller and ``A2A_USER_<context_id>`` otherwise,
so an allowlist of operator names cannot be satisfied by an unauthenticated
peer. It does NOT help on ``/run`` or ``/run_sse``, where ``user_id`` comes
from the request body; those need API-server authentication in front of them.
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.utils._function_call_names import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.genai import types

logger = logging.getLogger('google_adk.' + __name__)

_WORKFLOW_CONFIRMATION_PREFIX = 'wf_tool_confirmation:'


class OperatorOnlyConfirmationPlugin(BasePlugin):
  """Refuses tool confirmations from any user_id that is not an operator."""

  def __init__(
      self,
      operators: Iterable[str],
      name: str = 'operator_only_confirmation',
  ):
    super().__init__(name=name)
    self._operators = frozenset(operators)

  @staticmethod
  def _is_confirmation_answer(fr: types.FunctionResponse) -> bool:
    # The LlmAgent flow matches confirmation answers by name; the workflow
    # resume path keys resume inputs by id alone, so the id decides there and
    # the name is not consulted.
    if fr.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
      return True
    return (fr.id or '').startswith(_WORKFLOW_CONFIRMATION_PREFIX)

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    if invocation_context.user_id in self._operators:
      return None
    if not user_message or not user_message.parts:
      return None

    changed = False
    parts = []
    for part in user_message.parts:
      fr = part.function_response
      if fr is not None and self._is_confirmation_answer(fr):
        response = dict(fr.response) if isinstance(fr.response, dict) else {}
        if response.get('confirmed') is not False:
          response['confirmed'] = False
          part = part.model_copy(
              update={
                  'function_response': fr.model_copy(
                      update={'response': response}
                  )
              }
          )
          changed = True
          logger.warning(
              'Refusing tool confirmation %s from user_id %r: not an'
              ' allowlisted operator.',
              fr.id,
              invocation_context.user_id,
          )
      parts.append(part)

    if not changed:
      return None
    return user_message.model_copy(update={'parts': parts})
