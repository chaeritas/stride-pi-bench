"""A2A Agent wrapper for STRIDE Pi-Bench."""
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class Agent:
    """STRIDE-optimized policy compliance agent (A2A interface)."""

    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("STRIDE Agent analyzing policy compliance...")
        )

        # The actual policy compliance logic runs through Pi-Bench's orchestrator,
        # not through A2A message passing. This A2A server is for registration
        # and communication with the AgentBeats platform.
        #
        # When Pi-Bench evaluates this agent, it uses the STRIDEAgent class
        # from stride_agent.py directly via the --agent CLI flag.

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=f"STRIDE Agent received: {input_text[:200]}"))],
            name="Response",
        )
