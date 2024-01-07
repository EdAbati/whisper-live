from dataclasses import dataclass
from datetime import datetime


@dataclass
class Sentence:
    start_time: datetime
    end_time: datetime | None = None
    text: str = ""

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None

    def update_text(self, text: str) -> None:
        self.text = f"{self.text} {text}"
