from dataclasses import dataclass, field

from ai.step_structurer import StructuredDoc
from ai.screenshot_describer import ScreenshotDescription


@dataclass
class MergedStep:
    title: str
    instruction: str
    screenshot: str | None = None
    image_title: str | None = None
    image_description: str | None = None


@dataclass
class MergedDoc:
    title: str
    steps: list[MergedStep] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "steps": [
                {
                    "title": s.title,
                    "instruction": s.instruction,
                    "screenshot": s.screenshot,
                    "image_title": s.image_title,
                    "image_description": s.image_description,
                }
                for s in self.steps
            ],
        }


class DocMerger:
    @staticmethod
    def merge(doc: StructuredDoc,
              descriptions: dict[str, ScreenshotDescription]) -> MergedDoc:
        steps: list[MergedStep] = []
        for s in doc.steps:
            desc = descriptions.get(s.screenshot) if s.screenshot else None
            steps.append(MergedStep(
                title=s.title,
                instruction=s.instruction,
                screenshot=s.screenshot,
                image_title=desc.title if desc else None,
                image_description=desc.description if desc else None,
            ))
        return MergedDoc(title=doc.title, steps=steps)