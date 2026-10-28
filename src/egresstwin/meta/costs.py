"""Editable Meta Model API price table and budget guard.

Prices are in USD. Text prices are per million tokens, SAM image/video
prices are per 1,000 images / 1,000 frames, voice is per 1,000 minutes.
Edit PRICES to update rates; nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a charge would exceed META_BUDGET_USD."""


@dataclass(frozen=True)
class TokenPrice:
    """USD per million tokens."""

    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class Prices:
    """Whole price table (single editable place)."""

    text_per_million: dict[str, TokenPrice] = field(
        default_factory=lambda: {
            "muse-spark-1.3": TokenPrice(input=1.25, cached_input=0.15, output=4.25),
            "muse-spark-1.3-contributor": TokenPrice(
                input=0.10, cached_input=0.002, output=0.20
            ),
        }
    )
    sam_per_1000_images: float = 2.50
    sam_per_1000_video_frames: float = 0.20
    sam_video_min_frames: int = 50
    voice_per_1000_minutes: float = 3.00


PRICES = Prices()


def estimate_text_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    prices: Prices = PRICES,
) -> float:
    """Estimate a text call. Cached tokens are a subset of input tokens."""
    if input_tokens < 0 or output_tokens < 0 or cached_input_tokens < 0:
        raise ValueError("token counts must be >= 0")
    if cached_input_tokens > input_tokens:
        raise ValueError("cached_input_tokens cannot exceed input_tokens")
    try:
        price = prices.text_per_million[model]
    except KeyError:
        raise ValueError(f"unknown text model: {model}") from None
    fresh = input_tokens - cached_input_tokens
    return (
        fresh * price.input
        + cached_input_tokens * price.cached_input
        + output_tokens * price.output
    ) / 1_000_000


def estimate_sam_image_cost(num_images: int, prices: Prices = PRICES) -> float:
    """Estimate SAM image segmentation for num_images images."""
    if num_images < 0:
        raise ValueError("num_images must be >= 0")
    return num_images * prices.sam_per_1000_images / 1000


def estimate_sam_video_cost(num_frames: int, prices: Prices = PRICES) -> float:
    """Estimate SAM video segmentation; minimum 50 frames per request."""
    if num_frames < 0:
        raise ValueError("num_frames must be >= 0")
    billed = max(num_frames, prices.sam_video_min_frames)
    return billed * prices.sam_per_1000_video_frames / 1000


def estimate_voice_cost(minutes: float, prices: Prices = PRICES) -> float:
    """Estimate voice transcription for the given duration in minutes."""
    if minutes < 0:
        raise ValueError("minutes must be >= 0")
    return minutes * prices.voice_per_1000_minutes / 1000


@dataclass
class CostTracker:
    """Accumulates spend and enforces the budget cap."""

    budget_usd: float
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be > 0")

    @property
    def remaining_usd(self) -> float:
        """Budget left in USD."""
        return self.budget_usd - self.spent_usd

    def check(self, amount_usd: float) -> None:
        """Raise BudgetExceeded if charging amount_usd would pass the cap."""
        if amount_usd < 0:
            raise ValueError("amount_usd must be >= 0")
        if self.spent_usd + amount_usd > self.budget_usd:
            raise BudgetExceeded(
                f"charge of USD {amount_usd:.6f} would exceed budget: "
                f"spent USD {self.spent_usd:.6f} of USD {self.budget_usd:.6f}"
            )

    def charge(self, amount_usd: float) -> float:
        """Accumulate a charge, raising BudgetExceeded past the cap."""
        self.check(amount_usd)
        self.spent_usd += amount_usd
        return self.spent_usd
