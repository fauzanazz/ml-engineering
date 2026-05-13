from __future__ import annotations

import random
from dataclasses import dataclass, field


PRODUCT_NAMES = (
    "rekening tabungan",
    "rekening giro",
    "deposito",
    "kartu kredit",
    "kartu debit",
    "KPR",
    "KTA",
    "pinjaman multiguna",
    "cicilan kendaraan",
    "virtual account",
    "BI-FAST",
    "mobile banking",
    "internet banking",
    "QRIS",
    "paylater",
)

TRANSFER_METHODS = ("BI-FAST", "transfer online", "SKN", "RTGS", "virtual account")
MERCHANT_NAMES = ("Tokopedia", "Shopee", "PLN", "Telkom", "BPJS")
TENORS = (3, 6, 12, 24, 36, 60)


@dataclass
class EntitySampler:
    seed: int
    random_generator: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.random_generator = random.Random(self.seed)

    def sample_values(self) -> dict[str, str]:
        return {
            "product_name": self.random_generator.choice(PRODUCT_NAMES),
            "account_number": self._digits(self.random_generator.randint(10, 14)),
            "amount": self._amount(),
            "interest_rate": self._interest_rate(),
            "date": self._date(),
            "card_last4": self._digits(4),
            "tenor": f"{self.random_generator.choice(TENORS)} bulan",
            "transfer_method": self.random_generator.choice(TRANSFER_METHODS),
            "merchant_name": self.random_generator.choice(MERCHANT_NAMES),
        }

    def _digits(self, length: int) -> str:
        return "".join(str(self.random_generator.randint(0, 9)) for _ in range(length))

    def _amount(self) -> str:
        value = self.random_generator.randrange(10_000, 50_000_001, 10_000)
        return "Rp" + f"{value:,}".replace(",", ".")

    def _interest_rate(self) -> str:
        value = self.random_generator.randint(25, 180) / 10
        return f"{value:.1f}%".replace(".", ",")

    def _date(self) -> str:
        day = self.random_generator.randint(1, 28)
        month = self.random_generator.choice(
            ("Januari", "Februari", "Maret", "April", "Mei", "Juni")
        )
        return f"{day} {month} 2026"
