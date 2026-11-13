"""Instrument whitelist — the hard gate that keeps the bot on single stocks.

Two layers:
1. A static, operator-maintained universe file (symbol, venue, currency,
   stock_type). Only listed symbols are tradable.
2. A per-order check of the instrument *as reported by the broker*
   (secType == STK, stockType in {COMMON, ADR}, primary venue in
   NASDAQ/NYSE/XETRA, currency matching the venue).

Anything unknown is rejected (fail closed). Crypto, FX, CFDs, ETFs, futures,
options, commodities and bonds therefore cannot pass: their secType differs
(CRYPTO, CASH, CFD, FUT, OPT, CMDTY, BOND, FUND, WAR, IOPT ...) or their
stockType is ETF/ETN/ETC/FUND/... .
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from app.domain import Instrument

VENUE_CURRENCY = {"NASDAQ": "USD", "NYSE": "USD", "XETRA": "EUR"}

# IBKR primary-exchange codes -> our venue names
IBKR_EXCHANGE_TO_VENUE = {
    "NASDAQ": "NASDAQ",
    "ISLAND": "NASDAQ",
    "NYSE": "NYSE",
    "IBIS": "XETRA",      # IBKR's code for XETRA
    "IBIS2": "XETRA",
}

ALLOWED_SEC_TYPES = {"STK"}
BLOCKED_STOCK_TYPES = {
    "ETF", "ETN", "ETC", "ETP", "FUND", "CLOSED-END FUND", "MUTUAL FUND", "UNIT",
    "RIGHT", "WARRANT", "PREFERRED", "PFD", "STRUCTURED", "CERTIFICATE", "BOND",
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,6}([.][A-Z])?$")
_SUSPICIOUS = re.compile(r"(USD|EUR|BTC|ETH|USDT)[-/.]|[-/](USD|EUR|BTC|ETH|USDT)$")


class InstrumentNotAllowed(Exception):
    pass


class InstrumentGuard:
    def __init__(self, universe: dict[str, Instrument], venues: list[str], allowed_stock_types: list[str]):
        self.universe = universe
        self.venues = set(venues)
        self.allowed_stock_types = {s.upper() for s in allowed_stock_types}

    @classmethod
    def from_file(cls, path: Path, venues: list[str], allowed_stock_types: list[str]) -> "InstrumentGuard":
        return cls(load_universe(path), venues, allowed_stock_types)

    def check(self, inst: Instrument) -> None:
        """Raise InstrumentNotAllowed unless every rule passes."""
        sym = inst.symbol.upper()
        if not _SYMBOL_RE.match(sym) or _SUSPICIOUS.search(sym):
            raise InstrumentNotAllowed(f"{sym}: symbol format not allowed")
        if inst.sec_type.upper() not in ALLOWED_SEC_TYPES:
            raise InstrumentNotAllowed(f"{sym}: secType {inst.sec_type} is not a stock")
        stype = (inst.stock_type or "").upper()
        if not stype:
            raise InstrumentNotAllowed(f"{sym}: stock type unknown (fail closed)")
        if stype in BLOCKED_STOCK_TYPES or stype not in self.allowed_stock_types:
            raise InstrumentNotAllowed(f"{sym}: stock type {stype} not allowed")
        if inst.venue not in self.venues or inst.venue not in VENUE_CURRENCY:
            raise InstrumentNotAllowed(f"{sym}: venue {inst.venue} not allowed")
        if inst.currency != VENUE_CURRENCY[inst.venue]:
            raise InstrumentNotAllowed(f"{sym}: currency {inst.currency} does not match {inst.venue}")
        listed = self.universe.get(sym)
        if listed is None:
            raise InstrumentNotAllowed(f"{sym}: not in approved universe")
        if listed.venue != inst.venue or listed.currency != inst.currency:
            raise InstrumentNotAllowed(f"{sym}: venue/currency differ from approved universe entry")
        if listed.con_id and inst.con_id and listed.con_id != inst.con_id:
            raise InstrumentNotAllowed(f"{sym}: conId mismatch ({inst.con_id} != {listed.con_id})")

    def is_allowed(self, inst: Instrument) -> bool:
        try:
            self.check(inst)
            return True
        except InstrumentNotAllowed:
            return False

    def get(self, symbol: str) -> Instrument:
        inst = self.universe.get(symbol.upper())
        if inst is None:
            raise InstrumentNotAllowed(f"{symbol}: not in approved universe")
        self.check(inst)
        return inst

    def symbols(self) -> list[str]:
        return [s for s, i in self.universe.items() if self.is_allowed(i)]


def load_universe(path: Path) -> dict[str, Instrument]:
    out: dict[str, Instrument] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("symbol") or row["symbol"].startswith("#"):
                continue
            con_id = row.get("con_id") or None
            inst = Instrument(
                symbol=row["symbol"].strip().upper(),
                venue=row["venue"].strip().upper(),
                currency=row["currency"].strip().upper(),
                sec_type=(row.get("sec_type") or "STK").strip().upper(),
                stock_type=(row.get("stock_type") or "").strip().upper() or None,
                con_id=int(con_id) if con_id else None,
                name=(row.get("name") or "").strip(),
            )
            out[inst.symbol] = inst
    return out
