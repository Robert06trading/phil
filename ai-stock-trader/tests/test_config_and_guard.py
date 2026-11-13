import pytest

from app.domain import Instrument
from config import BrokerKind, RiskConfig, Settings, TradingMode, load_settings
from security.instrument_guard import InstrumentNotAllowed


def test_default_risk_values():
    r = RiskConfig()
    assert (r.risk_per_trade_pct, r.max_daily_loss_pct, r.max_weekly_loss_pct, r.max_open_positions,
            r.min_reward_risk) == (0.5, 2.0, 5.0, 3, 2.0)


def test_live_disabled_by_default_and_capital_not_hardcoded():
    s = Settings()
    assert s.live_trading_enabled is False
    assert s.mode == TradingMode.PAPER
    assert s.paper_starting_capital is None


@pytest.mark.parametrize("field,value", [("risk_per_trade_pct", 5.0), ("max_daily_loss_pct", 10),
                                         ("max_weekly_loss_pct", 20), ("max_open_positions", 50)])
def test_hard_ceilings(field, value):
    with pytest.raises(ValueError):
        RiskConfig(**{field: value})


def test_env_loading_nested_and_lists(tmp_path):
    s = load_settings({"TRADER_RISK__MAX_OPEN_POSITIONS": "2", "TRADER_VENUES": "NYSE,XETRA",
                       "TRADER_PAPER_STARTING_CAPITAL": "25000"}, dotenv=None)
    assert s.risk.max_open_positions == 2 and s.venues == ["NYSE", "XETRA"] and s.paper_starting_capital == 25000


def test_forbidden_venue_and_stock_type():
    with pytest.raises(ValueError):
        Settings(venues=["NASDAQ", "CBOE"])
    with pytest.raises(ValueError):
        Settings(allowed_stock_types=["COMMON", "ETF"])


def test_live_requires_ibkr():
    with pytest.raises(ValueError):
        Settings(mode=TradingMode.LIVE, broker=BrokerKind.PAPER)


def test_guard_allows_common_stocks(guard, universe):
    for sym in ("AAPL", "JPM", "SAP", "TSM"):
        guard.check(universe[sym])


@pytest.mark.parametrize("inst,why", [
    (Instrument("SPY", "NYSE", "USD", "STK", "ETF"), "ETF"),
    (Instrument("QQQ", "NASDAQ", "USD", "STK", "ETF"), "ETF"),
    (Instrument("BTC", "NASDAQ", "USD", "CRYPTO", None), "crypto"),
    (Instrument("BTC-USD", "NASDAQ", "USD", "STK", "COMMON"), "crypto symbol"),
    (Instrument("EUR", "NASDAQ", "USD", "CASH", None), "forex"),
    (Instrument("AAPL", "NASDAQ", "USD", "CFD", "COMMON"), "CFD"),
    (Instrument("ES", "NYSE", "USD", "FUT", None), "future"),
    (Instrument("AAPL", "NASDAQ", "USD", "OPT", None), "option"),
    (Instrument("GC", "NYSE", "USD", "CMDTY", None), "commodity"),
    (Instrument("T", "NYSE", "USD", "BOND", None), "bond"),
    (Instrument("AAPL", "NASDAQ", "USD", "STK", None), "unknown stock type"),
    (Instrument("AAPL", "ARCA", "USD", "STK", "COMMON"), "venue"),
    (Instrument("AAPL", "NASDAQ", "EUR", "STK", "COMMON"), "currency"),
    (Instrument("ZZZZ", "NASDAQ", "USD", "STK", "COMMON"), "not in universe"),
    (Instrument("SAP", "NYSE", "USD", "STK", "COMMON"), "venue differs from universe"),
    (Instrument("AAPL", "NASDAQ", "USD", "STK", "WARRANT"), "warrant"),
])
def test_guard_blocks_everything_else(guard, inst, why):
    with pytest.raises(InstrumentNotAllowed):
        guard.check(inst)


def test_guard_conid_mismatch(guard, universe):
    from dataclasses import replace
    guard.universe["AAPL"] = replace(universe["AAPL"], con_id=265598)
    with pytest.raises(InstrumentNotAllowed):
        guard.check(replace(universe["AAPL"], con_id=1))
