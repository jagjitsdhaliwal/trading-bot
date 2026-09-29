from datetime import date

from engine.risk.engine import RiskDecision, RiskEngine, RiskLimits


def make_engine() -> RiskEngine:
    limits = RiskLimits(
        max_daily_loss=300.0,
        max_trade_risk=50.0,
        max_trades_per_day=8,
        max_consecutive_losses=3,
        max_position_size=2,
        account_starting_balance=50_000.0,
        max_trailing_drawdown=2_000.0,
        profit_target=3_000.0,
    )
    engine = RiskEngine(limits)
    engine.start_new_day(date.today())
    return engine


def test_approves_valid_order():
    engine = make_engine()
    decision, _ = engine.check_order(proposed_risk_dollars=40, quantity=1)
    assert decision == RiskDecision.APPROVED


def test_rejects_oversized_position():
    engine = make_engine()
    decision, _ = engine.check_order(proposed_risk_dollars=40, quantity=5)
    assert decision == RiskDecision.REJECTED


def test_rejects_excess_trade_risk():
    engine = make_engine()
    decision, _ = engine.check_order(proposed_risk_dollars=100, quantity=1)
    assert decision == RiskDecision.REJECTED


def test_kill_switch_on_daily_loss():
    engine = make_engine()
    engine.record_trade_result(-310)
    decision, reason = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH
    assert "Daily loss" in reason


def test_kill_switch_on_consecutive_losses():
    engine = make_engine()
    engine.record_trade_result(-10)
    engine.record_trade_result(-10)
    engine.record_trade_result(-10)
    decision, reason = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH
    assert "consecutive losses" in reason


def test_kill_switch_on_trailing_drawdown():
    # Use a daily loss limit wide enough that only the drawdown check is
    # exercised -- isolates drawdown logic from the daily-loss check.
    limits = RiskLimits(
        max_daily_loss=10_000.0,
        max_trade_risk=50.0,
        max_trades_per_day=8,
        max_consecutive_losses=10,
        max_position_size=2,
        account_starting_balance=50_000.0,
        max_trailing_drawdown=2_000.0,
        profit_target=3_000.0,
    )
    engine = RiskEngine(limits)
    engine.start_new_day(date.today())

    engine.record_trade_result(500)  # peak now 50500
    engine.record_trade_result(-1900)  # drawdown from peak = 1900, under limit
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.APPROVED

    engine.record_trade_result(-200)  # drawdown now 2100, over 2000 limit
    decision, reason = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH
    assert "drawdown" in reason


def test_daily_loss_kill_switch_clears_on_new_day():
    engine = make_engine()
    engine.record_trade_result(-310)
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH

    # Same day again: still triggered, manual reset alone won't help because
    # daily_pnl still reflects the breach.
    engine.reset_kill_switch()
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH

    # A new day clears the daily counters AND the daily-loss kill switch.
    from datetime import timedelta

    engine.start_new_day(date.today() + timedelta(days=1))
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.APPROVED


def test_non_daily_kill_switch_requires_manual_reset():
    engine = make_engine()
    engine.record_trade_result(-10)
    engine.record_trade_result(-10)
    engine.record_trade_result(-10)  # consecutive-loss kill switch
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH

    from datetime import timedelta

    engine.start_new_day(date.today() + timedelta(days=1))
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH  # new day does NOT clear this

    # Resetting alone isn't enough -- consecutive_losses count is untouched,
    # so it immediately re-triggers. A winning trade is needed to actually
    # break the streak before the reset sticks.
    engine.reset_kill_switch()
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH

    engine.reset_kill_switch()
    engine.record_trade_result(15)  # winning trade breaks the streak
    decision, _ = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.APPROVED


def test_profit_target_triggers_stop():
    engine = make_engine()
    engine.record_trade_result(3100)
    decision, reason = engine.check_order(proposed_risk_dollars=10, quantity=1)
    assert decision == RiskDecision.KILL_SWITCH
    assert "Profit target" in reason
