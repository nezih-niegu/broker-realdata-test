"""Game configuration.

Defaults reproduce Section 5 of Cartea, Jaimungal & Sanchez-Betancourt (2026),
"Nash equilibrium between brokers and traders", Finance & Stochastics 30:951-982.
"""
from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class GameConfig:
    # horizon / grid
    T: float = 1.0
    n: int = 100
    # costs and penalties
    a: float = 1.2e-3      # broker's instantaneous cost in the lit market
    b: float = 1.0e-3      # informed trader's cost when trading with the broker
    c: float = 1.0e-3      # uninformed trader's cost when trading with the broker
    phi: float = 1.0       # broker terminal inventory penalty
    psi: float = 1.0       # informed terminal inventory penalty
    rB: float = 0.0        # broker running inventory penalty
    rI: float = 0.0        # informed running inventory penalty
    # Obizhaeva-Wang transient impact of the broker: dY = (Ups v - Xi Y) dt
    Ups: float = 1.0e-3
    Xi: float = 0.0
    # exogenous processes
    kappa_a: float = 5.0   # signal alpha: OU
    sig_a: float = 1.0
    kappa_x: float = 15.0  # uninformed flow xi: OU
    sig_x: float = 100.0
    sig_S: float = 1.0     # price noise
    qB0: float = 0.0
    qI0: float = 0.0
    Y0: float = 0.0
    # ---- extensions beyond the paper -------------------------------------
    # info="full": everybody sees everything (the paper).
    # info="anonB": only the broker loses alpha and client identities (informed keeps full view).
    # info="anon": broker does NOT see alpha, nor who sent which order; she sees
    #              only the lagged net client flow (eta + xi) and price changes.
    #              The informed trader sees alpha, his inventory and Y, not xi, not Q^B.
    info: str = "full"
    # pricing=True: the broker quotes the informed client's cost coefficient
    #               m_t in [b_lo, b_hi] each step (paper: fixed b). Trader sees the quote.
    pricing: bool = False
    b_lo: float = 0.5e-3
    b_hi: float = 5.0e-3
    # outside_cost=c_L: the informed trader may also trade directly in the lit market at
    #                   temporary cost c_L and with the same transient impact as the broker
    #                   (paper: no lit access). Informed policy then has two outputs.
    outside_cost: float = None

    @property
    def dt(self) -> float:
        return self.T / self.n

    def with_(self, **kw) -> "GameConfig":
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)
