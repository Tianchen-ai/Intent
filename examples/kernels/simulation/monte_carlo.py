import intent
import intent.language as I


PATHS = 262144
STEPS = 64


@intent.kernel
def barrier_option_paths(
    payoffs: I.Out[I.f32, (PATHS,)],
    seed: I.u64,
    initial_price: I.f32,
    strike: I.f32,
    barrier: I.f32,
    drift: I.f32,
    volatility: I.f32,
):
    for path in I.parallel(I.domain(0, PATHS)):
        price = initial_price
        knocked_out = False
        for step in range(STEPS):
            if knocked_out:
                break
            counter = path * STEPS + step
            uniform = I.random.uniform(seed, counter, dtype=I.f32)
            direction = I.select(uniform >= 0.5, volatility, -volatility)
            price = price * I.exp(drift + direction)
            knocked_out = price >= barrier
        payoff = I.maximum(price - strike, 0.0) if not knocked_out else 0.0
        payoffs[path] = payoff
