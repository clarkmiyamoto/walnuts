"""JAX-based reference implementation of the WALNUTS sampler.

This module mirrors the novel Hamiltonian Monte Carlo algorithm implemented
in :mod:`walnuts`'s C++ headers.  The implementation closely follows the
WALNUTS variant of the No-U-Turn Sampler (NUTS) described in
``Walsh et al., 2025`` and the accompanying C++ reference implementation.

The sampler operates in terms of *spans*, which summarise a Hamiltonian
trajectory between an earliest and a latest state.  The spans can be
combined recursively to form binary trees whose leaves correspond to states
produced by *macro* steps (each macro step may take one or more micro
leapfrog steps).  The sampler adaptively shortens the macro step whenever
Hamiltonian error exceeds a user supplied tolerance, ensuring reversibility
and detailed balance even when the macro integrator step is adjusted.

The Python implementation keeps the original structure:

* ``SpanW`` stores the backwards/forwards end points as well as the
  currently selected state from a trajectory segment together with cached
  gradients and log densities.
* ``macro_step`` integrates Hamilton's equations while repeatedly halving
  the step size until the Hamiltonian error is within tolerance.  It also
  checks that the resulting trajectory is reversible following Algorithm 3
  in the WALNUTS paper.
* ``build_span`` recursively constructs a trajectory tree in a specified
  direction using Barker updates to choose an intermediate state.
* ``transition_w`` orchestrates the outer NUTS loop, randomly expanding the
  trajectory forward or backward until a U-turn is detected or the maximum
  depth is reached.  States are chosen with a Metropolis decision.

The entire implementation is differentiable with respect to the state and
mass matrix thanks to JAX's functional array programming model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

Array = jax.Array


class Update(Enum):
    """Proposal update schemes for Markov transitions."""

    BARKER = auto()
    METROPOLIS = auto()


class Direction(Enum):
    """Integration direction for Hamiltonian trajectories."""

    BACKWARD = auto()
    FORWARD = auto()


@dataclass
class SpanW:
    """A span summarising a WALNUTS trajectory segment."""

    theta_bk: Array
    rho_bk: Array
    grad_theta_bk: Array
    logp_bk: Array
    theta_fw: Array
    rho_fw: Array
    grad_theta_fw: Array
    logp_fw: Array
    theta_select: Array
    grad_select: Array
    logp: Array

    @classmethod
    def from_initial_point(
        cls, theta: Array, rho: Array, grad_theta: Array, logp: Array
    ) -> "SpanW":
        """Create a span consisting of a single Hamiltonian state."""

        return cls(
            theta_bk=theta,
            rho_bk=rho,
            grad_theta_bk=grad_theta,
            logp_bk=logp,
            theta_fw=theta,
            rho_fw=rho,
            grad_theta_fw=grad_theta,
            logp_fw=logp,
            theta_select=theta,
            grad_select=grad_theta,
            logp=logp,
        )

    @classmethod
    def from_subspans(
        cls,
        span1: "SpanW",
        span2: "SpanW",
        theta_select: Array,
        grad_select: Array,
        logp: Array,
    ) -> "SpanW":
        """Concatenate two spans with a newly selected state."""

        return cls(
            theta_bk=span1.theta_bk,
            rho_bk=span1.rho_bk,
            grad_theta_bk=span1.grad_theta_bk,
            logp_bk=span1.logp_bk,
            theta_fw=span2.theta_fw,
            rho_fw=span2.rho_fw,
            grad_theta_fw=span2.grad_theta_fw,
            logp_fw=span2.logp_fw,
            theta_select=theta_select,
            grad_select=grad_select,
            logp=logp,
        )


def log_sum_exp(x1: Array, x2: Array) -> Array:
    """Stable log(exp(x1) + exp(x2)) for scalar inputs."""

    return jnp.logaddexp(x1, x2)


def logp_momentum(rho: Array, inv_mass: Array) -> Array:
    """Return the unnormalised log density of a momentum draw."""

    return -0.5 * jnp.dot(inv_mass * rho, rho)


def order_forward_backward(
    direction: Direction, span_old: SpanW, span_new: SpanW
) -> Tuple[SpanW, SpanW]:
    """Return spans ordered according to the chosen direction."""

    if direction is Direction.FORWARD:
        return span_old, span_new
    return span_new, span_old


def uturn(direction: Direction, span1: SpanW, span2: SpanW, inv_mass: Array) -> bool:
    """Return whether two spans form a U-turn in the mass metric."""

    span_bk, span_fw = order_forward_backward(direction, span1, span2)
    scaled_diff = inv_mass * (span_fw.theta_fw - span_bk.theta_bk)
    return bool(
        jnp.dot(span_fw.rho_fw, scaled_diff) < 0.0
        or jnp.dot(span_bk.rho_bk, scaled_diff) < 0.0
    )


class Random:
    """Random number utilities backed by a JAX PRNG key."""

    def __init__(self, key: Array, dtype=jnp.float64):
        self._key = key
        self._dtype = dtype

    def _split(self) -> Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def uniform_real_01(self) -> Array:
        return jax.random.uniform(
            self._split(), (), minval=0.0, maxval=1.0, dtype=self._dtype
        )

    def uniform_binary(self) -> bool:
        return bool(jax.random.bernoulli(self._split()))

    def standard_normal(self, shape: Union[Tuple[int, ...], int]) -> Array:
        if isinstance(shape, int):
            shape = (shape,)
        return jax.random.normal(self._split(), shape, dtype=self._dtype)

    @property
    def key(self) -> Array:
        return self._key


LogProbAndGrad = Callable[[Array], Tuple[Array, Array]]
AdaptHandler = Callable[[Array], None]


def leapfrog(
    logp_grad: LogProbAndGrad,
    inv_mass: Array,
    step: Array,
    theta: Array,
    rho: Array,
    grad: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Perform one leapfrog step of Hamiltonian dynamics."""

    half_step = 0.5 * step
    rho_half = rho + half_step * grad
    theta_next = theta + step * (inv_mass * rho_half)
    logp_theta_next, grad_next = logp_grad(theta_next)
    rho_next = rho_half + half_step * grad_next
    logp_next = logp_theta_next + logp_momentum(rho_next, inv_mass)
    return theta_next, rho_next, grad_next, logp_next


def within_tolerance(
    logp_grad: LogProbAndGrad,
    inv_mass: Array,
    step: Array,
    num_steps: int,
    max_error: Array,
    logp_start: Array,
    theta: Array,
    rho: Array,
    grad: Array,
) -> bool:
    """Return ``True`` if a macro step satisfies the Hamiltonian tolerance."""

    half_step = 0.5 * step
    theta_next = theta
    rho_next = rho
    grad_next = grad
    logp = logp_start
    logp_next = logp_start
    for _ in range(num_steps):
        rho_next = rho_next + half_step * grad_next
        theta_next = theta_next + step * (inv_mass * rho_next)
        logp_theta_next, grad_next = logp_grad(theta_next)
        rho_next = rho_next + half_step * grad_next
        logp_next = logp_theta_next + logp_momentum(rho_next, inv_mass)
    return bool(jnp.abs(logp_next - logp) <= max_error)


def reversible(
    logp_grad: LogProbAndGrad,
    inv_mass: Array,
    step: Array,
    num_steps: int,
    max_error: Array,
    logp_next: Array,
    theta: Array,
    rho: Array,
    grad: Array,
) -> bool:
    """Check whether the selected macro step is reversible."""

    if num_steps == 1:
        return True

    step_curr = step
    steps_curr = num_steps
    while steps_curr >= 2:
        theta_next = theta
        rho_next = -rho
        grad_next = grad
        steps_curr //= 2
        step_curr *= 2
        if within_tolerance(
            logp_grad,
            inv_mass,
            step_curr,
            steps_curr,
            max_error,
            logp_next,
            theta_next,
            rho_next,
            grad_next,
        ):
            return False
    return True


def macro_step(
    direction: Direction,
    logp_grad: LogProbAndGrad,
    inv_mass: Array,
    step: Array,
    max_error: Array,
    span: SpanW,
    adapt_handler: AdaptHandler,
) -> Optional[Tuple[Array, Array, Array, Array]]:
    """Attempt a macro step; halve the step size until tolerance is met."""

    if direction is Direction.FORWARD:
        theta = span.theta_fw
        rho = span.rho_fw
        grad = span.grad_theta_fw
        logp = span.logp_fw
        step_dir = step
    else:
        theta = span.theta_bk
        rho = span.rho_bk
        grad = span.grad_theta_bk
        logp = span.logp_bk
        step_dir = -step

    current_step = step_dir
    num_steps = 1
    for _ in range(10):
        theta_next = theta
        rho_next = rho
        grad_next = grad
        half_step = 0.5 * current_step
        logp_theta_next = logp
        for _ in range(num_steps):
            rho_next = rho_next + half_step * grad_next
            theta_next = theta_next + current_step * (inv_mass * rho_next)
            logp_theta_next, grad_next = logp_grad(theta_next)
            rho_next = rho_next + half_step * grad_next
        logp_next = logp_theta_next + logp_momentum(rho_next, inv_mass)

        if num_steps == 1:
            min_accept = jnp.exp(-jnp.abs(logp - logp_next))
            adapt_handler(min_accept)

        if jnp.abs(logp - logp_next) <= max_error:
            if reversible(
                logp_grad,
                inv_mass,
                current_step,
                num_steps,
                max_error,
                logp_next,
                theta_next,
                rho_next,
                grad_next,
            ):
                return theta_next, rho_next, grad_next, logp_next
            return None

        num_steps *= 2
        current_step *= 0.5

    return None


def combine(
    update: Update,
    direction: Direction,
    rng: Random,
    span_old: SpanW,
    span_new: SpanW,
) -> SpanW:
    """Combine two spans and update the selected state."""

    logp_total = log_sum_exp(span_old.logp, span_new.logp)
    if update is Update.METROPOLIS:
        log_denominator = span_old.logp
    else:
        log_denominator = logp_total

    update_logprob = span_new.logp - log_denominator
    update_decision = jnp.log(rng.uniform_real_01()) < update_logprob
    theta_selected = jax.lax.select(update_decision, span_new.theta_select, span_old.theta_select)
    grad_selected = jax.lax.select(update_decision, span_new.grad_select, span_old.grad_select)

    span_bk, span_fw = order_forward_backward(direction, span_old, span_new)
    return SpanW.from_subspans(span_bk, span_fw, theta_selected, grad_selected, logp_total)


def build_leaf(
    direction: Direction,
    logp_grad: LogProbAndGrad,
    span: SpanW,
    inv_mass: Array,
    step: Array,
    max_error: Array,
    adapt_handler: AdaptHandler,
) -> Optional[SpanW]:
    """Extend a span by a single macro step."""

    result = macro_step(direction, logp_grad, inv_mass, step, max_error, span, adapt_handler)
    if result is None:
        return None
    theta_next, rho_next, grad_next, logp_next = result
    return SpanW.from_initial_point(theta_next, rho_next, grad_next, logp_next)


def build_span(
    direction: Direction,
    rng: Random,
    logp_grad: LogProbAndGrad,
    inv_mass: Array,
    step: Array,
    depth: int,
    max_error: Array,
    last_span: SpanW,
    adapt_handler: AdaptHandler,
) -> Optional[SpanW]:
    """Recursively construct a span of ``2**depth`` states."""

    if depth == 0:
        return build_leaf(direction, logp_grad, last_span, inv_mass, step, max_error, adapt_handler)

    subspan1 = build_span(
        direction,
        rng,
        logp_grad,
        inv_mass,
        step,
        depth - 1,
        max_error,
        last_span,
        adapt_handler,
    )
    if subspan1 is None:
        return None

    subspan2 = build_span(
        direction,
        rng,
        logp_grad,
        inv_mass,
        step,
        depth - 1,
        max_error,
        subspan1,
        adapt_handler,
    )
    if subspan2 is None:
        return None

    if uturn(direction, subspan1, subspan2, inv_mass):
        return None

    return combine(Update.BARKER, direction, rng, subspan1, subspan2)


def transition_w(
    rng: Random,
    logp_grad: LogProbAndGrad,
    inv_mass: Array,
    chol_mass: Array,
    step: Array,
    max_depth: int,
    theta: Array,
    max_error: Array,
    adapt_handler: AdaptHandler,
) -> Tuple[Array, Array]:
    """Generate the next state and gradient using the WALNUTS transition."""

    rho = rng.standard_normal(theta.shape) * chol_mass
    logp_theta, grad = logp_grad(theta)
    logp = logp_theta + logp_momentum(rho, inv_mass)
    span_accum = SpanW.from_initial_point(theta, rho, grad, logp)

    for depth in range(max_depth):
        def expand(direction: Direction) -> bool:
            nonlocal span_accum
            next_span = build_span(
                direction,
                rng,
                logp_grad,
                inv_mass,
                step,
                depth,
                max_error,
                span_accum,
                adapt_handler,
            )
            if next_span is None:
                return True

            combined_uturn = uturn(direction, span_accum, next_span, inv_mass)
            span_accum = combine(Update.METROPOLIS, direction, rng, span_accum, next_span)
            return combined_uturn

        go_forward = rng.uniform_binary()
        did_uturn = expand(Direction.FORWARD) if go_forward else expand(Direction.BACKWARD)
        if did_uturn:
            break

    return span_accum.theta_select, span_accum.grad_select


class WalnutsSampler:
    """A thin convenience wrapper that matches the C++ sampler API."""

    def __init__(
        self,
        key: Array,
        logprob_fn: Callable[[Array], Array],
        theta: Array,
        inv_mass: Array,
        macro_step_size: Array,
        max_depth: int,
        max_error: Array,
        dtype=jnp.float64,
    ) -> None:
        self._rng = Random(key, dtype=dtype)
        self._logprob_fn = logprob_fn
        self._theta = jnp.asarray(theta, dtype=dtype)
        self._inv_mass = jnp.asarray(inv_mass, dtype=dtype)
        self._chol_mass = 1.0 / jnp.sqrt(self._inv_mass)
        self._macro_step = jnp.asarray(macro_step_size, dtype=dtype)
        self._max_depth = int(max_depth)
        self._max_error = jnp.asarray(max_error, dtype=dtype)
        self._dtype = dtype

        self._value_and_grad = jax.value_and_grad(logprob_fn)

    def __call__(self) -> Array:
        theta_next, _ = self.step()
        return theta_next

    def step(self) -> Tuple[Array, Array]:
        theta = self._theta
        theta_next, grad_next = transition_w(
            self._rng,
            self._value_and_grad,
            self._inv_mass,
            self._chol_mass,
            self._macro_step,
            self._max_depth,
            theta,
            self._max_error,
            lambda _: None,
        )
        self._theta = theta_next
        return theta_next, grad_next

    @property
    def inverse_mass(self) -> Array:
        return self._inv_mass

    @property
    def macro_step_size(self) -> Array:
        return self._macro_step

    @property
    def max_error(self) -> Array:
        return self._max_error
