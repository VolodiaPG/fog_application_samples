from __future__ import annotations

import csv
import functools
import math
import os
import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from math import expm1
from typing import Any, Callable, Dict, List, Tuple

import numpy  # type: ignore
import scipy.integrate as integrate  # type: ignore
import simpy  # type: ignore
from alive_progress import alive_bar  # type: ignore

import expe
from definitions import FOG_NODES, LEVELS, NETWORK, gen_net

MS = 1
SECS = 1000

SIM_TIME = expe.FUNCTION_RESERVATION_FINISHES_AFTER * SECS

RANDOM_SEED = expe.RANDOM_SEED
if RANDOM_SEED is not None and RANDOM_SEED != "":
    random.seed(RANDOM_SEED)


@dataclass
class SLA:
    mem: float
    core: float
    latency: float
    duration: float


@dataclass(order=True)
class Bid:
    bid: float
    bidder: str
    acc_latency: float


@dataclass
class ProvisionedFunction:
    bid: float
    acc_latency: float
    latency: float
    timestamp_start: float | int
    timestamp_end: float | int
    cpu_reservation_used_sla: float
    node_cpu: float


@dataclass
class Monitoring:
    currently_provisioned = 0
    total_provisioned = 0
    total_submitted = 0
    earnings: Dict[str, List[ProvisionedFunction]] = field(
        default_factory=lambda: defaultdict(lambda: [])
    )


class Request(ABC):
    """A visitor pattern driven request. Can act upon a given Fog node in any way desired way"""

    @abstractmethod
    def __call__(self, node: FogNode, caller: str) -> Any:
        pass


class Pricing(ABC):
    """A visitor pattern driven cost estimator, from a given SLA"""

    @abstractmethod
    def __call__(
        self, node: FogNode, *, sla: SLA, accumulated_latency: float, caller: str
    ) -> Bid:
        pass


class ConstantPricing(Pricing):
    def __init__(self, price: float | Callable[[], float]) -> None:
        self.price = price() if callable(price) else price

    def __call__(self, node: FogNode, accumulated_latency: float, **_) -> Bid:
        return Bid(self.price, node.name, accumulated_latency)


class RandomPricing(Pricing):
    def __init__(self, max_price: float) -> None:
        self.max_price = max_price

    def __call__(self, node: FogNode, accumulated_latency: float, **_) -> Bid:
        return Bid(random.uniform(0, self.max_price), node.name, accumulated_latency)


class LinearPricing(Pricing):
    def __init__(
        self,
        slope: float,
        initial_price: float | Callable[[], float] = 0.0,
    ) -> None:
        self.slope = slope
        self.initial_price = (
            initial_price() if callable(initial_price) else initial_price
        )

    def price(self, utilization):
        # return self.slope * utilization
        return self.initial_price + self.slope * utilization

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            (node.cores_used) / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        return Bid(price, node.name, accumulated_latency)


class LinearUniformRandomPricing(Pricing):
    def __init__(
        self,
        slope: float,
        initial_price: float | Callable[[], float] = 0.0,
    ) -> None:
        self.slope = slope
        self.initial_price = (
            initial_price() if callable(initial_price) else initial_price
        )

    def price(self, utilization):
        return self.initial_price + self.slope * utilization

    def price_no_init(self, utilization):
        return self.slope * utilization

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            (node.cores_used) / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price_no_init, _ = integrate.quad(
            self.price_no_init,
            (node.cores_used) / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price += random.uniform(price_no_init, price * 10)
        return Bid(price, node.name, accumulated_latency)


class LinearNormalRandomPricing(Pricing):
    def __init__(
        self,
        slope: float,
        initial_price: float | Callable[[], float] = 0.0,
    ) -> None:
        self.slope = slope
        self.initial_price = (
            initial_price() if callable(initial_price) else initial_price
        )

    def price(self, utilization):
        return self.initial_price + self.slope * utilization

    def price_no_init(self, utilization):
        return self.slope * utilization

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            (node.cores_used) / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price_no_init, _ = integrate.quad(
            self.price_no_init,
            (node.cores_used) / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price = max(
            price_no_init,
            min(
                price + random.normalvariate(0, self.initial_price),
                price * 2,
                accumulated_latency,
            ),
        )
        return Bid(price, node.name, accumulated_latency)


class ExpPricing(Pricing):
    def __init__(self, multiplier: float) -> None:
        self.multiplier = multiplier

    def price(self, utilization):
        return expm1(utilization * self.multiplier)

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        return Bid(price, node.name, accumulated_latency)


class ExpPricingLatency(Pricing):
    def __init__(self, multiplier: float, multiplier_latency: float) -> None:
        self.multiplier = multiplier
        self.multiplier_latency = multiplier_latency

    def price(self, utilization):
        return expm1(utilization * self.multiplier)

    def price_lat(self, utilization):
        return expm1(utilization * self.multiplier)

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price2, _ = integrate.quad(
            self.price_lat,
            0,
            accumulated_latency / sla.latency,
        )
        price = price + price2
        return Bid(price, node.name, accumulated_latency)


class LinearPerPartPricing(Pricing):
    def __init__(
        self,
        slopes: List[float],
        breaking_points: List[float],
        initial_price: float | Callable[[], float] = 0.0,
    ) -> None:
        assert len(slopes) == len(breaking_points) + 1
        assert len(breaking_points) != 0
        self.slopes = slopes
        self.breaking_points = breaking_points
        self.initial_price = (
            initial_price() if callable(initial_price) else initial_price
        )

    def price(self, utilization):
        slope_ii = 0
        for ii, break_point in enumerate(self.breaking_points):
            if utilization <= break_point:
                break
            slope_ii = ii + 1
        return self.initial_price + self.slopes[slope_ii] * utilization

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
            limit=150,
        )
        return Bid(price, node.name, accumulated_latency)


class LinearLatencySafePricing(Pricing):
    def __init__(
        self,
        network: Dict[Tuple[str, str], float],
        slopes: List[float],
        breaking_points: List[float],
        latency_slopes: List[float],
        latency_breaking_points: List[float],
        initial_price: float | Callable[[], float] = 0.0,
    ) -> None:
        assert len(latency_slopes) == len(latency_breaking_points) + 1
        assert len(latency_breaking_points) != 0
        self.network = network
        self.usage_pricing = LinearPerPartPricing(
            slopes, breaking_points, initial_price
        )
        self.latency_slopes = latency_slopes
        self.latency_breaking_points = latency_breaking_points

    def price(self, latency):
        slope_ii = 0
        for ii, break_point in enumerate(self.latency_breaking_points):
            if latency <= break_point:
                break
            slope_ii = ii + 1
        return self.latency_slopes[slope_ii] * latency + 1

    def __call__(
        self, node: FogNode, sla: SLA, accumulated_latency: float, caller: str, **_
    ) -> Bid:
        usage_price = self.usage_pricing(
            node, sla=sla, accumulated_latency=accumulated_latency, caller=caller
        )
        price = 0
        if caller != "":
            self.network[(caller, node.name)]
            price, _ = integrate.quad(
                self.price,
                # (accumulated_latency - latency_from_coming_node) / sla.latency,
                0,
                accumulated_latency / sla.latency,
                limit=150,
            )

        # if price < 1:
        #     price = 1
        usage_price.bid = usage_price.bid * price
        return usage_price


class LogisticPricing(Pricing):
    def __init__(
        self, A: float, K: float, C: float, Q: float, V: float, B: float
    ) -> None:
        self.A = A
        self.K = K
        self.C = C
        self.Q = Q
        self.V = V
        self.B = B

    def price(self, utilization):
        return self.A + (self.K - self.A) / (
            math.pow(
                self.C + self.Q * math.exp(-1.0 * self.B * utilization), 1 / self.V
            )
        )

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        return Bid(price, node.name, accumulated_latency)


class LogisticAccPricing(Pricing):
    def __init__(
        self, A: float, K: float, C: float, Q: float, V: float, B: float
    ) -> None:
        self.A = A
        self.K = K
        self.C = C
        self.Q = Q
        self.V = V
        self.B = B

    def price(self, utilization):
        return self.A + (self.K - self.A) / (
            math.pow(
                self.C + self.Q * math.exp(-1.0 * self.B * utilization), 1 / self.V
            )
        )

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price2, _ = integrate.quad(
            self.price,
            0,
            accumulated_latency / sla.latency,
        )
        return Bid(price + price2, node.name, accumulated_latency)


class LogisticAccMinPricing(Pricing):
    def __init__(
        self, A: float, K: float, C: float, Q: float, V: float, B: float
    ) -> None:
        self.A = A
        self.K = K
        self.C = C
        self.Q = Q
        self.V = V
        self.B = B

    def price(self, utilization):
        return self.A + (self.K - self.A) / (
            math.pow(
                self.C + self.Q * math.exp(-1.0 * self.B * utilization), 1 / self.V
            )
        )

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price2, _ = integrate.quad(
            self.price,
            0,
            accumulated_latency / sla.latency,
        )
        return Bid(max(price, min(price, price2)), node.name, accumulated_latency)


class LogisticAccMaxPricing(Pricing):
    def __init__(
        self, A: float, K: float, C: float, Q: float, V: float, B: float
    ) -> None:
        self.A = A
        self.K = K
        self.C = C
        self.Q = Q
        self.V = V
        self.B = B

    def price(self, utilization):
        return self.A + (self.K - self.A) / (
            math.pow(
                self.C + self.Q * math.exp(-1.0 * self.B * utilization), 1 / self.V
            )
        )

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price2, _ = integrate.quad(
            self.price,
            0,
            accumulated_latency / sla.latency,
        )
        return Bid(max(price, price2), node.name, accumulated_latency)


class LogisticAccSumPricing(Pricing):
    def __init__(
        self, A: float, K: float, C: float, Q: float, V: float, B: float
    ) -> None:
        self.A = A
        self.K = K
        self.C = C
        self.Q = Q
        self.V = V
        self.B = B

    def price(self, utilization):
        return self.A + (self.K - self.A) / (
            math.pow(
                self.C + self.Q * math.exp(-1.0 * self.B * utilization), 1 / self.V
            )
        )

    def __call__(self, node: FogNode, sla: SLA, accumulated_latency: float, **_) -> Bid:
        price, _ = integrate.quad(
            self.price,
            node.cores_used / node.cores,
            (node.cores_used + sla.core) / node.cores,
        )
        price2, _ = integrate.quad(
            self.price,
            0,
            accumulated_latency / sla.latency,
        )
        return Bid(price + price2, node.name, accumulated_latency)


class AuctionBidRequest(Request):
    def __init__(
        self,
        env,
        sla: SLA,
        network: Dict[Tuple[str, str], float],
        acc_latency=0.0,
    ) -> None:
        super().__init__()
        self.env = env
        self.acc_latency = acc_latency
        self.sla = sla
        self.network = network

    def __call__(self, node: FogNode, caller: str):
        bids: Any = []
        requests: List[Any] = []
        for child in node.children + [node.parent] if node.parent is not None else []:
            if child.name == caller:
                continue
            delay = self.network[(node.name, child.name)]
            if delay + self.acc_latency <= self.sla.latency:
                req = env.process(
                    node.send(
                        child,
                        AuctionBidRequest(
                            self.env,
                            self.sla,
                            self.network,
                            self.acc_latency + delay,
                        ),
                    )
                )
                requests.append(req)

        for req in requests:
            req = yield req
            bids.extend(req)

        if (
            node.cores_used + self.sla.core <= node.cores
            and node.mem_used + self.sla.mem <= node.mem
        ):
            bids.append(
                node.pricing_strat(
                    node,
                    sla=self.sla,
                    accumulated_latency=self.acc_latency,
                    caller=caller,
                )
            )

        return bids


class EdgeWardRequest(Request):
    def __init__(
        self,
        env,
        sla: SLA,
        network: Dict[Tuple[str, str], float],
        acc_latency=0.0,
    ) -> None:
        super().__init__()
        self.env = env
        self.acc_latency = acc_latency
        self.sla = sla
        self.network = network

    def __call__(self, node: FogNode, caller: str):
        bids = list()
        if node.parent is None:
            return []

        if (
            node.cores_used + self.sla.core <= node.cores
            and node.mem_used + self.sla.mem <= node.mem
        ):
            bids.append(
                node.pricing_strat(
                    node,
                    sla=self.sla,
                    accumulated_latency=self.acc_latency,
                    caller=caller,
                )
            )

        if len(bids) > 0:
            return bids

        parent = node.parent
        delay = self.network[(node.name, parent.name)]
        req = yield env.process(
            node.send(
                parent,
                EdgeWardRequest(
                    self.env,
                    self.sla,
                    self.network,
                    self.acc_latency + delay,
                ),
            )
        )

        return req


@dataclass(order=True)
class SortableFogNode:
    node: FogNode
    latency: float


class EdgeFirstRequest(Request):
    def __init__(
        self,
        env,
        sla: SLA,
        network: Dict[Tuple[str, str], float],
        acc_latency=0.0,
    ) -> None:
        super().__init__()
        self.env = env
        self.acc_latency = acc_latency
        self.sla = sla
        self.network = network

    def __call__(self, node: FogNode, caller: str):
        if (
            node.cores_used + self.sla.core <= node.cores
            and node.mem_used + self.sla.mem <= node.mem
        ):
            return [
                node.pricing_strat(
                    node,
                    sla=self.sla,
                    accumulated_latency=self.acc_latency,
                    caller=caller,
                )
            ]

        nodes: Any = []
        for child in node.children + [node.parent] if node.parent is not None else []:
            if child.name == caller:
                continue
            delay = self.network[(node.name, child.name)]
            if delay + self.acc_latency > self.sla.latency:
                continue
            nodes.append(SortableFogNode(child, self.acc_latency + delay))

        nodes = sorted(nodes, key=lambda x: x.latency)

        for nodepack in nodes:
            node = nodepack.node
            delay = nodepack.latency
            req = yield env.process(
                node.send(
                    child,
                    EdgeFirstRequest(
                        self.env,
                        self.sla,
                        self.network,
                        delay,
                    ),
                )
            )
            if len(req) != 0:
                return req

        return []


@dataclass
class EdgeFirstRequestData:
    distance: float
    bid: Bid


class EdgeFirstRequestTwo(Request):
    def __init__(
        self,
        env,
        sla: SLA,
        network: Dict[Tuple[str, str], float],
        acc_latency=0.0,
    ) -> None:
        super().__init__()
        self.env = env
        self.acc_latency = acc_latency
        self.sla = sla
        self.network = network

    def __call__(self, node: FogNode, caller: str):
        # yield self.env.timeout(0)  # Make the method a generator
        ret = yield self.env.process(self.call_inner(node, caller))
        if caller == "":
            return [ret.bid] if ret is not None else []
        return ret

    def call_inner(self, node: FogNode, caller: str):
        if (
            node.cores_used + self.sla.core <= node.cores
            and node.mem_used + self.sla.mem <= node.mem
        ):
            return EdgeFirstRequestData(
                self.acc_latency,
                node.pricing_strat(
                    node,
                    sla=self.sla,
                    accumulated_latency=self.acc_latency,
                    caller=caller,
                ),
            )

        nodes: Any = []
        for child in node.children + [node.parent] if node.parent is not None else []:
            if child.name == caller:
                continue
            delay = self.network[(node.name, child.name)]
            if delay + self.acc_latency > self.sla.latency:
                continue
            nodes.append(SortableFogNode(child, self.acc_latency + delay))

        nodes = sorted(nodes, key=lambda x: x.latency)

        for ii, nodepack in enumerate(nodes):
            node = nodepack.node
            delay = nodepack.latency
            req = yield env.process(
                node.send(
                    child,
                    EdgeFirstRequestTwo(
                        self.env,
                        self.sla,
                        self.network,
                        delay,
                    ),
                )
            )
            if req is None:
                continue
            if ii == len(nodes) - 1:
                return req
            if req.distance < nodes[ii + 1].latency:
                return req

        return None


@dataclass
class FurthestPlacementRequestData:
    rank: float
    bid: Bid


class FurthestPlacementRequest(Request):
    def __init__(
        self,
        env,
        sla: SLA,
        network: Dict[Tuple[str, str], float],
        acc_latency=0.0,
    ) -> None:
        super().__init__()
        self.env = env
        self.acc_latency = acc_latency
        self.sla = sla
        self.network = network

    def __call__(self, node: FogNode, caller: str):
        bids: Any = []
        requests: List[Any] = []
        for child in node.children + [node.parent] if node.parent is not None else []:
            if child.name == caller:
                continue
            delay = self.network[(node.name, child.name)]
            if delay + self.acc_latency <= self.sla.latency:
                req = env.process(
                    node.send(
                        child,
                        FurthestPlacementRequest(
                            self.env,
                            self.sla,
                            self.network,
                            self.acc_latency + delay,
                        ),
                    )
                )
                requests.append(req)

        for req in requests:
            req = yield req
            bids.extend(req)

        if (
            node.cores_used + self.sla.core <= node.cores
            and node.mem_used + self.sla.mem <= node.mem
        ):
            bids.append(
                FurthestPlacementRequestData(
                    LEVELS.get(node.name),
                    node.pricing_strat(
                        node,
                        sla=self.sla,
                        accumulated_latency=self.acc_latency,
                        caller=caller,
                    ),
                )
            )

        if caller == "":  # First call
            bids = sorted(bids, key=lambda x: x.rank)
            return [bids[0].bid] if len(bids) > 0 else []

        return bids


class CloudOnlyRequest(Request):
    def __init__(
        self, env, sla: SLA, network: Dict[Tuple[str, str], float], *args
    ) -> None:
        super().__init__()
        self.env = env
        self.sla = sla
        self.network = network

    def __call__(self, node: FogNode, caller: str):
        if node.parent is None:
            if (
                node.cores_used + self.sla.core <= node.cores
                and node.mem_used + self.sla.mem <= node.mem
            ):
                return [
                    node.pricing_strat(
                        node, sla=self.sla, accumulated_latency=0, caller=caller
                    )
                ]
            return []

        ret = yield env.process(
            node.send(node.parent, CloudOnlyRequest(self.env, self.sla, self.network))
        )
        return ret


class ProvisionRequest(Request):
    def __init__(
        self, env, monitoring: Monitoring, sla: SLA, price: float, acc_latency: float
    ) -> None:
        super().__init__()
        self.env = env
        self.price = price
        self.sla = sla
        self.monitoring = monitoring
        self.acc_latency = acc_latency

    def __call__(self, node: FogNode, _caller: str) -> Any:
        # yield self.env.timeout(0)
        if node.cores_used + self.sla.core > node.cores:
            return False
        if node.mem_used + self.sla.mem > node.mem:
            return False

        node.cores_used += self.sla.core
        node.mem_used += self.sla.mem
        node.provisioned.append(self.sla)

        self.monitoring.earnings[node.name].append(
            ProvisionedFunction(
                self.price,
                self.acc_latency,
                self.sla.latency,
                self.env.now,
                self.env.now + self.sla.duration,
                self.sla.core,
                node.cores,
            )
        )
        self.monitoring.currently_provisioned += 1
        self.monitoring.total_provisioned += 1

        yield self.env.timeout(self.sla.duration)

        node.cores_used -= self.sla.core
        node.mem_used -= self.sla.mem
        node.provisioned.remove(self.sla)

        self.monitoring.currently_provisioned -= 1
        return True


class FogNode(object):
    def __init__(
        self,
        env,
        network: Dict[Tuple[str, str], float],
        name: str,
        parent: FogNode | None,
        cores: float,
        mem: float,
        pricing_strat: Pricing,
    ):
        self.env = env
        self.network = network
        self.name = name
        self.children: List[FogNode] = []
        self.parent = parent

        self.cores = cores
        self.mem = mem
        self.cores_used = 0.0
        self.mem_used = 0.0
        self.provisioned: List[SLA] = []

        self.pricing_strat = pricing_strat

    def add_children(self, node: FogNode):
        node.parent = self
        self.children.append(node)

    def send(self, destination: FogNode, request: Request):
        yield self.env.timeout(self.network[(self.name, destination.name)])
        ret = yield self.env.process(request(destination, self.name))
        return ret


class MarketPlaceFirstAuction:
    def __init__(self, env, network: Dict[str, FogNode], monitoring: Monitoring):
        self.env = env
        self.network = network
        self.monitoring = monitoring

    def send(self, destination: str, request: Request):
        ret = yield env.process(request(self.network[destination], ""))
        return ret

    def auction(self, first_node: str, sla: SLA, auction: Request):
        bids: List[Bid] = yield self.env.process(self.send(first_node, auction))
        if len(bids) == 0:
            return

        bids = sorted(bids, key=lambda x: x.bid)

        winner = bids[0]
        ret = yield self.env.process(
            self.send(
                winner.bidder,
                ProvisionRequest(
                    env, self.monitoring, sla, winner.bid, winner.acc_latency
                ),
            )
        )
        return ret


class MarketPlaceSecondAuction:
    def __init__(self, env, network: Dict[str, FogNode], monitoring: Monitoring):
        self.env = env
        self.network = network
        self.monitoring = monitoring

    def send(self, destination: str, request: Request):
        ret = yield env.process(request(self.network[destination], ""))
        return ret

    def auction(self, first_node: str, sla: SLA, auction: Request):
        bids: List[Bid] = yield self.env.process(self.send(first_node, auction))
        if len(bids) == 0:
            return

        bids = sorted(bids, key=lambda x: x.bid)

        winner = bids[0]
        price = bids[1].bid if len(bids) > 1 else bids[0].bid
        # print(winner.bidder)
        ret = yield self.env.process(
            self.send(
                winner.bidder,
                ProvisionRequest(env, self.monitoring, sla, price, winner.acc_latency),
            )
        )
        return ret


def submit_function(
    env, network, marketplace, function: expe.Function, mon: Monitoring, strat_type
):
    #  CPU is in millicpu
    sla = SLA(
        function.mem, float(function.cpu) / 1000.0, function.latency * MS, 800 * SECS
    )
    yield env.timeout(function.sleep_before_start * SECS)
    mon.total_submitted += 1
    yield env.process(
        marketplace.auction(function.target_node, sla, strat_type(env, sla, network))
    )


def init_network(
    env, latencies, node, pricing_strategy, parent=None, flat_list={}, level=0
):
    children = node["children"] if "children" in node else []
    fog_node = FogNode(
        env,
        latencies,
        node["name"],
        parent,
        node["flavor"]["reserved_core"],
        node["flavor"]["reserved_mem"],
        # LinearPricing(2.0, level, 3, 20.0),
        pricing_strategy(level + 1),
    )
    flat_list[node["name"]] = fog_node

    for child in children:
        child, _ = init_network(
            env, latencies, child, pricing_strategy, node, flat_list, level + 1
        )
        fog_node.add_children(child)

    return fog_node, flat_list


def generate_latencies(net):
    ret = {}

    def gen_network_cb(source, destination, delay):
        ret[(source, destination)] = delay

    gen_net(net, gen_network_cb)

    return ret


def generate_initial_price_nikos(
    location: int,
    max_location: int,
    max_initial_price: float,
):
    # Ensure location is within the valid range (1 to max_location)
    location = min(max(1, location), max_location - location)

    # Calculate the base price as an inverse function of location
    # base_price = max_initial_price - (max_initial_price * (location - 1) / (max_location - 1))
    base_price = (
        (1 - max_initial_price) * location + max_location * max_initial_price - 1
    ) / (
        max_location - 1
    )  # to guarantee that price for location=1 is max_price and price for max_location is 1
    # Add random noise to the base price to create variation
    noise = (max_initial_price - 1) / (max_location - 1) * 2  # half of the slope
    random_variation = random.uniform(
        -noise, noise
    )  # Adjust the range of variation as needed

    # Calculate the final price by adding random variation to the base price
    price = base_price + random_variation

    # Ensure the price is within the desired range (1 to max_initial_price )
    price = min(max(1, price), max_initial_price)
    return price


def generate_initial_price(
    location: int,
    max_location: int,
    max_initial_price: float,
):
    # Ensure location is within the valid range (1 to max_location)
    location = min(max(0, location), max_location - location)

    # Calculate the base price as an inverse function of location
    # base_price = max_initial_price - (max_initial_price * (location - 1) / (max_location - 1))
    base_price = (max_initial_price * location + max_location * max_initial_price) / (
        max_location
    )  # to guarantee that price for location=1 is max_price and price for max_location is 1
    # Add random noise to the base price to create variation
    noise = max_initial_price / max_location * 2  # half of the slope
    random_variation = random.uniform(
        -noise, noise
    )  # Adjust the range of variation as needed

    # Calculate the final price by adding random variation to the base price
    price = base_price + random_variation

    # Ensure the price is within the desired range (1 to max_initial_price )
    price = min(max(0, price), max_initial_price)
    return price


def choose_from(env_var, mapping):
    key = os.getenv(env_var, "")
    ret = mapping.get(key)
    if ret is None:
        print(f"{env_var} not in [{' '.join(list(mapping.keys()))}]")
        print(
            f"{env_var} ({key}) not in [{' '.join(list(mapping.keys()))}]",
            file=sys.stderr,
        )
    else:
        print(f"Using {env_var} {key}", file=sys.stderr)
    return ret, key


monitoring = Monitoring()

# Setup and start the simulation
max_level = 0
for level in LEVELS.values():
    max_level = max(max_level, level)

latencies = generate_latencies(NETWORK)

placement_strategy, placement_strategy_name = choose_from(
    "PLACEMENT_STRATEGY",
    {
        "auction": AuctionBidRequest,
        "edge_first": EdgeFirstRequest,
        "edge_first2": EdgeFirstRequestTwo,
        "edge_ward": EdgeWardRequest,
        "furthest": FurthestPlacementRequest,
        "cloud_only": CloudOnlyRequest,
    },
)

pricing_strategy, pricing_strategy_name = choose_from(
    "PRICING_STRATEGY",
    {
        "same": functools.partial(lambda _: ConstantPricing(1.0)),
        "constant": ConstantPricing,
        "constant_rnd": functools.partial(
            lambda _: ConstantPricing(
                lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "random": functools.partial(lambda _: RandomPricing(10.0)),
        "linear_random_uniform": functools.partial(
            lambda level: LinearUniformRandomPricing(
                8.0,
                #   lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear_random_normal": functools.partial(
            lambda level: LinearNormalRandomPricing(
                8.0, lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear_initial": functools.partial(
            lambda level: LinearPricing(
                8.0, lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear": functools.partial(
            lambda level: LinearPricing(
                8.0,
                # lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear_low": functools.partial(
            lambda level: LinearPricing(
                1.0,
                # lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear_mid": functools.partial(
            lambda level: LinearPricing(
                2.0,
                # lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear_high": functools.partial(
            lambda level: LinearPricing(
                4.0,
                # lambda: generate_initial_price(level, max_level, 10.0)
            )
        ),
        "linear_wo_init": functools.partial(lambda _: LinearPricing(8.0)),
        "linear_part": functools.partial(
            lambda level: LinearPerPartPricing(
                [1.0, 2.0, 8.0],
                [0.2, 0.5],
                lambda: generate_initial_price(level, max_level, 10.0),
            )
        ),
        "linear_latency": functools.partial(
            lambda level: LinearLatencySafePricing(
                latencies,
                slopes=[2.0, 8.0, 4.0, 2.0],
                breaking_points=[0.33, 0.5, 0.8],
                latency_slopes=[1, 8, 16.0],
                latency_breaking_points=[0.7, 0.9],
                # initial_price=lambda: generate_initial_price(level, max_level, 10.0),
                initial_price=0,
            )
        ),
        "linear_part_wo_init": functools.partial(
            lambda _: LinearPerPartPricing(
                [1.0, 2.0, 8.0],
                [0.2, 0.5],
            )
        ),
        "exp": functools.partial(lambda _: ExpPricing(2.2)),
        "exp_lat": functools.partial(lambda _: ExpPricingLatency(2.2, 2.2)),
        "logistic": functools.partial(lambda _: LogisticPricing(0, 1, 1, 10, 0.9, 5)),
        "logistic_inv": functools.partial(
            lambda _: LogisticPricing(1, 0, 1, 10, 0.9, 5)
        ),
        "logistic_acc": functools.partial(
            lambda _: LogisticAccPricing(0, 1, 1, 10, 0.9, 5)
        ),
        "logistic_acc_min": functools.partial(
            lambda _: LogisticAccMinPricing(0, 1, 1, 10, 0.9, 5)
        ),
        "logistic_acc_max": functools.partial(
            lambda _: LogisticAccMaxPricing(0, 1, 1, 10, 0.9, 5)
        ),
        "logistic_acc_sum": functools.partial(
            lambda _: LogisticAccSumPricing(0, 1, 1, 10, 0.9, 5)
        ),
    },
)
marketplace_strategy, marketplace_strategy_name = choose_from(
    "MARKETPLACE_STRATEGY",
    {
        "first_price": MarketPlaceFirstAuction,
        "second_price": MarketPlaceSecondAuction,
    },
)

if not pricing_strategy or not placement_strategy or not marketplace_strategy:
    exit(1)

env = simpy.Environment()

_first_node, network = init_network(env, latencies, NETWORK, pricing_strategy)

marketplace = marketplace_strategy(env, network, monitoring)

nb_submitted_functions_low_latency = 0
nb_submitted_functions_high_latency = 0
functions = expe.load_functions(os.getenv("EXPE_SAVE_FILE"))
for function in functions:
    # Use the id of the node instead of its name
    function.target_node = function.target_node.replace("'", "")
    if function.latency_type == "low":
        nb_submitted_functions_low_latency += 1
    else:
        nb_submitted_functions_high_latency += 1
    env.process(
        submit_function(
            env, latencies, marketplace, function, monitoring, placement_strategy
        )
    )

# Execute!
JOB_INDEX_STR = os.getenv(
    "JOB_INDEX", "0"
)  # 0 means not running in gnuparallel ; if string is composed of 11...11111 then it means that its the first element
JOB_INDEX = int(JOB_INDEX_STR)
if JOB_INDEX == 11:
    JOB_INDEX = 1

if JOB_INDEX == 0:
    with alive_bar(
        SIM_TIME, title="Simulating...", ctrl_c=False, dual_line=True, file=sys.stderr
    ) as bar:
        for ii in range(SIM_TIME):
            if ii % (SECS) == 0:
                bar.text = (
                    f"--> Currently provisioned {monitoring.currently_provisioned},"
                    "failed to provision "
                    f"{monitoring.total_submitted - monitoring.total_provisioned},"
                    f" total is {monitoring.total_submitted}..."
                )
                bar(SECS)
            env.run(until=1 + ii)
else:
    env.run(SIM_TIME)


print(
    (
        f"--> Done {monitoring.total_provisioned},"
        f" failed to provision {monitoring.total_submitted - monitoring.total_provisioned} functions; "
        f"total is {monitoring.total_submitted}."
    ),
    file=sys.stderr,
)

earnings: List[List[List[float]]] = [[] for _ in range(max_level + 1)]
provisioned: List[List[float]] = [[] for _ in range(max_level + 1)]
count = [0] * (max_level + 1)

for node, level in LEVELS.items():
    earnings[level].append([ee.bid for ee in monitoring.earnings[node]])
    provisioned[level].append(len(monitoring.earnings[node]))
    count[level] += 1


def quantile(x, q):
    if len(x) == 0:
        return float("nan")
    return numpy.quantile(x, q)


if JOB_INDEX == 0:
    for ii in range(max_level + 1):
        earn = functools.reduce(lambda x, y: x + y, earnings[ii])
        prov = provisioned[ii]
        print(
            (
                f"Lvl {ii} ({count[ii]} nodes):\n"
                f"Earnings: tot: {numpy.sum(earn):.2f} "
                f"med: {numpy.median(earn):.2f} [{quantile(earn, .025):.2f},{quantile(earn, .975):.2f}] "
                f"avg: {numpy.mean(earn):.2f}\n"
                f"Provisioned: tot: {numpy.sum(prov)} "
                f"med: {numpy.median(prov):.2f} [{quantile(prov, .025):.2f},{quantile(prov, .975):.2f}] "
                f"avg: {numpy.mean(prov):.2f}"
            ),
            file=sys.stderr,
        )

out_dir = os.getenv("CSV_OUT_DIR", "./")
seed = RANDOM_SEED or int(-1)
JOB_ID = os.getenv("JOB_ID", "0-0")
with open(f"{out_dir}/{JOB_INDEX}.data.csv", "w") as file:
    writer = csv.writer(file, delimiter="\t")
    if JOB_INDEX <= 1:
        writer.writerow(
            [
                "job_id",
                "node",
                "placement",
                "pricing",
                "marketplace",
                "seed",
                "level",
                "earning",
                "acc_latency",
                "latency",
                "timestamp_start",
                "timestamp_end",
                "node_cpu_reservation_sla",
                "node_cpu",
                "sim_time",
                "nb_submitted_functions_low_latency",
                "nb_submitted_functions_high_latency",
                "nb_nodes",
            ]
        )
    nb_nodes = len(FOG_NODES)
    for node, level in LEVELS.items():
        for ee in monitoring.earnings[node]:
            writer.writerow(
                [
                    JOB_ID,
                    node,
                    placement_strategy_name,
                    pricing_strategy_name,
                    marketplace_strategy_name,
                    seed,
                    level,
                    ee.bid,
                    ee.acc_latency,
                    ee.latency,
                    ee.timestamp_start,
                    min(ee.timestamp_end, SIM_TIME),
                    ee.cpu_reservation_used_sla,
                    ee.node_cpu,
                    SIM_TIME,
                    nb_submitted_functions_low_latency,
                    nb_submitted_functions_high_latency,
                    nb_nodes,
                ]
            )

with open(f"{out_dir}/{JOB_INDEX}.levels.csv", "w") as file:
    writer = csv.writer(file, delimiter="\t")
    if JOB_INDEX <= 1:
        writer.writerow(["job_id", "level", "node"])
    for node, level in LEVELS.items():
        writer.writerow(
            [
                JOB_ID,
                level,
                node,
            ]
        )
