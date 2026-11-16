from pathlib import Path

import pytest

from world.build import build_dataset
from world.config import WorldConfig
from world.kb import (
    PROP_BOOL,
    PROP_CAT,
    PROP_INDIVIDUAL,
    REL_FUNCTIONAL,
    REL_INVERSE,
    REL_SYMMETRIC,
    REL_TRANSITIVE,
    World,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "world"


@pytest.fixture(scope="session")
def tiny_cfg() -> WorldConfig:
    return WorldConfig.from_yaml(CONFIGS / "tiny.yaml")


@pytest.fixture(scope="session")
def tiny(tiny_cfg):
    return build_dataset(tiny_cfg)


@pytest.fixture(scope="session")
def default_ds():
    return build_dataset(WorldConfig.from_yaml(CONFIGS / "default.yaml"))


@pytest.fixture(params=["tiny", "default"])
def ds(request, tiny, default_ds):
    return {"tiny": tiny, "default": default_ds}[request.param]


@pytest.fixture
def mini() -> World:
    """Hand-built world with known answers.

    animal(C0) ─ bird(C1) ─ penguin(C2): pingu(E0), robo(E3)
               │          └ sparrow(C3): tweety(E1)
               └ mammal(C4) ─ dog(C5):   rex(E2)

    flies(P0):  bird=True, penguin=False (override), robo=True (override of override)
    fur(P1):    mammal=True
    color(P2):  animal=V0, rex=V1 (override)
    lucky(P3):  individual; tweety=True
    R0 functional: E0->E1->E2->E3->E0
    R1 symmetric:  E0-E1 (stated E0,E1), E2-E3 (stated E3,E2)
    R2/R3 inverse pair, R2 = (E0 E2)(E1 E3); stated: R2(E0,E2) R3(E3,E1) R2(E2,E0) R3(E1,E3)
    R4 transitive: E0 -> E1 -> E2, E3 -> E2
    """
    cats = ["C0", "C1", "C2", "C3", "C4", "C5"]
    parent = {"C0": None, "C1": "C0", "C2": "C1", "C3": "C1", "C4": "C0", "C5": "C4"}
    level = {"C0": 1, "C1": 2, "C2": 3, "C3": 3, "C4": 2, "C5": 3}
    leaf = {"E0": "C2", "E1": "C3", "E2": "C5", "E3": "C2"}
    facts = [("isa", e, c) for e, c in leaf.items()]
    facts += [("sub", c, p) for c, p in parent.items() if p]
    facts += [
        ("prop", "C1", "P0", True),
        ("prop", "C2", "P0", False),
        ("prop", "E3", "P0", True),
        ("prop", "C4", "P1", True),
        ("prop", "C0", "P2", "V0"),
        ("prop", "E2", "P2", "V1"),
        ("prop", "E1", "P3", True),
        ("rel", "R0", "E0", "E1"),
        ("rel", "R0", "E1", "E2"),
        ("rel", "R0", "E2", "E3"),
        ("rel", "R0", "E3", "E0"),
        ("rel", "R1", "E0", "E1"),
        ("rel", "R1", "E3", "E2"),
        ("rel", "R2", "E0", "E2"),
        ("rel", "R3", "E3", "E1"),
        ("rel", "R2", "E2", "E0"),
        ("rel", "R3", "E1", "E3"),
        ("rel", "R4", "E0", "E1"),
        ("rel", "R4", "E1", "E2"),
        ("rel", "R4", "E3", "E2"),
    ]
    ids = list(leaf) + cats + ["P0", "P1", "P2", "P3", "V0", "V1", "R0", "R1", "R2", "R3", "R4"]
    return World(
        config=WorldConfig(),
        lexicon={i: i for i in ids},
        entities=list(leaf),
        categories=cats,
        cat_parent=parent,
        cat_level=level,
        entity_leaf=leaf,
        properties={
            "P0": {"kind": PROP_BOOL},
            "P1": {"kind": PROP_BOOL},
            "P2": {"kind": PROP_CAT, "values": ["V0", "V1"]},
            "P3": {"kind": PROP_INDIVIDUAL},
        },
        relations={
            "R0": {"kind": REL_FUNCTIONAL, "inverse": None},
            "R1": {"kind": REL_SYMMETRIC, "inverse": None},
            "R2": {"kind": REL_INVERSE, "inverse": "R3"},
            "R3": {"kind": REL_INVERSE, "inverse": "R2"},
            "R4": {"kind": REL_TRANSITIVE, "inverse": None},
        },
        facts=facts,
    )
