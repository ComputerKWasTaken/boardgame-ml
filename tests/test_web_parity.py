"""Cockpit/CLI config parity tests — the guard against silent drift.

The web cockpit builds training configs in Python
(:func:`boardgame_ml.web.jobs.build_config`); the CLI builds them
from the shipped ``configs/*.yaml``. These are two descriptions of
the same runs, and the legacy project's core disease was exactly
this: two config paths that quietly disagreed, so what you launched
was not what you thought you launched (autopsy finding #2).

The concrete trap this guards: racing's shipped configs carry
``gamma: 0.995`` — the braking horizon (DESIGN.md, M5). The
cockpit's original builder never set gamma at all, so a cockpit
racing run silently used the framework default 0.99, under which
the car provably never learns to brake. Likewise Match 3 needs its
MultiInputPolicy, [256,256] nets, board env_kwargs, and 8 envs from
the YAML, none of which the cockpit originally passed.

So: for every shipped YAML, ``build_config`` for the same
(game, algorithm) pair must reproduce it on every load-bearing
field. The comparison reads the YAML files themselves — editing a
shipped config's field without updating the cockpit's table (or
vice versa) turns this suite red, by design.

A shipped YAML whose filename stem names a **variant** in
``PAIR_VARIANTS`` (M7's generalist racing configs — same pair as
the canonical run, different ``env_kwargs``) is routed through
``build_config(..., variant=stem)`` instead of the bare pair, so
the one-description-per-config rule survives having two shipped
configs for one pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.web.jobs import (
    GAME_ALGORITHMS,
    PAIR_DEFAULTS,
    PAIR_VARIANTS,
    PRESETS,
    build_config,
    default_algorithm,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

#: The fields a run's behavior actually depends on. (total_timesteps,
#: eval cadence, and output_dir are budget/presentation choices the
#: cockpit legitimately drives from its presets and UI.)
PARITY_FIELDS = (
    "gamma",
    "policy",
    "policy_kwargs",
    "env_kwargs",
    "n_envs",
    "learning_rate",
    "hyperparams",
    "self_play",
)


def _shipped_configs() -> list[tuple[str, ExperimentConfig]]:
    return [
        (path.name, ExperimentConfig.from_yaml(path))
        for path in sorted(CONFIGS_DIR.glob("*.yaml"))
    ]


_SHIPPED = _shipped_configs()


@pytest.mark.parametrize(
    "filename,shipped", _SHIPPED, ids=[name for name, _ in _SHIPPED]
)
def test_shipped_pair_is_representable(filename, shipped):
    """Every shipped YAML's (game, algorithm) pair has a cockpit
    defaults-table entry — no shipped run is CLI-only."""
    assert (shipped.game, shipped.algorithm) in PAIR_DEFAULTS, (
        f"{filename}: pair ({shipped.game!r}, {shipped.algorithm!r}) "
        "missing from web.jobs.PAIR_DEFAULTS"
    )
    assert shipped.algorithm in GAME_ALGORITHMS[shipped.game]


@pytest.mark.parametrize(
    "filename,shipped", _SHIPPED, ids=[name for name, _ in _SHIPPED]
)
def test_build_config_matches_shipped_yaml(filename, shipped):
    """build_config(game, algorithm) == the shipped YAML on every
    load-bearing field (variant YAMLs route through their variant)."""
    stem = Path(filename).stem
    variant = stem if stem in PAIR_VARIANTS else None
    built = build_config(
        game=shipped.game, algorithm=shipped.algorithm, variant=variant
    )
    for field in PARITY_FIELDS:
        assert getattr(built, field) == getattr(shipped, field), (
            f"{filename}: cockpit build_config {field}="
            f"{getattr(built, field)!r} != shipped YAML "
            f"{getattr(shipped, field)!r} — update PAIR_DEFAULTS / "
            "PAIR_VARIANTS or the YAML so the two agree"
        )


def test_variants_have_shipped_yamls_and_vice_versa():
    """Every PAIR_VARIANTS key is a shipped config's stem (no orphan
    variants), and the variant's declared pair matches that YAML's."""
    stems = {Path(name).stem: cfg for name, cfg in _SHIPPED}
    for stem, spec in PAIR_VARIANTS.items():
        assert stem in stems, f"variant {stem!r} has no shipped YAML"
        shipped = stems[stem]
        assert (spec["game"], spec["algorithm"]) == (
            shipped.game,
            shipped.algorithm,
        )


def test_generalist_variants_carry_the_track_distribution():
    """The M7 trap, named: the generalist configs differ from the
    canonical racing runs ONLY by env_kwargs — a fresh v2 circuit
    per episode from the training range. A variant that quietly
    dropped the randomization (or leaked the exam range) would
    train a specialist while claiming to train a generalist."""
    for stem, algorithm in (
        ("racing_sac_generalist", "sac"),
        ("racing_ppo_generalist", "ppo"),
    ):
        cfg = build_config(game="racing", algorithm=algorithm, variant=stem)
        assert cfg.env_kwargs == {
            "track_version": 2,
            "randomize_track": True,
            "track_seed_lo": 10_000,
            "track_seed_hi": 11_000,
        }
        # Everything else is the pair's canonical recipe.
        bare = build_config(game="racing", algorithm=algorithm)
        for field in PARITY_FIELDS:
            if field != "env_kwargs":
                assert getattr(cfg, field) == getattr(bare, field)


def test_variant_rejects_wrong_pair_and_unknown_names():
    with pytest.raises(ValueError, match="not \('racing', 'ppo'\)"):
        build_config(
            game="racing", algorithm="ppo", variant="racing_sac_generalist"
        )
    with pytest.raises(ValueError, match="Unknown config variant"):
        build_config(game="racing", algorithm="sac", variant="nope")


def test_match3_cockpit_config_is_the_shipped_one():
    """The M3 trap, named: Match 3 from the cockpit must carry the
    Dict-observation policy, the big nets, the board parameters,
    and 8 envs — or it cannot learn (and once could not even run)."""
    cfg = build_config(game="match3", algorithm="maskable_ppo")
    assert cfg.policy == "MultiInputPolicy"
    assert cfg.policy_kwargs == {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]}
    }
    assert cfg.env_kwargs == {
        "board_rows": 8,
        "board_cols": 8,
        "num_gems": 6,
        "max_moves": 20,
    }
    assert cfg.n_envs == 8
    assert cfg.hyperparams["batch_size"] == 512


def test_connect_four_cockpit_config_carries_the_league():
    """The M6 trap, named: Connect Four from the cockpit must carry
    the self-play league (mixture, snapshot cadence, pool cap) —
    without the ``self_play`` block the trainer would build a
    single-agent run against an empty seat and learn nothing."""
    cfg = build_config(game="connect_four", algorithm="maskable_ppo")
    assert cfg.self_play["snapshot_interval"] == 10_000
    assert cfg.self_play["snapshot_share"] == 0.55
    assert cfg.self_play["heuristic_share"] == 0.2
    assert cfg.self_play["heuristic"] == "connect_four_greedy_noisy"
    assert cfg.self_play["search"] == "connect_four_minimax"
    assert cfg.self_play["search_share"] == 0.1
    assert cfg.self_play["pool_max"] == 24
    assert cfg.policy_kwargs["net_arch"] == [256, 256]
    assert cfg.n_envs == 8
    shipped = ExperimentConfig.from_yaml(
        CONFIGS_DIR / "connect_four_maskable_ppo.yaml"
    )
    assert cfg.self_play == shipped.self_play


def test_racing_sac_cockpit_config_is_the_shipped_one():
    """The M5 trap, named: cockpit SAC must not fall back to gamma
    0.99 (the car never brakes), 4 envs, or SB3's 1M... buffer
    defaults — it must be the shipped experiment."""
    cfg = build_config(game="racing", algorithm="sac")
    assert cfg.gamma == 0.995
    assert cfg.n_envs == 1
    assert cfg.env_kwargs == {"track_seed": 7}
    assert cfg.policy_kwargs == {"net_arch": [256, 256]}
    assert cfg.hyperparams["buffer_size"] == 300_000
    assert cfg.hyperparams["batch_size"] == 256
    assert cfg.hyperparams["tau"] == 0.005
    assert cfg.hyperparams["ent_coef"] == "auto"
    assert cfg.hyperparams["learning_starts"] == 5000


def test_racing_ppo_cockpit_config_carries_braking_gamma():
    cfg = build_config(game="racing", algorithm="ppo")
    assert cfg.gamma == 0.995
    assert cfg.hyperparams["gae_lambda"] == 0.98
    assert cfg.hyperparams["ent_coef"] == 0.005


def test_presets_cover_every_game_and_the_sac_pair():
    """Every game resolves a budget in every preset (no silent
    fallback for a listed game), and racing's SAC-scaled budgets
    exist — SAC steps cost ~30x PPO's on CPU, so the PPO numbers
    must not leak into a SAC job."""
    from boardgame_ml.envs import GAMES

    for preset, tables in PRESETS.items():
        for game in GAMES:
            assert game in tables["timesteps"], (preset, game)
    standard = PRESETS["standard"]["timesteps"]
    assert standard["match3"] == 700_000  # the shipped budget
    assert standard["connect_four"] == 2_000_000  # the shipped M6 budget
    assert standard["racing"] == 600_000  # the shipped PPO budget
    assert standard["racing:sac"] == 150_000  # the shipped SAC run
    assert PRESETS["quick"]["timesteps"]["racing:sac"] == 50_000
    assert PRESETS["long"]["timesteps"]["racing:sac"] == 300_000
    assert PRESETS["quick"]["timesteps"]["match3"] == 100_000
    assert PRESETS["long"]["timesteps"]["match3"] == 1_000_000
    # Match 3's shipped eval cadence survives in every preset.
    for preset in PRESETS:
        assert PRESETS[preset]["eval_freq"]["match3"] == 25_000


def test_default_algorithm_choices():
    assert default_algorithm("racing") == "sac"  # recommended default
    assert default_algorithm("blackjack") == "maskable_ppo"
    assert default_algorithm("match3") == "maskable_ppo"
    assert default_algorithm("flappy") == "ppo"
    assert GAME_ALGORITHMS["racing"] == ["ppo", "sac"]
    assert GAME_ALGORITHMS["blackjack"] == ["maskable_ppo", "dqn"]


def test_user_overrides_still_win_over_the_table():
    """The table supplies defaults; explicit UI fields override —
    and only the fields they name (gamma stays the shipped 0.995)."""
    cfg = build_config(
        game="racing",
        algorithm="sac",
        timesteps=1234,
        learning_rate=0.5,
        n_envs=3,
        eval_freq=100,
        eval_episodes=2,
    )
    assert cfg.total_timesteps == 1234
    assert cfg.learning_rate == 0.5
    assert cfg.n_envs == 3
    assert cfg.eval_freq == 100
    assert cfg.eval_episodes == 2
    assert cfg.gamma == 0.995
    assert cfg.hyperparams["buffer_size"] == 300_000


def test_pair_defaults_are_copies_not_shared_state():
    """Mutating one built config must not corrupt the table (and
    therefore the next job, or /api/meta's readout)."""
    cfg = build_config(game="match3", algorithm="maskable_ppo")
    cfg.hyperparams["batch_size"] = -1
    cfg.policy_kwargs["net_arch"]["pi"].append(-1)
    fresh = build_config(game="match3", algorithm="maskable_ppo")
    assert fresh.hyperparams["batch_size"] == 512
    assert fresh.policy_kwargs["net_arch"]["pi"] == [256, 256]


def test_meta_endpoint_exposes_effective_settings(tmp_path):
    """The Train view's readout is fed by /api/meta: per-game
    algorithm lists, the preselected defaults, and the effective
    per-pair settings themselves."""
    from fastapi.testclient import TestClient

    from boardgame_ml.web.app import create_app

    meta = TestClient(create_app(tmp_path)).get("/api/meta").json()
    assert meta["game_algorithms"]["match3"] == ["maskable_ppo"]
    assert set(meta["game_algorithms"]["racing"]) == {"ppo", "sac"}
    assert set(meta["game_algorithms"]) == {
        "blackjack",
        "higher_or_lower",
        "match3",
        "flappy",
        "racing",
        "connect_four",
        "pong",
        "game2048",
    }
    assert meta["game_algorithms"]["game2048"] == ["maskable_ppo"]
    g48 = meta["pair_defaults"]["game2048:maskable_ppo"]
    assert g48["gamma"] == 0.995
    assert g48["n_envs"] == 8
    assert meta["default_algorithms"]["racing"] == "sac"
    sac = meta["pair_defaults"]["racing:sac"]
    assert sac["gamma"] == 0.995
    assert sac["n_envs"] == 1
    assert sac["hyperparams"]["buffer_size"] == 300_000
    m3 = meta["pair_defaults"]["match3:maskable_ppo"]
    assert m3["policy"] == "MultiInputPolicy"
    assert m3["env_kwargs"]["num_gems"] == 6
    # Preset timesteps resolve per game and per pair.
    standard = meta["presets"]["standard"]["timesteps"]
    assert standard["match3"] == 700_000
    assert standard["racing:sac"] == 150_000
