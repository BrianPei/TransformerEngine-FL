#!/usr/bin/env python3
"""FlagCICD plan generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
import generate_matrix as gm
import flagcicd_contract as contract

LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
RISK_ORDER = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4}


def load_project(path: Path) -> dict[str, Any]:
    project = yaml.safe_load(path.read_text())
    if project.get("schema_version") != "flagcicd.project/v1":
        raise ValueError("project.yaml must be flagcicd.project/v1")
    return project


def load_platforms(configs_dir: Path) -> dict[str, Any]:
    """Load platform configs with hashes."""
    platforms = {}
    for path in sorted(configs_dir.glob("*.yml")):
        if path.stem in gm._NON_PLATFORM_STEMS:
            continue
        config_bytes = path.read_bytes()
        config = yaml.safe_load(config_bytes)
        platforms[path.stem] = {
            "config": config,
            "config_hash": hashlib.sha256(config_bytes).hexdigest(),
            "config_path": path.relative_to(configs_dir.parent).as_posix(),
        }
    return platforms


def select_platforms(
    stage: str,
    platforms: dict,
    changed: list[str],
    labels: list[str],
    requested: list[str],
) -> tuple[list[str], bool]:
    """Select platforms, return (selected, lint_only)."""
    all_names = sorted(platforms.keys())

    if stage == "pr":
        watch = {n: platforms[n]["config"].get("paths_watch", []) for n in all_names}
        selected, _ = gm.select_platforms(watch, changed, "pull_request", labels)
        lint_only = len(selected) == 0 and len(changed) > 0
        return selected, lint_only

    if stage in ("branch", "nightly", "release"):
        return all_names, False

    if stage == "manual":
        if not requested:
            raise ValueError("manual stage requires requested platforms")
        unknown = set(requested) - set(all_names)
        if unknown:
            raise ValueError(f"Unknown platforms: {', '.join(sorted(unknown))}")
        return sorted(set(requested)), False

    raise ValueError(f"Unknown stage: {stage}")


def match_risk_rules(changed: list[str], risk_rules: dict) -> tuple[str, str]:
    """Match changed files against risk rules.

    Returns: (minimum_level, risk_level)
    """
    import fnmatch

    if not changed:
        return "L0", "R0"

    max_level = "L0"
    max_risk = "R0"

    for rule in risk_rules.get("rules", []):
        patterns = rule.get("paths", [])
        matched = any(any(fnmatch.fnmatch(path, pat) for pat in patterns) for path in changed)

        if matched:
            rule_level = rule.get("minimum_level", "L0")
            rule_risk = rule.get("risk", "R0")
            if rule_level not in LEVEL_ORDER:
                raise ValueError(f"Unknown risk-rule level: {rule_level}")
            if rule_risk not in RISK_ORDER:
                raise ValueError(f"Unknown risk-rule risk: {rule_risk}")

            if LEVEL_ORDER[rule_level] > LEVEL_ORDER[max_level]:
                max_level = rule_level
            if RISK_ORDER[rule_risk] > RISK_ORDER[max_risk]:
                max_risk = rule_risk

    # If no rules matched, use unknown defaults
    if max_level == "L0":
        max_level = risk_rules.get("unknown_level", "L2")
        max_risk = risk_rules.get("unknown_risk", "R2")

    return max_level, max_risk


def resolve_level(
    stage_def: dict, requested: str, project: dict, changed: list[str], labels: list[str]
) -> tuple[str, str]:
    """Resolve test level and risk level.

    Returns: (test_level, risk_level)
    """
    if stage_def.get("fixed_level"):
        return stage_def["default_level"], stage_def.get("default_risk", "R1")

    # Start with stage default
    test_level = stage_def["default_level"] if requested == "auto" else requested
    if test_level not in LEVEL_ORDER:
        raise ValueError(f"Unknown requested test level: {requested}")
    risk_level = stage_def.get("default_risk", "R1")

    # Apply risk rules
    if changed and "risk_rules" in project:
        rule_level, rule_risk = match_risk_rules(changed, project["risk_rules"])
        if LEVEL_ORDER[rule_level] > LEVEL_ORDER[test_level]:
            test_level = rule_level
        if RISK_ORDER[rule_risk] > RISK_ORDER[risk_level]:
            risk_level = rule_risk

    # Label-based escalation
    for label in labels:
        if label.startswith("level:"):
            requested_level = label.split(":", 1)[1].upper()
            if requested_level not in LEVEL_ORDER:
                raise ValueError(f"Unknown level label: {label}")
            if LEVEL_ORDER[requested_level] > LEVEL_ORDER[test_level]:
                test_level = requested_level

    return test_level, risk_level


def group_matches_paths(group_def: dict, changed: list[str], platform: str) -> bool:
    """Check if test group should run based on paths_watch.

    Returns: True if group should run (paths_watch empty or matched)
    """
    import fnmatch

    paths_watch = group_def.get("paths_watch", [])
    if not paths_watch:
        return True  # No filter means always run

    if not changed:
        return True  # No diff means conservative - run all

    # CI/config changes should always trigger tests (bypass paths_watch)
    ci_config_patterns = [".github/", ".flagcicd/", "setup.py", "pyproject.toml", "CMakeLists.txt"]
    for path in changed:
        for pattern in ci_config_patterns:
            if path.startswith(pattern):
                return True

    # Expand {platform} placeholder
    expanded = [p.replace("{platform}", platform) for p in paths_watch]

    # Check if any changed file matches any pattern
    for path in changed:
        for pattern in expanded:
            if pattern.endswith("/"):
                # Directory prefix match
                if path.startswith(pattern):
                    return True
            else:
                # Exact or glob match
                if fnmatch.fnmatch(path, pattern):
                    return True

    return False


def build_test_groups(
    project: dict,
    platforms: dict,
    selected_platforms: list[str],
    test_level: str,
    changed: list[str],
    blocked: list[dict[str, Any]] | None = None,
) -> list[dict]:
    """Build test_groups list from catalog, filtered by level and paths_watch."""
    catalog = project["test_groups"]
    level_num = LEVEL_ORDER[test_level]
    groups = []

    for platform in selected_platforms:
        platform_cfg = platforms[platform]["config"]
        device_types = platform_cfg.get("device_types", [])
        nproc = platform_cfg.get("nproc_per_node", 1)

        # Preflight always included at L0
        if level_num >= 0:
            for device in device_types:
                groups.append(
                    {
                        "platform": platform,
                        "device": device,
                        "group": "preflight",
                        "kind": "preflight",
                        "level": "L0",
                        "required": True,
                        "lane": "preflight",
                        "resources": {
                            "class": "shared_cpu",
                            "devices": 0,
                            "exclusive": False,
                            "timeout_minutes": 90,
                        },
                        "coverage": False,
                        "exec": None,
                    }
                )

        # Catalog groups
        for group_def in catalog:
            group_level = group_def.get("level", "L1")
            if LEVEL_ORDER[group_level] > level_num:
                continue  # Out of scope

            name = group_def["name"]
            if name == "preflight":
                continue  # Already added

            # Check blocked at catalog level
            catalog_blocked = group_def.get("blocked")

            # Check paths_watch filter (skip if blocked)
            if not catalog_blocked and not group_matches_paths(group_def, changed, platform):
                continue  # Changed files don't match this group's watch paths

            # Check if platform has matrix entry
            matrix_key = group_def.get("matrix")
            exec_blocked = None
            if matrix_key:
                matrix_entries = platform_cfg.get(matrix_key, [])
                matched = [e for e in matrix_entries if e.get("name") == name]

                if not matched:
                    if group_def.get("required") and not catalog_blocked:
                        raise ValueError(
                            f"{platform} missing required group {name} in {matrix_key}"
                        )
                    continue  # Excluded: no matrix entry

                if len(matched) > 1:
                    raise ValueError(f"{platform} has duplicate {name} in {matrix_key}")

                exec_entry = dict(matched[0])

                # Check for exec-level blocking
                exec_blocked = exec_entry.get("blocked")

                # Strip planner-only keys
                exec_entry.pop("timeout_minutes", None)
                exec_entry.pop("blocked", None)
                exec_entry.pop("name", None)
            else:
                exec_entry = None

            # Determine if blocked
            blocked_reason = catalog_blocked or exec_blocked

            # Keep blocked groups visible in the plan.  They are not executable
            # matrix entries, but silently dropping them makes a release plan
            # look complete and prevents the gate from explaining the gap.
            if blocked_reason:
                if blocked is not None:
                    for device in device_types:
                        blocked.append(
                            {
                                "platform": platform,
                                "device": device,
                                "group": name,
                                "kind": group_def["kind"],
                                "level": group_level,
                                "required": group_def.get("required", False),
                                "reason": str(blocked_reason),
                            }
                        )
                continue

            # Resolve resources
            res_class = group_def["resources"]["class"]
            class_def = project["resource_classes"][res_class]
            devices = nproc if class_def["devices"] == "nproc_per_node" else class_def["devices"]
            exclusive = class_def["exclusive"]
            timeout = group_def["resources"].get("timeout_minutes", 90)

            lane = "serial" if exclusive or devices > 1 else "parallel"

            for device in device_types:
                groups.append(
                    {
                        "platform": platform,
                        "device": device,
                        "group": name,
                        "kind": group_def["kind"],
                        "level": group_level,
                        "required": group_def.get("required", False),
                        "lane": lane,
                        "resources": {
                            "class": res_class,
                            "devices": devices,
                            "exclusive": exclusive,
                            "timeout_minutes": timeout,
                        },
                        "coverage": bool(
                            group_def.get("coverage", False)
                            and platform_cfg.get("coverage", {}).get("enabled", False)
                        ),
                        "exec": exec_entry,
                    }
                )

    return groups


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the plan."""
    source_root = Path(args.source_root).resolve()
    project_path = source_root / args.project
    configs_dir = source_root / args.configs_dir

    project = load_project(project_path)
    platforms = load_platforms(configs_dir)

    # Resolve stage
    stage_name = args.stage
    if stage_name not in project["stages"]:
        raise ValueError(f"Unknown stage: {stage_name}")
    stage_def = project["stages"][stage_name]

    # Get source SHA
    result = subprocess.run(
        ["git", "rev-parse", args.source_sha or "HEAD"],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=True,
    )
    source_sha = result.stdout.strip()

    # Get base SHA for PR
    base_sha = None
    if stage_name == "pr" and args.base_ref:
        result = subprocess.run(
            ["git", "merge-base", f"origin/{args.base_ref}", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            base_sha = result.stdout.strip()

    # Get changed files
    changed = []
    if args.changed_files:
        changed = json.loads(args.changed_files)
    elif stage_name == "pr" and args.base_ref:
        try:
            changed = gm._changed_files(args.base_ref)
        except subprocess.CalledProcessError:
            pass

    # Get labels
    labels = []
    if args.labels_from_event and stage_name == "pr":
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and Path(event_path).exists():
            try:
                event = json.loads(Path(event_path).read_text())
                labels = [l["name"] for l in event.get("pull_request", {}).get("labels", [])]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass

    # Parse requested platforms
    requested_platforms = []
    if args.platforms:
        requested_platforms = (
            json.loads(args.platforms) if args.platforms.startswith("[") else [args.platforms]
        )

    # Select platforms
    selected, lint_only = select_platforms(
        stage_name, platforms, changed, labels, requested_platforms
    )

    # Resolve level and risk
    test_level, risk_level = resolve_level(
        stage_def, args.requested_level, project, changed, labels
    )
    if args.requested_risk != "auto":
        if args.requested_risk not in RISK_ORDER or args.requested_risk == "R0":
            raise ValueError(f"Unknown requested risk: {args.requested_risk}")
        if RISK_ORDER[args.requested_risk] > RISK_ORDER[risk_level]:
            risk_level = args.requested_risk

    # Build platform entries
    platform_entries = []
    for pname in selected:
        p = platforms[pname]
        platform_entries.append(
            {
                "name": pname,
                "config": p["config_path"],
                "config_hash": p["config_hash"],
                "device_types": p["config"].get("device_types", []),
                "nproc_per_node": p["config"].get("nproc_per_node", 1),
                "coverage_enabled": bool(
                    p["config"].get("coverage", {}).get("enabled", False)
                ),
                "coverage_required": bool(
                    p["config"].get("coverage", {}).get("required", False)
                ),
                "reasons": ["selected"],
            }
        )

    # Build test groups
    blocked: list[dict[str, Any]] = []
    test_groups = (
        build_test_groups(project, platforms, selected, test_level, changed, blocked)
        if not lint_only
        else []
    )

    if stage_name == "pr":
        coverage_policy = {"method": "absolute"}
    elif stage_name == "nightly":
        coverage_policy = {
            "method": "statistical",
            "sigma": 3.0,
            "window_runs": 14,
            "minimum_history": 8,
        }
    else:
        coverage_policy = {"method": "none"}

    # Build plan
    plan = {
        "schema_version": contract.PLAN_SCHEMA,
        "repository": project["repository"],
        "stage": stage_name,
        "selection": stage_def["selection"],
        "lint_only": lint_only,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "test_level": test_level,
        "risk_level": risk_level,
        "risk_rule_version": project.get("risk_rules", {}).get("version", "v1"),
        "changed_files": {
            "count": len(changed),
            "documentation_only": (
                all(gm._is_documentation(f) for f in changed) if changed else False
            ),
        },
        "common_path_triggers": [],
        "platforms": platform_entries,
        "test_groups": test_groups,
        "excluded": [],
        "blocked": blocked,
        "reasons": [f"{stage_name} stage with {len(selected)} platforms, level {test_level}"],
        "coverage_policy": coverage_policy,
    }

    # Compute plan hash
    plan["plan_hash"] = contract.compute_plan_hash(plan)

    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pr", "branch", "nightly", "manual", "release"],
    )
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--changed-files", default="")
    parser.add_argument("--platforms", default="")
    parser.add_argument("--requested-level", default="auto")
    parser.add_argument("--requested-risk", default="auto")
    parser.add_argument("--labels-from-event", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--github-output", action="store_true")

    args = parser.parse_args()

    try:
        plan = build_plan(args)

        # Write plan file
        if args.output:
            args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

        # Write GitHub outputs
        if args.github_output:
            gh_out = os.environ.get("GITHUB_OUTPUT")
            if gh_out:
                with open(gh_out, "a") as f:
                    # Compact JSON for output
                    plan_json = contract.canonical_json(plan)
                    f.write(f"plan={plan_json}\n")
                    f.write(f"plan_hash={plan['plan_hash']}\n")
                    f.write(f"platforms={json.dumps([p['name'] for p in plan['platforms']])}\n")
                    f.write(f"source_sha={plan['source_sha']}\n")
                    f.write(f"test_level={plan['test_level']}\n")
                    f.write(f"risk_level={plan['risk_level']}\n")
                    f.write(f"stage={plan['stage']}\n")
                    f.write(f"lint_only={json.dumps(plan['lint_only'])}\n")
        else:
            # Default: write to stdout if no output options specified
            print(json.dumps(plan, indent=2, sort_keys=True))

        # Write step summary
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as f:
                f.write(f"### Plan: {plan['stage']} / {plan['test_level']}\n\n")
                f.write(
                    "- Platforms:"
                    f" {', '.join(p['name'] for p in plan['platforms']) or 'none (lint only)'}\n"
                )
                f.write(f"- Test groups: {len(plan['test_groups'])}\n")
                f.write(f"- Plan hash: `{plan['plan_hash'][:12]}...`\n\n")

        return 0

    except (OSError, ValueError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print(f"::error::Plan generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
