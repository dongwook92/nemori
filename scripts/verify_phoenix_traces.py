#!/usr/bin/env python3
"""Verify Nemori LLM traces through the Phoenix REST API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from ipaddress import ip_address
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


def _is_local_hostname(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or "." not in normalized:
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-time", required=True, help="ISO-8601 lower bound")
    parser.add_argument(
        "--base-url",
        default=os.getenv("PHOENIX_BASE_URL", "http://localhost:6006"),
    )
    parser.add_argument(
        "--project",
        default=os.getenv("PHOENIX_PROJECT_NAME", "nemori"),
    )
    parser.add_argument("--require-phase", action="append", default=[])
    parser.add_argument("--agent-id")
    parser.add_argument("--user-id")
    return parser.parse_args()


def _fetch_spans(args: argparse.Namespace) -> list[dict[str, Any]]:
    parsed_base_url = urlparse(args.base_url)
    if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.hostname:
        raise ValueError("Phoenix base URL must be an HTTP(S) URL")
    if parsed_base_url.username is not None or parsed_base_url.password is not None:
        raise ValueError("Phoenix base URL must not contain credentials")
    if parsed_base_url.scheme == "http" and not _is_local_hostname(
        parsed_base_url.hostname
    ):
        raise ValueError("Phoenix base URL must use HTTPS for non-local hosts")

    query = urlencode(
        {
            "start_time": args.start_time,
            "order": "desc",
            "limit": 1000,
        }
    )
    url = (
        f"{args.base_url.rstrip('/')}/v1/projects/{quote(args.project, safe='')}/spans"
        f"?{query}"
    )
    headers = {}
    if api_key := os.getenv("PHOENIX_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    # The URL scheme is restricted above; urlopen cannot access file/custom schemes.
    with urlopen(Request(url, headers=headers), timeout=10) as response:  # nosec B310
        payload = json.load(response)
    spans = payload.get("data", [])
    return spans if isinstance(spans, list) else []


def main() -> int:
    args = _parse_args()
    spans = _fetch_spans(args)
    llm_spans = [span for span in spans if span.get("span_kind") == "LLM"]
    phase_counts = Counter(
        span.get("attributes", {}).get("nemori.llm.phase")
        for span in spans
        if span.get("attributes", {}).get("nemori.llm.phase")
    )

    failures = []
    if not llm_spans:
        failures.append("no LLM spans found")
    missing_phases = sorted(set(args.require_phase) - set(phase_counts))
    if missing_phases:
        failures.append(f"missing phases: {', '.join(missing_phases)}")

    tenant_matches = llm_spans
    if args.agent_id is not None:
        tenant_matches = [
            span
            for span in tenant_matches
            if span.get("attributes", {}).get("metadata.agent_id") == args.agent_id
        ]
    if args.user_id is not None:
        tenant_matches = [
            span
            for span in tenant_matches
            if span.get("attributes", {}).get("metadata.user_id") == args.user_id
        ]
    if (args.agent_id is not None or args.user_id is not None) and not tenant_matches:
        failures.append("no LLM spans matched the requested tenant")

    spans_by_id = {
        span.get("context", {}).get("span_id"): span
        for span in spans
        if span.get("context", {}).get("span_id")
    }
    parent_names = sorted(
        {
            parent.get("name", "")
            for span in llm_spans
            if (parent := spans_by_id.get(span.get("parent_id"))) is not None
        }
    )
    if llm_spans and not any(name.startswith("nemori.llm.") for name in parent_names):
        failures.append("LLM spans are not linked to Nemori phase spans")

    summary = {
        "span_count": len(spans),
        "trace_count": len(
            {
                span.get("context", {}).get("trace_id")
                for span in spans
                if span.get("context", {}).get("trace_id")
            }
        ),
        "llm_span_count": len(llm_spans),
        "phase_counts": dict(sorted(phase_counts.items())),
        "llm_parent_names": parent_names,
        "tenant_match_count": len(tenant_matches),
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
