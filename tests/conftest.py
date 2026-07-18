"""Shared test fixtures and configuration."""

# This module is an executable live-API smoke test. Ignoring it prevents pytest
# collection from importing it and loading local .env credentials.
collect_ignore = ["test_integration_openrouter.py"]
