# Implementation Phases

## Phase 1: Foundation
- Added local manager authentication with `admin` and `user` roles.
- Added a typed WebSocket event envelope for metrics, server jobs, model downloads, and LiteLLM changes.
- Restricted mutating server and management APIs to admin users.

## Phase 2: Model Jobs
- Added an admin model catalog on top of the built-in defaults.
- Split model download from server start into asynchronous jobs.
- Added download status persistence and WebSocket progress events.

## Phase 3: LiteLLM Management
- Added backend wrappers for LiteLLM users, teams, virtual keys, budgets, and spend logs.
- Added a Postgres service for LiteLLM persistence in Docker Compose.
- Added a manager UI for local users and LiteLLM key generation.

## Phase 4: Operations
- Routed LiteLLM and metrics to the backend-managed vLLM process to avoid two active vLLM targets by default.
- Kept the standalone vLLM Compose service behind the `standalone-vllm` profile for manual use.
- Added HF token propagation and admin credentials to environment examples.
