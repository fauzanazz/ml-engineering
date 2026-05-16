# Design: Local RT Transport + Warehouse Foundation

## Spec Reference

`.planning/spec.md`

## Approved Approach

Use Docker-backed local production-like infrastructure:

- Redpanda as Kafka-compatible real-time transport.
- PostgreSQL as local warehouse stand-in.
- Python producer/consumer/demo workflow.
- Recommendation request events.
- One command: `foundation-rt-demo`.
- Default tests expect Docker services available.

## Tradeoffs

Pros:

- Close to production concepts without managed cloud services.
- Strong demo/tutorial value.
- Clear migration path to real Kafka and managed warehouse adapters.

Cons:

- Requires Docker.
- Tests can fail when Docker is unavailable or ports conflict.
- Slower feedback loop than pure unit tests.

## Implementation Plan

1. Add local infra.
   - Create `docker-compose.yml`.
   - Add `redpanda` and `postgres` services.
   - Expose local ports and healthchecks where practical.

2. Add runtime dependencies.
   - Kafka client dependency.
   - PostgreSQL dependency.

3. Add RT foundation package.
   - Create `src/ml_production_ecosystem/recommendation/rt_transport.py`.
   - Define topic, producer, consumer, serialization, deserialization.
   - Keep adapter replaceable.

4. Add warehouse package.
   - Create `src/ml_production_ecosystem/recommendation/warehouse.py`.
   - Create schema/table.
   - Insert processed recommendation request results.
   - Query/read back rows.

5. Add demo workflow.
   - Create `src/ml_production_ecosystem/recommendation/rt_demo.py`.
   - Add CLI script `foundation-rt-demo`.
   - Flow: ensure table, produce sample events, consume events, store rows, read rows, print summary.

6. Add tests.
   - Transport produce/consume integration test.
   - Warehouse create/insert/read integration test.
   - End-to-end demo workflow test.
   - Default `uv run pytest` expects Docker services running.

7. Add docs.
   - Update `docs/domains/foundation/README.md` with Docker/demo/test commands.
   - Explain Redpanda/Kafka and PostgreSQL/warehouse mapping.

8. Verify.
   - `docker compose up -d`
   - `uv run pytest`
   - `uv run foundation-rt-demo`
   - `docker compose down`
