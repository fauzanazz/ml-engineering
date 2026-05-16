#!/usr/bin/env node
const { spawnSync } = require("node:child_process");

const args = process.argv.slice(2);
const command = ["--from", "ml-production-ecosystem", "ml-struct", "new", ...args];
const result = spawnSync("uvx", command, { stdio: "inherit" });

if (result.error) {
  console.error("create-ml-struct requires uvx on PATH.");
  console.error("Install uv: https://docs.astral.sh/uv/");
  process.exit(1);
}

process.exit(result.status ?? 1);
