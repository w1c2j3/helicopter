import { expect, test } from "@playwright/test";
import { existsSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

import { CORE_BENCHMARK_CATALOG } from "../lib/benchmarkCatalog";

const CLIENT_ROOT = process.cwd();
const REPO_SRC = path.resolve(CLIENT_ROOT, "..");
const BACKEND_API_ROUTES = path.join(REPO_SRC, "scoreboard-server", "scoreboard_server", "routes", "api");
const CLIENT_API = path.join(CLIENT_ROOT, "lib", "api");
const CLIENT_DTOS_API = path.join(CLIENT_ROOT, "lib", "dtos", "api");

function leafFiles(root: string, extension: string, ignoreRootIndex = false): string[] {
  const leaves: string[] = [];
  const walk = (directory: string) => {
    for (const name of readdirSync(directory)) {
      const full = path.join(directory, name);
      if (statSync(full).isDirectory()) {
        walk(full);
        continue;
      }
      if (!name.endsWith(extension)) continue;
      const relative = path.relative(root, full).split(path.sep).join("/");
      if (ignoreRootIndex && relative === "index.ts") continue;
      leaves.push(relative.slice(0, -extension.length));
    }
  };
  walk(root);
  return leaves.sort();
}

test("client API and DTO trees mirror backend API routes", () => {
  const backendLeaves = leafFiles(BACKEND_API_ROUTES, ".py")
    .filter((leaf) => !leaf.endsWith("/__init__") && leaf !== "__init__")
    .map((leaf) => leaf.replaceAll("__init__", "index"))
    .sort();
  const apiLeaves = leafFiles(CLIENT_API, ".ts", true);
  const dtoLeaves = leafFiles(CLIENT_DTOS_API, ".ts");

  expect(existsSync(path.join(CLIENT_ROOT, "lib", "api.ts"))).toBe(false);
  expect(existsSync(path.join(CLIENT_ROOT, "lib", "types.ts"))).toBe(false);
  expect(apiLeaves).toEqual(backendLeaves);
  expect(dtoLeaves).toEqual(backendLeaves);
});

test("core research matrix keeps the balanced 20-benchmark scope", () => {
  const domainCounts = CORE_BENCHMARK_CATALOG.reduce<Record<string, number>>(
    (counts, benchmark) => ({
      ...counts,
      [benchmark.domain]: (counts[benchmark.domain] ?? 0) + 1,
    }),
    {},
  );

  expect(CORE_BENCHMARK_CATALOG).toHaveLength(20);
  expect(domainCounts.knowledge).toBe(8);
  expect(domainCounts.math).toBe(6);
  expect(domainCounts.coding).toBe(4);
  expect(domainCounts.instruction_following).toBe(2);
  expect(new Set(CORE_BENCHMARK_CATALOG.map((benchmark) => benchmark.key)).size).toBe(20);
});
