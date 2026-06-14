/* tslint:disable */
/* eslint-disable */

/**
 * Analyze a position: return the engine's best move and the 0..100 win split.
 * `state` is the TS `GameState` object; `depth` is the search depth (plies).
 * `k` controls the logistic squash (larger = scores look closer to 50/50).
 */
export function analyze_state(state: any, depth: number, k: number): any;

/**
 * Analyze with a node budget. Returns the deepest completed depth plus budget
 * metadata so the browser can use a high max depth with stable tail latency.
 */
export function analyze_state_budgeted(state: any, depth: number, node_limit: bigint, k: number): any;

/**
 * Convenience: just the move (TS `Move` JSON), matching the old `chooseMove`.
 */
export function choose_move(state: any, depth: number): any;

/**
 * Generate the whole pruned state graph from `state` in one call. BFS happens
 * entirely in Rust (one shared transposition table), so the UI makes a single
 * call instead of thousands of round-trips. `max_depth` controls how far ahead
 * we precompute; `max_nodes` is the hard memory ceiling (sets `capped` if hit).
 */
export function generate_graph_js(state: any, rank_depth: number, k: number, margin: number, win_k: number, max_depth: number, max_nodes: number): any;

/**
 * Pruned successors for the state graph: top-`k` strong moves (depth `depth`),
 * useless walls and blunders (worse than best by > `margin`) removed. Each
 * entry carries the move + the resulting SOUTH win-chance, already scored —
 * the graph needs no separate analysis pass.
 */
export function top_moves_js(state: any, depth: number, k: number, margin: number, win_k: number): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly analyze_state: (a: any, b: number, c: number) => [number, number, number];
    readonly analyze_state_budgeted: (a: any, b: number, c: bigint, d: number) => [number, number, number];
    readonly choose_move: (a: any, b: number) => [number, number, number];
    readonly generate_graph_js: (a: any, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly top_moves_js: (a: any, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
