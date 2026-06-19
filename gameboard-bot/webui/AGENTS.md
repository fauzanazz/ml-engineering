# gameboard-bot WebUI

TanStack Start app with Cloudflare Workers deployment.

## Scaffold Commands

```
# initial scaffold
bun x @tanstack/cli@latest create my-tanstack-app --agent --deployment cloudflare

# intent setup
bun x @tanstack/intent@latest install
bun x @tanstack/intent@latest list
```

## Stack

- **Framework**: TanStack Start (React 19, SSR)
- **Router**: TanStack Router (file-based routes in `src/routes/`)
- **Styling**: Tailwind CSS v4
- **Build**: Vite 8 + `@cloudflare/vite-plugin`
- **Deploy**: Cloudflare Workers via Wrangler
- **Package manager**: bun

## Project Structure

```
src/
  routes/
    __root.tsx      # root layout (HeadContent, Scripts, Outlet)
    index.tsx       # home route
    about.tsx       # about route
  components/
    Header.tsx
    Footer.tsx
    ThemeToggle.tsx
  router.tsx        # createRouter entry
  styles.css        # Tailwind base styles
public/             # static assets
wrangler.jsonc      # Cloudflare Workers config
vite.config.ts      # Vite + cloudflare plugin + tanstackStart plugin
```

## Scripts

```
bun run dev       # local dev server on :3000
bun run build     # production build
bun run preview   # preview built output
bun run deploy    # build + wrangler deploy
bun run test      # vitest
```

## Environment Variables

- No required env vars for base scaffold.
- Add `VITE_`-prefixed vars for client-side access (see start-core/execution-model skill).
- Server-only secrets: use `process.env` without `VITE_` prefix in server functions.

## Deployment Notes

- Target: Cloudflare Workers (not Pages).
- Entry: `@tanstack/react-start/server-entry` (set in `wrangler.jsonc`).
- Compatibility: `nodejs_compat` flag required for Node.js APIs on Workers.
- Deploy: `bun run deploy` runs build then `wrangler deploy`.
- Cloudflare account + `wrangler login` needed before first deploy.

## Architectural Decisions

- Minimal scaffold — no auth, no DB, no extra integrations.
- File-based routing via TanStack Router plugin (auto-generates `src/routeTree.gen.ts` on build).
- SSR enabled by default via `tanstackStart()` Vite plugin.
- `#/*` import alias maps to `./src/*` (configured in `package.json#imports` + tsconfig).

## Known Gotchas

- `bun x @tanstack/cli@latest` must run outside an empty directory without a `package.json` (bun handles it; npm fails with ENOENT).
- `npm npx` misparses `--agent` and `--deployment` as npm config flags — use `bun x` instead.
- `routeTree.gen.ts` is auto-generated; do not edit manually.
- Wrangler `name` set to `wallchess-webui`.

## Next Steps

- [x] Renamed to `wallchess-webui` in `wrangler.jsonc` and `package.json`.
- [ ] Add chess board UI routes under `src/routes/`.
- [ ] Configure Cloudflare Workers KV or D1 if persistence needed.

---

<!-- intent-skills:start -->
## Skill Loading

Before substantial work:
- Skill check: run `bunx @tanstack/intent@latest list`, or use skills already listed in context.
- Skill guidance: if one local skill clearly matches the task, run `bunx @tanstack/intent@latest load <package>#<skill>` and follow the returned `SKILL.md`.
- Monorepos: when working across packages, run the skill check from the workspace root and prefer the local skill for the package being changed.
- Multiple matches: prefer the most specific local skill for the package or concern you are changing; load additional skills only when the task spans multiple packages or concerns.
<!-- intent-skills:end -->
