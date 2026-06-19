import { expect, test } from "@playwright/test";
import path from "node:path";

const fixtureLibrary = {
  selected_id: "kicau-mania",
  effects: {
    "kicau-mania": {
      name: "Kicau Mania",
      right_sticker: "assets/nick.gif",
      left_sticker: "assets/cat.gif",
      audio: "assets/Kicau Mania Cutted.mp3",
      audio_tracks: [{ id: "song", name: "Song", path: "assets/Kicau Mania Cutted.mp3", volume: 0.7, loop: true, muted: false }],
      selected_audio: "assets/Kicau Mania Cutted.mp3",
      audio_volume: 0.7,
      audio_loop: true,
      scale: 0.25,
      right_x: 0.72,
      right_y: 0.12,
      left_x: 0.04,
      left_y: 0.12,
      layers: [
        { id: "right", name: "Gif 1", asset_path: "assets/nick.gif", x: 0.72, y: 0.12, scale: 0.25, rotation: 0, opacity: 1, hidden: false, chroma_key_green: false, chroma_tolerance: 80, enter_animation: "pop", loop_animation: "bob", animation_speed: 1 },
        { id: "left", name: "Gif 2", asset_path: "assets/cat.gif", x: 0.04, y: 0.12, scale: 0.25, rotation: 0, opacity: 1, hidden: false, chroma_key_green: true, chroma_tolerance: 80, enter_animation: "slide", loop_animation: "pulse", animation_speed: 1 },
      ],
      trigger_labels: ["kicau"],
      activate_threshold: 0.7,
      deactivate_threshold: 0.45,
    },
  },
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      value: {
        getUserMedia: async () => {
          const canvas = document.createElement("canvas");
          canvas.width = 640;
          canvas.height = 360;
          const context = canvas.getContext("2d")!;
          context.fillStyle = "#384233";
          context.fillRect(0, 0, canvas.width, canvas.height);
          context.fillStyle = "#f4f0e6";
          context.fillRect(80, 80, 180, 180);
          return canvas.captureStream(12);
        },
      },
      configurable: true,
    });
  });
  await page.route("**/api/effects", async (route) => route.fulfill({ json: fixtureLibrary }));
  await page.route("**/api/assets", async (route) => route.fulfill({ json: { assets: [{ path: "assets/user/custom.gif", name: "custom.gif", type: "image", tag: "user" }] } }));
  await page.route("**/api/runtime/status", async (route) => route.fulfill({ json: { connected: false, label: "manual", confidence: 0, fps: 0, active_effect: null } }));
  await page.route("**/api/effects/kicau-mania/versions", async (route) => route.fulfill({ json: { versions: [] } }));
  await page.route("**/api/select-effect/kicau-mania", async (route) => route.fulfill({ json: fixtureLibrary }));
  await page.route("**/assets/nick.gif", async (route) => route.fulfill({ path: path.resolve("../assets/nick.gif") }));
  await page.route("**/assets/cat.gif", async (route) => route.fulfill({ path: path.resolve("../assets/cat.gif") }));
  await page.route("**/assets/example.png", async (route) => route.fulfill({ path: path.resolve("../assets/example.png") }));
});

test("editor renders controls, camera simulator, and responsive canvas", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Effects" })).toBeVisible();
  await expect(page.getByRole("button", { name: "layers" })).toBeVisible();
  await expect(page.locator(".safe-frame")).toBeVisible();
  await expect(page.locator(".properties")).toBeVisible();
  await expect(page.locator(".asset-rail")).toBeVisible();
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Start camera" }).click();
  await expect(page.getByRole("button", { name: "Stop camera" })).toBeVisible();
  await page.getByRole("button", { name: "Activate" }).click();
  await expect(page.locator(".safe-frame[data-active-preview='true']")).toBeVisible();

  const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
  expect(horizontalOverflow).toBe(false);

  await page.screenshot({ path: testInfo.outputPath(`editor-${testInfo.project.name}.png`), fullPage: true });
});
