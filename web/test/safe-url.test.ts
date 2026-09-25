import assert from "node:assert/strict";
import test from "node:test";
import { safeExternalImageUrl } from "../src/safe-url";

test("external image URLs allow ordinary HTTP(S) references", () => {
  assert.equal(safeExternalImageUrl("https://images.example.test/cover.png?size=2"),
    "https://images.example.test/cover.png?size=2");
  assert.equal(safeExternalImageUrl("http://images.example.test/cover.png"),
    "http://images.example.test/cover.png");
  assert.equal(safeExternalImageUrl(" HTTPS://IMAGES.EXAMPLE.TEST/a.png "),
    "https://images.example.test/a.png");
});

test("external image URLs reject executable, local, ambiguous, and credentialed references", () => {
  for (const value of [
    "", "javascript:alert(1)", "data:image/svg+xml,<svg onload=alert(1)>",
    "blob:https://images.example.test/123", "file:///C:/private.png", "/relative.png",
    "//images.example.test/cover.png", "http:images.example.test/cover.png",
    "https://user:secret@images.example.test/cover.png",
    "https://images.example.test/\ncover.png", "https://[invalid",
  ]) {
    assert.equal(safeExternalImageUrl(value), null, value);
  }
});
