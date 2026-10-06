Anime graph companion — Windows x64 candidate package

This is a portable engineering candidate, not an approved desktop release.
It contains no anime dataset or viewing history and sends no local data to a service.
The web app remains the recommendation client; this app inspects a local graph.

Requirements
- Windows x64 with a desktop session.
- Microsoft Edge WebView2 Evergreen Runtime (installed on many Windows systems):
  https://developer.microsoft.com/microsoft-edge/webview2/#download-section
- Microsoft Visual C++ Redistributable for x64, supplying VCRUNTIME140.dll and VCRUNTIME140_1.dll:
  https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist

Extract the complete anime-graph-desktop folder. Run Launch.cmd to check the
package EXE hash and runtime prerequisites before opening the app. The checker
does not download or install anything. It displays the official links if a
prerequisite is missing. You can also run anime_graph_desktop.exe directly on
a machine where the prerequisites are already installed.

The app starts with No data. Open invented demo only for synthetic examples.
To inspect a local graph, choose Select local manifest and pick a compatible
release-manifest-v1 file beside its hash-bound graph.compact.json. The app
checks only that graph and manifest, not the rest of a bundle, source rights,
or publication approval. Invalid selections do not switch to the demo.

The overview is limited to 300 titles and 1,400 signed pairs. Pair preference
weights and co-rater support are not similarity scores or recommendations.
No relative data directory is required. Keep private ratings and histories
out of any shared package or diagnostic report.
