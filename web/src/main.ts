import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import "./style.css";

interface GraphNode {
  id: string;
  label: string;
  nodeType: "user" | "anime";
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  edgeType: "user-anime" | "anime-anime";
  weight: number;
}

interface GraphData {
  generatedAt: string;
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app container");
}

app.innerHTML = `
  <main class="layout">
    <section class="panel">
      <h1>What Anime Should I Watch</h1>
      <p class="subhead">
        Graph built from anonymized user ratings. User nodes connect to rated anime using normalized scores.
      </p>

      <div class="stats" id="stats"></div>

      <label class="control">
        <span>Min absolute edge weight</span>
        <input id="min-weight" type="range" min="0" max="4" value="0" step="0.05" />
        <output id="min-weight-value">0.00</output>
      </label>

      <label class="checkbox">
        <input id="toggle-anime-edges" type="checkbox" checked />
        <span>Show anime-to-anime edges</span>
      </label>
    </section>

    <section class="graph-shell">
      <div id="graph"></div>
    </section>
  </main>
`;

const statsEl = mustElement<HTMLDivElement>("#stats");
const graphContainer = mustElement<HTMLDivElement>("#graph");
const minWeightInput = mustElement<HTMLInputElement>("#min-weight");
const minWeightValue = mustElement<HTMLOutputElement>("#min-weight-value");
const toggleAnimeEdges = mustElement<HTMLInputElement>("#toggle-anime-edges");

let renderer: Sigma | null = null;

const graphData = await fetchGraph();
renderGraph(graphData, 0, true);

minWeightInput.addEventListener("input", () => {
  const value = Number.parseFloat(minWeightInput.value);
  minWeightValue.textContent = value.toFixed(2);
  renderGraph(graphData, value, toggleAnimeEdges.checked);
});

toggleAnimeEdges.addEventListener("change", () => {
  const value = Number.parseFloat(minWeightInput.value);
  renderGraph(graphData, value, toggleAnimeEdges.checked);
});

function renderGraph(
  graphDataValue: GraphData,
  minAbsoluteWeight: number,
  showAnimeAnimeEdges: boolean,
): void {
  const graph = new Graph({ multi: true, type: "undirected" });

  for (const node of graphDataValue.nodes) {
    const isUser = node.nodeType === "user";
    graph.addNode(node.id, {
      label: node.label,
      nodeType: node.nodeType,
      size: isUser ? 5.5 : 3,
      color: isUser ? "#ff8a00" : "#0f8b8d",
      x: Math.random(),
      y: Math.random(),
    });
  }

  for (const edge of graphDataValue.edges) {
    const edgePassesTypeFilter =
      showAnimeAnimeEdges || edge.edgeType !== "anime-anime";
    const edgePassesWeightFilter =
      Math.abs(edge.weight) >= minAbsoluteWeight || edge.edgeType === "user-anime";

    if (!edgePassesTypeFilter || !edgePassesWeightFilter) {
      continue;
    }

    graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
      size: edge.edgeType === "user-anime" ? 1.6 : 0.7,
      color: edge.edgeType === "user-anime" ? "#f4d35e88" : "#6fffe988",
      weight: edge.weight,
      edgeType: edge.edgeType,
    });
  }

  forceAtlas2.assign(graph, {
    iterations: 300,
    settings: forceAtlas2.inferSettings(graph),
  });

  if (renderer) {
    renderer.kill();
  }

  renderer = new Sigma(graph, graphContainer, {
    renderEdgeLabels: false,
    labelRenderedSizeThreshold: 14,
    allowInvalidContainer: false,
  });

  statsEl.innerHTML = [
    statLine("Generated", new Date(graphDataValue.generatedAt).toLocaleString()),
    statLine("Users", String(graphDataValue.userCount)),
    statLine("Anime", String(graphDataValue.animeCount)),
    statLine("Visible edges", String(graph.size)),
  ].join("");
}

function statLine(label: string, value: string): string {
  return `<div class="stat-row"><span>${label}</span><strong>${value}</strong></div>`;
}

async function fetchGraph(): Promise<GraphData> {
  const response = await fetch("./data/graph.json");
  if (!response.ok) {
    throw new Error(`Unable to load graph.json (${response.status})`);
  }
  return (await response.json()) as GraphData;
}

function mustElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing element ${selector}`);
  }
  return element;
}
