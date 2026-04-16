"""
Synapse — Associative memory with linked concept traversal.

Mimics how human memory works: thinking of "coffee" activates
"morning", "energy", "routine" through associative links.
"""

import json
from typing import Optional


class Synapse:
    """Associative memory graph with bidirectional links.

    Example:
        >>> brain = Synapse()
        >>> brain.link("trading", ["binance", "strategy", "risk"])
        >>> brain.link("binance", ["api", "demo", "watchlist"])
        >>> brain.activate("trading")
        ['binance', 'strategy', 'risk']
        >>> brain.activate("trading", depth=2)
        ['binance', 'api', 'demo', 'watchlist', 'strategy', 'risk']
    """

    def __init__(self):
        self._links: dict[str, list[str]] = {}
        self._weights: dict[str, dict[str, float]] = {}  # concept → {assoc: weight}
        self._frequencies: dict[str, dict[str, int]] = {}  # concept → {assoc: use_count}
        self._metadata: dict[str, dict] = {}

    def link(self, concept: str, associations: list[str], bidirectional: bool = True,
             weights: dict[str, float] = None) -> None:
        """Link a concept to its associations with optional weights.

        Args:
            concept: The source concept.
            associations: List of related concepts.
            bidirectional: If True, each association also links back to concept.
            weights: Optional dict of {association: weight} (0.0-1.0).
                     Higher weight = stronger association = recalled first.
                     Default weight is 0.5 for all unspecified associations.
        """
        if concept not in self._links:
            self._links[concept] = []
        if concept not in self._weights:
            self._weights[concept] = {}
        if concept not in self._frequencies:
            self._frequencies[concept] = {}

        for assoc in associations:
            if assoc not in self._links[concept]:
                self._links[concept].append(assoc)

            # Set weight (explicit > default 0.5)
            w = (weights or {}).get(assoc, 0.5)
            self._weights[concept][assoc] = max(0.0, min(1.0, w))

            # Init frequency
            if assoc not in self._frequencies[concept]:
                self._frequencies[concept][assoc] = 0

            if bidirectional:
                if assoc not in self._links:
                    self._links[assoc] = []
                if assoc not in self._weights:
                    self._weights[assoc] = {}
                if assoc not in self._frequencies:
                    self._frequencies[assoc] = {}
                if concept not in self._links[assoc]:
                    self._links[assoc].append(concept)
                # Bidirectional weight defaults to 0.5 (unless explicitly set)
                if concept not in self._weights[assoc]:
                    self._weights[assoc][concept] = 0.5
                if concept not in self._frequencies[assoc]:
                    self._frequencies[assoc][concept] = 0

    def activate(self, concept: str, depth: int = 1, weighted: bool = True) -> list:
        """Activate a concept and traverse associations up to depth levels.

        Like thinking of a word and having related concepts come to mind.
        Uses BFS for correct level-order traversal without duplicates.
        Results are sorted by cumulative weight along the strongest path
        (product of link weights from root → node).

        Args:
            concept: The concept to activate.
            depth: How many levels of association to traverse (1 = direct only).
            weighted: If True, sort results by weight (descending).

        Returns:
            Ordered list of activated concepts (strongest first).
        """
        if depth < 1:
            return []

        seen = {concept}
        result = []
        cumulative: dict[str, float] = {}  # concept → best cumulative weight
        frontier = [(concept, 1.0)]  # (node, cumulative_weight_from_root)

        for _ in range(depth):
            next_frontier = []
            for node, node_weight in frontier:
                for assoc in self._links.get(node, []):
                    if assoc not in seen:
                        seen.add(assoc)
                        link_weight = self.get_weight(node, assoc)
                        cum_weight = node_weight * link_weight
                        cumulative[assoc] = cum_weight
                        result.append(assoc)
                        next_frontier.append((assoc, cum_weight))
                        # Increment frequency
                        if node in self._frequencies and assoc in self._frequencies[node]:
                            self._frequencies[node][assoc] += 1
                    else:
                        # Node already seen via different path — keep stronger weight
                        link_weight = self.get_weight(node, assoc)
                        cum_weight = node_weight * link_weight
                        if assoc in cumulative and cum_weight > cumulative[assoc]:
                            cumulative[assoc] = cum_weight
            frontier = next_frontier

        if weighted and result:
            result.sort(key=lambda c: cumulative.get(c, 0.5), reverse=True)

        return result

    def get_weight(self, concept: str, assoc: str) -> float:
        """Get the weight of an association. Returns 0.5 if not set."""
        return self._weights.get(concept, {}).get(assoc, 0.5)

    def set_weight(self, concept: str, assoc: str, weight: float) -> None:
        """Manually set the weight of an association (0.0-1.0)."""
        if concept not in self._weights:
            self._weights[concept] = {}
        self._weights[concept][assoc] = max(0.0, min(1.0, weight))

    def strengthen(self, concept: str, assoc: str, boost: float = 0.1) -> float:
        """Strengthen an association by boosting its weight.

        Mimics how repeated exposure strengthens neural connections.
        Returns the new weight.

        Args:
            concept: The source concept.
            assoc: The association to strengthen.
            boost: How much to increase weight (default 0.1).
        """
        current = self.get_weight(concept, assoc)
        new_weight = min(1.0, current + boost)
        self.set_weight(concept, assoc, new_weight)
        return new_weight

    def weaken(self, concept: str, assoc: str, decay: float = 0.05) -> float:
        """Weaken an association by reducing its weight.

        Mimics how unused connections decay over time.
        Returns the new weight.
        """
        current = self.get_weight(concept, assoc)
        new_weight = max(0.0, current - decay)
        self.set_weight(concept, assoc, new_weight)
        return new_weight

    def top_associations(self, concept: str, limit: int = 5) -> list:
        """Get top associations by weight, like "what comes to mind first."

        Returns:
            List of (association, weight) tuples, sorted by weight descending.
        """
        assocs = self._weights.get(concept, {})
        ranked = sorted(assocs.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def get_frequency(self, concept: str, assoc: str) -> int:
        """Get how many times an association has been activated."""
        return self._frequencies.get(concept, {}).get(assoc, 0)

    def connections(self, concept: str) -> dict:
        """Get structured connection info for a concept.

        Returns:
            Dict with 'direct' (list), 'count' (int), and 'strength' info.
        """
        direct = self._links.get(concept, [])
        return {
            "concept": concept,
            "direct": direct,
            "count": len(direct),
            "is_hub": len(direct) >= 5,  # 5+ connections = hub node
        }

    def find_path(self, start: str, end: str, max_depth: int = 5) -> Optional[list]:
        """Find shortest path between two concepts (BFS).

        Returns:
            List of concepts from start to end, or None if no path exists.
        """
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            for neighbor in self._links.get(current, []):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def hubs(self, min_connections: int = 3) -> list:
        """Find hub concepts (most connected nodes).

        Returns:
            List of (concept, connection_count) sorted by count descending.
        """
        hubs = [
            (concept, len(links))
            for concept, links in self._links.items()
            if len(links) >= min_connections
        ]
        return sorted(hubs, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict:
        """Export the full graph as a dict."""
        return {
            "links": self._links,
            "weights": self._weights,
            "frequencies": self._frequencies,
            "metadata": self._metadata,
        }

    def export(self) -> str:
        """Export as minified JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> "Synapse":
        """Create from a dict."""
        s = cls()
        s._links = data.get("links", {})
        s._weights = data.get("weights", {})
        s._frequencies = data.get("frequencies", {})
        s._metadata = data.get("metadata", {})
        return s

    @classmethod
    def from_json(cls, json_str: str) -> "Synapse":
        """Create from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        total_links = sum(len(v) for v in self._links.values())
        return f"Synapse(concepts={len(self._links)}, links={total_links})"
