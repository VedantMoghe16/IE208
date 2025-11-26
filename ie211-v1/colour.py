"""
Competitive Graph Coloring Game System
Complete implementation with chromatic number algorithms and optimal game strategies
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy
from collections import defaultdict
import time


class ChromaticNumberSolver:
    """
    Multiple algorithms to find or approximate the chromatic number of a graph
    """
    
    def __init__(self, graph: nx.Graph):
        self.G = graph
        self.n = len(graph.nodes())
        
    def exact_chromatic_number(self, max_colors: Optional[int] = None) -> int:
        """
        Find exact chromatic number using backtracking
        Time complexity: O(k^n) where k is chromatic number
        """
        if max_colors is None:
            max_colors = self.n
            
        for k in range(1, max_colors + 1):
            coloring = {}
            if self._can_color_with_k(k, coloring, list(self.G.nodes())):
                return k
        return self.n
    
    def _can_color_with_k(self, k: int, coloring: Dict, remaining: List) -> bool:
        """Recursive backtracking to check if graph can be colored with k colors"""
        if not remaining:
            return True
            
        node = remaining[0]
        neighbor_colors = {coloring.get(neighbor) for neighbor in self.G.neighbors(node)}
        
        for color in range(k):
            if color not in neighbor_colors:
                coloring[node] = color
                if self._can_color_with_k(k, coloring, remaining[1:]):
                    return True
                del coloring[node]
        
        return False
    
    def greedy_chromatic_upper_bound(self) -> Tuple[int, Dict]:
        """
        Greedy coloring algorithm - provides upper bound
        Time complexity: O(n + m)
        Returns: (number of colors used, coloring dict)
        """
        coloring = {}
        
        # Order nodes by degree (Welsh-Powell algorithm)
        nodes_by_degree = sorted(self.G.nodes(), 
                                key=lambda x: self.G.degree(x), 
                                reverse=True)
        
        for node in nodes_by_degree:
            neighbor_colors = {coloring.get(neighbor) for neighbor in self.G.neighbors(node)}
            
            # Find smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            coloring[node] = color
        
        return max(coloring.values()) + 1, coloring
    
    def clique_lower_bound(self) -> int:
        """
        Find size of maximum clique as lower bound on chromatic number
        Uses greedy approach (not optimal but fast)
        """
        max_clique_size = 1
        
        for node in self.G.nodes():
            # Find clique containing this node
            clique = {node}
            candidates = set(self.G.neighbors(node))
            
            while candidates:
                # Pick node with most connections to current clique
                next_node = max(candidates, 
                              key=lambda x: len(set(self.G.neighbors(x)) & clique))
                
                # Check if it forms a clique
                if all(self.G.has_edge(next_node, c) for c in clique):
                    clique.add(next_node)
                    candidates = candidates & set(self.G.neighbors(next_node))
                else:
                    candidates.remove(next_node)
            
            max_clique_size = max(max_clique_size, len(clique))
        
        return max_clique_size
    
    def dsatur_coloring(self) -> Tuple[int, Dict]:
        """
        DSatur (Degree of Saturation) algorithm - often finds optimal or near-optimal coloring
        Better than greedy in practice
        """
        coloring = {}
        uncolored = set(self.G.nodes())
        
        while uncolored:
            # Choose node with highest saturation degree (most colored neighbors)
            # Break ties by degree
            node = max(uncolored, key=lambda x: (
                len({coloring.get(n) for n in self.G.neighbors(x) if n in coloring}),
                self.G.degree(x)
            ))
            
            # Find smallest available color
            neighbor_colors = {coloring.get(n) for n in self.G.neighbors(node) if n in coloring}
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[node] = color
            uncolored.remove(node)
        
        return max(coloring.values()) + 1, coloring


class GameState:
    """Represents the state of a graph coloring game"""
    
    def __init__(self, graph: nx.Graph, num_colors: int, coloring: Optional[Dict] = None):
        self.G = graph
        self.num_colors = num_colors
        self.coloring = coloring if coloring else {}
        self.current_player = 0
        
    def clone(self):
        """Create a deep copy of the game state"""
        new_state = GameState(self.G, self.num_colors, self.coloring.copy())
        new_state.current_player = self.current_player
        return new_state
    
    def get_legal_moves(self, node: int) -> List[int]:
        """Get all legal colors for a given node"""
        if node in self.coloring:
            return []
        
        neighbor_colors = {self.coloring.get(n) for n in self.G.neighbors(node) 
                          if n in self.coloring}
        return [c for c in range(self.num_colors) if c not in neighbor_colors]
    
    def get_all_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal (node, color) pairs"""
        moves = []
        for node in self.G.nodes():
            if node not in self.coloring:
                for color in self.get_legal_moves(node):
                    moves.append((node, color))
        return moves
    
    def make_move(self, node: int, color: int):
        """Apply a move to the game state"""
        if node in self.coloring:
            raise ValueError(f"Node {node} already colored")
        if color not in self.get_legal_moves(node):
            raise ValueError(f"Color {color} not legal for node {node}")
        
        self.coloring[node] = color
        self.current_player = 1 - self.current_player
    
    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        """
        Check if game is over
        Returns: (is_terminal, winner) where winner is None if game continues,
                 0/1 for player who won, or -1 for draw (all nodes colored)
        """
        # Check if all nodes are colored
        if len(self.coloring) == len(self.G.nodes()):
            return True, -1  # Draw
        
        # Check if current player has any legal moves
        has_moves = any(self.get_legal_moves(node) 
                       for node in self.G.nodes() 
                       if node not in self.coloring)
        
        if not has_moves:
            return True, 1 - self.current_player  # Previous player wins
        
        return False, None
    
    def uncolored_nodes(self) -> List[int]:
        """Get list of uncolored nodes"""
        return [n for n in self.G.nodes() if n not in self.coloring]


class OptimalStrategyEngine:
    """
    Implements optimal and heuristic strategies for graph coloring games
    """
    
    def __init__(self, game_state: GameState):
        self.state = game_state
        
    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, 
                maximizing: bool, max_depth: int = 5) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Minimax with alpha-beta pruning
        Returns: (evaluation score, best move)
        """
        is_terminal, winner = state.is_terminal()
        
        if is_terminal:
            if winner == -1:  # Draw
                return 0, None
            
            # If a terminal state is reached, the current player has no moves and has lost.
            if maximizing:
                return -1000, None  # The maximizing player has lost.
            else:
                return 1000, None   # The minimizing player has lost, so the maximizing player wins.
        
        if depth >= max_depth:
            return self.evaluate_position(state), None
        
        best_move = None
        moves = state.get_all_legal_moves()
        
        if not moves:
            # This case should be covered by is_terminal, but as a fallback
            return (-1000, None) if maximizing else (1000, None)
        
        if maximizing:
            max_eval = float('-inf')
            for node, color in moves:
                new_state = state.clone()
                new_state.make_move(node, color)
                eval_score, _ = self.minimax(new_state, depth + 1, alpha, beta, False, max_depth)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (node, color)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for node, color in moves:
                new_state = state.clone()
                new_state.make_move(node, color)
                eval_score, _ = self.minimax(new_state, depth + 1, alpha, beta, True, max_depth)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (node, color)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval, best_move
    
    def evaluate_position(self, state: GameState) -> float:
        """
        Heuristic evaluation function for non-terminal positions
        Higher is better for current player
        """
        score = 0
        
        # Factor 1: Number of remaining moves (mobility)
        uncolored = state.uncolored_nodes()
        current_mobility = sum(len(state.get_legal_moves(node)) for node in uncolored)
        
        # Simulate opponent's mobility
        opponent_state = state.clone()
        opponent_state.current_player = 1 - state.current_player
        opponent_mobility = sum(len(opponent_state.get_legal_moves(node)) for node in uncolored)
        
        score += (current_mobility - opponent_mobility) * 10
        
        # Factor 2: Control of high-degree vertices
        for node in uncolored:
            degree = state.G.degree(node)
            moves = len(state.get_legal_moves(node))
            
            if moves > 0:
                # Prefer positions where we have options on important nodes
                score += degree * moves * 2
        
        # Factor 3: Number of uncolored nodes (want to delay running out of moves)
        score += len(uncolored) * 5
        
        return score
    
    def find_best_move(self, max_depth: int = 5) -> Tuple[int, int]:
        """
        Find the best move using minimax algorithm
        Returns: (node, color)
        """
        _, best_move = self.minimax(self.state, 0, float('-inf'), float('inf'), True, max_depth)
        
        if best_move is None:
            # Fallback to heuristic if minimax fails
            best_move = self.heuristic_move()
        
        return best_move
    
    def heuristic_move(self) -> Tuple[int, int]:
        """
        Fast heuristic-based move selection
        Strategy: Color high-degree nodes with maximum flexibility
        """
        uncolored = self.state.uncolored_nodes()
        
        best_score = float('-inf')
        best_move = None
        
        for node in uncolored:
            legal_colors = self.state.get_legal_moves(node)
            
            if not legal_colors:
                continue
            
            degree = self.state.G.degree(node)
            
            # Count uncolored neighbors
            uncolored_neighbors = sum(1 for n in self.state.G.neighbors(node) 
                                     if n not in self.state.coloring)
            
            for color in legal_colors:
                # Score based on:
                # 1. Node importance (degree)
                # 2. Number of uncolored neighbors (blocking potential)
                # 3. Color flexibility (how many options this leaves)
                score = degree * 10 + uncolored_neighbors * 5 + len(legal_colors)
                
                if score > best_score:
                    best_score = score
                    best_move = (node, color)
        
        return best_move if best_move else (uncolored[0], self.state.get_legal_moves(uncolored[0])[0])


class VariableNodesStrategy:
    """
    Strategy for game variant where players can color multiple nodes per turn
    """
    
    def __init__(self, game_state: GameState, max_nodes_per_turn: int):
        self.state = game_state
        self.max_nodes = max_nodes_per_turn
    
    def find_best_multi_move(self) -> List[Tuple[int, int]]:
        """
        Find best set of nodes to color in one turn (same color)
        Strategy: Find independent set of high-value nodes
        """
        uncolored = self.state.uncolored_nodes()
        
        best_set = []
        best_score = float('-inf')
        
        # Try each color
        for color in range(self.state.num_colors):
            # Find nodes that can be colored with this color
            candidates = [n for n in uncolored if color in self.state.get_legal_moves(n)]
            
            if not candidates:
                continue
            
            # Find independent set among candidates (no two are adjacent)
            independent_set = self._find_independent_set(candidates, self.max_nodes)
            
            # Score this set
            score = sum(self.state.G.degree(n) for n in independent_set)
            
            if score > best_score:
                best_score = score
                best_set = [(n, color) for n in independent_set]
        
        return best_set if best_set else [self.heuristic_single_move()]
    
    def _find_independent_set(self, nodes: List[int], max_size: int) -> List[int]:
        """Find maximal independent set using greedy algorithm"""
        independent = []
        remaining = set(nodes)
        
        while remaining and len(independent) < max_size:
            # Choose node with minimum degree in remaining subgraph
            node = min(remaining, key=lambda n: sum(1 for x in remaining if self.state.G.has_edge(n, x)))
            independent.append(node)
            
            # Remove node and its neighbors
            remaining.remove(node)
            remaining -= set(self.state.G.neighbors(node)) & remaining
        
        return independent
    
    def heuristic_single_move(self) -> Tuple[int, int]:
        """Fallback to single move if multi-move fails"""
        engine = OptimalStrategyEngine(self.state)
        return engine.heuristic_move()


class ResourceLimitedStrategy:
    """
    Strategy for resource-limited variant where each color has limited uses
    """
    
    def __init__(self, game_state: GameState, color_resources: List[int]):
        self.state = game_state
        self.resources = color_resources
    
    def find_best_move_with_resources(self) -> Tuple[int, int]:
        """
        Find best move considering resource constraints
        Strategy: Conserve rare colors, use abundant ones first
        """
        uncolored = self.state.uncolored_nodes()
        
        best_score = float('-inf')
        best_move = None
        
        for node in uncolored:
            legal_colors = self.state.get_legal_moves(node)
            
            # Filter by available resources
            available_colors = [c for c in legal_colors if self.resources[c] > 0]
            
            if not available_colors:
                continue
            
            degree = self.state.G.degree(node)
            
            for color in available_colors:
                # Penalize using rare colors unless necessary
                rarity_penalty = 100 / (self.resources[color] + 1)
                
                # Count how many uncolored neighbors need different color
                critical_neighbors = sum(1 for n in self.state.G.neighbors(node)
                                       if n not in self.state.coloring and
                                       color in self.state.get_legal_moves(n))
                
                # Score: prioritize important nodes, but conserve rare colors
                score = degree * 10 + critical_neighbors * 5 - rarity_penalty
                
                if score > best_score:
                    best_score = score
                    best_move = (node, color)
        
        return best_move


# ==================== DEMO AND TESTING ====================

def create_test_graphs():
    """Create various test graphs"""
    graphs = {}
    
    # Petersen graph (chromatic number = 3)
    G_petersen = nx.petersen_graph()
    graphs['petersen'] = G_petersen
    
    # Complete graph K5 (chromatic number = 5)
    graphs['complete_5'] = nx.complete_graph(5)
    
    # Cycle graph C5 (chromatic number = 3)
    graphs['cycle_5'] = nx.cycle_graph(5)
    
    # Wheel graph W7 (chromatic number = 3)
    graphs['wheel_7'] = nx.wheel_graph(7)
    
    # Random graph
    graphs['random'] = nx.erdos_renyi_graph(10, 0.3)
    
    return graphs


def demo_chromatic_number():
    """Demonstrate chromatic number algorithms"""
    print("="*60)
    print("CHROMATIC NUMBER COMPUTATION")
    print("="*60)
    
    graphs = create_test_graphs()
    
    for name, G in graphs.items():
        print(f"\n{name.upper()} Graph:")
        print(f"  Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
        
        solver = ChromaticNumberSolver(G)
        
        # Lower bound
        lower = solver.clique_lower_bound()
        print(f"  Lower bound (max clique): {lower}")
        
        # Upper bound (greedy)
        greedy_colors, _ = solver.greedy_chromatic_upper_bound()
        print(f"  Upper bound (greedy): {greedy_colors}")
        
        # DSatur
        dsatur_colors, _ = solver.dsatur_coloring()
        print(f"  DSatur coloring: {dsatur_colors}")
        
        # Exact (only for small graphs)
        if len(G.nodes()) <= 10:
            start = time.time()
            exact = solver.exact_chromatic_number(max_colors=greedy_colors)
            elapsed = time.time() - start
            print(f"  Exact chromatic number: {exact} (computed in {elapsed:.3f}s)")


def demo_game_strategies():
    """Demonstrate game playing strategies"""
    print("\n" + "="*60)
    print("GAME STRATEGY DEMONSTRATION")
    print("="*60)
    
    # Create game on Petersen graph
    G = nx.petersen_graph()
    num_colors = 2  # Less than chromatic number (3)
    
    print(f"\nGame Setup:")
    print(f"  Graph: Petersen (chromatic number = 3)")
    print(f"  Available colors: {num_colors}")
    print(f"  Players: 2 (optimal AI vs optimal AI)")
    
    state = GameState(G, num_colors)
    move_count = 0
    
    print("\nGame Progress:")
    while True:
        is_terminal, winner = state.is_terminal()
        
        if is_terminal:
            if winner == -1:
                print(f"\nGame ended in a draw! All {len(G.nodes())} nodes colored.")
            else:
                print(f"\nPlayer {winner + 1} wins after {move_count} moves!")
                print(f"Player {state.current_player + 1} has no legal moves.")
            break
        
        # Find best move
        engine = OptimalStrategyEngine(state)
        node, color = engine.find_best_move(max_depth=4)
        
        move_count += 1
        print(f"  Move {move_count}: Player {state.current_player + 1} colors node {node} with color {color}")
        
        state.make_move(node, color)
        
        # Safety limit
        if move_count > 20:
            print("\n(Game truncated for demo)")
            break


def demo_variable_nodes_strategy():
    """Demonstrate variable nodes per turn strategy"""
    print("\n" + "="*60)
    print("VARIABLE NODES STRATEGY")
    print("="*60)
    
    G = nx.cycle_graph(8)
    state = GameState(G, 3)
    
    print(f"\nGame: Cycle graph C8, up to 2 nodes per turn")
    
    strategy = VariableNodesStrategy(state, max_nodes_per_turn=2)
    moves = strategy.find_best_multi_move()
    
    print(f"Recommended multi-move:")
    for node, color in moves:
        print(f"  Color node {node} with color {color}")

def demo_resource_limited_strategy():
    """Demonstrate resource-limited strategy"""
    print("\n" + "="*60)
    print("RESOURCE-LIMITED STRATEGY")
    print("="*60)
    
    G = nx.petersen_graph()
    num_colors = 3
    
    # Each color can be used a limited number of times
    color_resources = [3, 3, 4] 
    
    state = GameState(G, num_colors)
    
    print(f"\nGame: Petersen graph, {num_colors} colors")
    print(f"Color resources: {color_resources}")
    
    # Make a few moves to create a non-trivial state
    state.make_move(0, 2)
    state.make_move(1, 0)
    color_resources[2] -= 1
    color_resources[0] -= 1
    
    print(f"Initial moves made. Player {state.current_player + 1}'s turn.")
    print(f"Remaining resources: {color_resources}")
    
    strategy = ResourceLimitedStrategy(state, color_resources)
    move = strategy.find_best_move_with_resources()
    
    if move:
        node, color = move
        print(f"\nRecommended move with resource constraints:")
        print(f"  Color node {node} with color {color} (remaining: {color_resources[color]-1})")
    else:
        print("\nNo legal moves found with available resources.")


if __name__ == "__main__":
    # Run all demonstrations
    demo_chromatic_number()
    demo_game_strategies()
    demo_variable_nodes_strategy()
    demo_resource_limited_strategy()
    
    print("\n" + "="*60)
    print("STRATEGIC INSIGHTS")
    print("="*60)
    print("""
Key Strategies for Competitive Graph Coloring:

1. BASE GAME (1 node per turn, C < χ(G)):
   - Control high-degree vertices early
   - Force opponent into positions with limited choices
   - Maintain color flexibility for yourself
   - Use minimax with evaluation based on mobility

2. VARIABLE NODES (k nodes per turn, same color):
   - Find maximum independent sets
   - Prioritize nodes that block opponent's options
   - Balance between coloring many nodes vs. strategic positions

3. RESOURCE-LIMITED:
   - Conserve rare colors for critical positions
   - Use abundant colors on less important nodes
   - Plan color usage over entire game horizon

4. VARIABLE NODES (different colors):
   - Maximize board control per turn
   - Color complementary nodes that strengthen your position
   - Leave opponent with difficult choices

Complexity Analysis:
- Chromatic number: NP-complete
- Optimal game play: PSPACE-complete
- Heuristic strategies run in polynomial time
""")
