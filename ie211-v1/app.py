"""
Flask API server for Graph Coloring Game
Exposes the game logic from colour.py as REST API endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
from colour import (
    GameState, 
    OptimalStrategyEngine, 
    ChromaticNumberSolver,
    VariableNodesStrategy,
    ResourceLimitedStrategy
)
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Store game sessions (in production, use Redis or database)
game_sessions = {}


def graph_from_adjacency(adj_dict):
    """Convert adjacency dictionary to NetworkX graph"""
    G = nx.Graph()
    for node, neighbors in adj_dict.items():
        G.add_node(int(node))
        for neighbor in neighbors:
            G.add_edge(int(node), int(neighbor))
    return G


def adjacency_from_graph(G):
    """Convert NetworkX graph to adjacency dictionary"""
    adj = {}
    for node in G.nodes():
        adj[node] = list(G.neighbors(node))
    return adj


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/api/graph/chromatic-number', methods=['POST'])
def get_chromatic_number():
    """Calculate chromatic number for a graph"""
    try:
        data = request.json
        adj = data.get('adjacency', {})
        
        if not adj:
            return jsonify({'error': 'Adjacency list required'}), 400
        
        G = graph_from_adjacency(adj)
        solver = ChromaticNumberSolver(G)
        
        # Get bounds
        lower_bound = solver.clique_lower_bound()
        greedy_colors, greedy_coloring = solver.greedy_chromatic_upper_bound()
        dsatur_colors, dsatur_coloring = solver.dsatur_coloring()
        
        # Try exact for small graphs
        exact = None
        if len(G.nodes()) <= 10:
            try:
                exact = solver.exact_chromatic_number(max_colors=greedy_colors)
            except:
                pass
        
        return jsonify({
            'lower_bound': lower_bound,
            'upper_bound': greedy_colors,
            'dsatur': dsatur_colors,
            'exact': exact,
            'greedy_coloring': {str(k): v for k, v in greedy_coloring.items()},
            'dsatur_coloring': {str(k): v for k, v in dsatur_coloring.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/init', methods=['POST'])
def init_game():
    """Initialize a new game"""
    try:
        data = request.json
        adj = data.get('adjacency', {})
        num_colors = data.get('num_colors', 3)
        game_mode = data.get('game_mode', 'base')
        
        if not adj:
            return jsonify({'error': 'Adjacency list required'}), 400
        
        G = graph_from_adjacency(adj)
        game_state = GameState(G, num_colors)
        
        # Create session
        import uuid
        session_id = str(uuid.uuid4())
        game_sessions[session_id] = {
            'game_state': game_state,
            'game_mode': game_mode,
            'max_nodes_per_turn': data.get('max_nodes_per_turn', 1),
            'color_resources': data.get('color_resources', None)
        }
        
        # Get chromatic number info
        solver = ChromaticNumberSolver(G)
        lower_bound = solver.clique_lower_bound()
        greedy_colors, _ = solver.greedy_chromatic_upper_bound()
        
        return jsonify({
            'session_id': session_id,
            'chromatic_number': {
                'lower_bound': lower_bound,
                'upper_bound': greedy_colors
            },
            'game_state': {
                'coloring': {str(k): v for k, v in game_state.coloring.items()},
                'current_player': game_state.current_player,
                'num_colors': game_state.num_colors
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/move', methods=['POST'])
def make_move():
    """Make a move in the game"""
    try:
        data = request.json
        session_id = data.get('session_id')
        node = data.get('node')
        color = data.get('color')
        
        if not session_id or session_id not in game_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = game_sessions[session_id]
        game_state = session['game_state']
        
        # Check if move is legal
        if node in game_state.coloring:
            return jsonify({'error': 'Node already colored'}), 400
        
        legal_moves = game_state.get_legal_moves(node)
        if color not in legal_moves:
            return jsonify({'error': 'Illegal move'}), 400
        
        # Check resource limits if applicable
        if session['game_mode'] == 'resourceLimited' and session['color_resources']:
            resources = session['color_resources'][game_state.current_player]
            if resources[color] <= 0:
                return jsonify({'error': 'Color resource exhausted'}), 400
            resources[color] -= 1
        
        # Make the move
        game_state.make_move(node, color)
        
        # Check game status
        is_terminal, winner = game_state.is_terminal()
        
        return jsonify({
            'game_state': {
                'coloring': {str(k): v for k, v in game_state.coloring.items()},
                'current_player': game_state.current_player,
                'num_colors': game_state.num_colors
            },
            'game_over': is_terminal,
            'winner': winner if is_terminal else None,
            'uncolored_nodes': game_state.uncolored_nodes()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/ai-move', methods=['POST'])
def ai_move():
    """Get AI's best move"""
    try:
        data = request.json
        session_id = data.get('session_id')
        max_depth = data.get('max_depth', 4)
        
        if not session_id or session_id not in game_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = game_sessions[session_id]
        game_state = session['game_state']
        
        # Check if game is over
        is_terminal, winner = game_state.is_terminal()
        if is_terminal:
            return jsonify({
                'game_over': True,
                'winner': winner
            })
        
        # Find best move
        engine = OptimalStrategyEngine(game_state)
        best_move = engine.find_best_move(max_depth=max_depth)
        
        if not best_move:
            # Fallback to heuristic
            best_move = engine.heuristic_move()
        
        if not best_move:
            return jsonify({'error': 'No legal moves available'}), 400
        
        node, color = best_move
        
        return jsonify({
            'node': node,
            'color': color
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/state', methods=['POST'])
def get_game_state():
    """Get current game state"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in game_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = game_sessions[session_id]
        game_state = session['game_state']
        
        is_terminal, winner = game_state.is_terminal()
        
        return jsonify({
            'game_state': {
                'coloring': {str(k): v for k, v in game_state.coloring.items()},
                'current_player': game_state.current_player,
                'num_colors': game_state.num_colors
            },
            'game_over': is_terminal,
            'winner': winner if is_terminal else None,
            'uncolored_nodes': game_state.uncolored_nodes(),
            'legal_moves': {
                str(node): game_state.get_legal_moves(node)
                for node in game_state.uncolored_nodes()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset a game session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id and session_id in game_sessions:
            session = game_sessions[session_id]
            G = session['game_state'].G
            num_colors = session['game_state'].num_colors
            game_state = GameState(G, num_colors)
            session['game_state'] = game_state
            
            return jsonify({
                'game_state': {
                    'coloring': {},
                    'current_player': 0,
                    'num_colors': num_colors
                }
            })
        else:
            return jsonify({'error': 'Invalid session ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')

