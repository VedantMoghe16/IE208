import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Info, Settings, Cpu, Users } from 'lucide-react';

// Graph coloring algorithms and game logic
class GraphColoringSystem {
  constructor(adjacencyList) {
    this.adj = adjacencyList;
    this.n = Object.keys(adjacencyList).length;
  }

  // Find chromatic number using backtracking
  findChromaticNumber() {
    for (let colors = 1; colors <= this.n; colors++) {
      const coloring = new Array(this.n).fill(-1);
      if (this.canColorWithK(0, colors, coloring)) {
        return colors;
      }
    }
    return this.n;
  }

  canColorWithK(vertex, k, coloring) {
    if (vertex === this.n) return true;

    for (let color = 0; color < k; color++) {
      if (this.isSafe(vertex, color, coloring)) {
        coloring[vertex] = color;
        if (this.canColorWithK(vertex + 1, k, coloring)) return true;
        coloring[vertex] = -1;
      }
    }
    return false;
  }

  isSafe(vertex, color, coloring) {
    for (const neighbor of this.adj[vertex]) {
      if (coloring[neighbor] === color) return false;
    }
    return true;
  }

  // Greedy coloring (upper bound on chromatic number)
  greedyColoring() {
    const coloring = new Array(this.n).fill(-1);
    coloring[0] = 0;

    for (let v = 1; v < this.n; v++) {
      const available = new Set(Array.from({ length: this.n }, (_, i) => i));
      
      for (const neighbor of this.adj[v]) {
        if (coloring[neighbor] !== -1) {
          available.delete(coloring[neighbor]);
        }
      }
      
      coloring[v] = Math.min(...available);
    }

    return Math.max(...coloring) + 1;
  }

  // Check if a move is legal
  isLegalMove(vertex, color, currentColoring) {
    for (const neighbor of this.adj[vertex]) {
      if (currentColoring[neighbor] === color) return false;
    }
    return true;
  }

  // Get available moves for a vertex
  getAvailableMoves(vertex, numColors, currentColoring) {
    const moves = [];
    for (let c = 0; c < numColors; c++) {
      if (this.isLegalMove(vertex, c, currentColoring)) {
        moves.push(c);
      }
    }
    return moves;
  }

  // Minimax with alpha-beta pruning for optimal strategy
  minimax(coloring, depth, isMaximizing, alpha, beta, numColors, maxDepth = 5) {
    const uncolored = coloring.map((c, i) => c === -1 ? i : -1).filter(x => x !== -1);
    
    if (uncolored.length === 0) return isMaximizing ? -1000 : 1000;
    if (depth >= maxDepth) return this.evaluatePosition(coloring, isMaximizing);

    if (isMaximizing) {
      let maxEval = -Infinity;
      for (const v of uncolored) {
        const moves = this.getAvailableMoves(v, numColors, coloring);
        if (moves.length === 0) return -1000; // Loss
        
        for (const color of moves) {
          const newColoring = [...coloring];
          newColoring[v] = color;
          const eval_score = this.minimax(newColoring, depth + 1, false, alpha, beta, numColors, maxDepth);
          maxEval = Math.max(maxEval, eval_score);
          alpha = Math.max(alpha, eval_score);
          if (beta <= alpha) break;
        }
      }
      return maxEval;
    } else {
      let minEval = Infinity;
      for (const v of uncolored) {
        const moves = this.getAvailableMoves(v, numColors, coloring);
        if (moves.length === 0) return 1000; // Win for maximizer
        
        for (const color of moves) {
          const newColoring = [...coloring];
          newColoring[v] = color;
          const eval_score = this.minimax(newColoring, depth + 1, true, alpha, beta, numColors, maxDepth);
          minEval = Math.min(minEval, eval_score);
          beta = Math.min(beta, eval_score);
          if (beta <= alpha) break;
        }
      }
      return minEval;
    }
  }

  evaluatePosition(coloring, isMaximizing) {
    const uncolored = coloring.filter(c => c === -1).length;
    let flexibility = 0;
    
    for (let v = 0; v < this.n; v++) {
      if (coloring[v] === -1) {
        flexibility += this.getAvailableMoves(v, 3, coloring).length;
      }
    }
    
    return isMaximizing ? flexibility - uncolored * 2 : uncolored * 2 - flexibility;
  }

  // Find best move using minimax
  findBestMove(coloring, numColors, isMaximizing) {
    const uncolored = coloring.map((c, i) => c === -1 ? i : -1).filter(x => x !== -1);
    let bestMove = null;
    let bestValue = isMaximizing ? -Infinity : Infinity;

    for (const v of uncolored) {
      const moves = this.getAvailableMoves(v, numColors, coloring);
      
      for (const color of moves) {
        const newColoring = [...coloring];
        newColoring[v] = color;
        const value = this.minimax(newColoring, 0, !isMaximizing, -Infinity, Infinity, numColors, 4);
        
        if ((isMaximizing && value > bestValue) || (!isMaximizing && value < bestValue)) {
          bestValue = value;
          bestMove = { vertex: v, color };
        }
      }
    }
    
    return bestMove;
  }
}

const GraphColoringGame = () => {
  const [gameMode, setGameMode] = useState('base'); // base, variableNodes, resourceLimited, variableColors
  const [playerMode, setPlayerMode] = useState('pvp'); // pvp, pve
  const [graph, setGraph] = useState(null);
  const [coloring, setColoring] = useState([]);
  const [currentPlayer, setCurrentPlayer] = useState(0);
  const [numColors, setNumColors] = useState(3);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  const [chromaticNumber, setChromaticNumber] = useState(null);
  const [selectedVertex, setSelectedVertex] = useState(null);
  const [selectedColor, setSelectedColor] = useState(null);
  const [maxNodesPerTurn, setMaxNodesPerTurn] = useState(2);
  const [coloredThisTurn, setColoredThisTurn] = useState([]);
  const [resourceLimits, setResourceLimits] = useState([3, 3, 3, 3, 3]);
  const [p1Resources, setP1Resources] = useState([3, 3, 3, 3, 3]);
  const [p2Resources, setP2Resources] = useState([3, 3, 3, 3, 3]);
  const [showInfo, setShowInfo] = useState(false);
  const canvasRef = useRef(null);

  const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];
  const playerColors = ['#dc2626', '#2563eb'];

  // Generate a sample graph
  const generateGraph = (type = 'petersen') => {
    let adj = {};
    let positions = [];
    
    if (type === 'petersen') {
      // Petersen graph (chromatic number = 3)
      const n = 10;
      for (let i = 0; i < n; i++) adj[i] = [];
      
      // Outer pentagon
      for (let i = 0; i < 5; i++) {
        adj[i].push((i + 1) % 5);
        adj[(i + 1) % 5].push(i);
      }
      
      // Inner pentagram
      for (let i = 0; i < 5; i++) {
        adj[i + 5].push(5 + (i + 2) % 5);
        adj[5 + (i + 2) % 5].push(i + 5);
      }
      
      // Connect outer to inner
      for (let i = 0; i < 5; i++) {
        adj[i].push(i + 5);
        adj[i + 5].push(i);
      }
      
      // Positions for visualization
      for (let i = 0; i < 5; i++) {
        const angle = (i * 2 * Math.PI / 5) - Math.PI / 2;
        positions.push({ x: 250 + 150 * Math.cos(angle), y: 250 + 150 * Math.sin(angle) });
      }
      for (let i = 0; i < 5; i++) {
        const angle = (i * 2 * Math.PI / 5) - Math.PI / 2;
        positions.push({ x: 250 + 70 * Math.cos(angle), y: 250 + 70 * Math.sin(angle) });
      }
    } else if (type === 'wheel') {
      // Wheel graph (chromatic number = 3 or 4 depending on size)
      const n = 7;
      for (let i = 0; i < n; i++) adj[i] = [];
      
      // Center node connected to all
      for (let i = 1; i < n; i++) {
        adj[0].push(i);
        adj[i].push(0);
      }
      
      // Cycle
      for (let i = 1; i < n; i++) {
        adj[i].push(i === n - 1 ? 1 : i + 1);
        adj[i === n - 1 ? 1 : i + 1].push(i);
      }
      
      positions.push({ x: 250, y: 250 });
      for (let i = 1; i < n; i++) {
        const angle = ((i - 1) * 2 * Math.PI / (n - 1)) - Math.PI / 2;
        positions.push({ x: 250 + 150 * Math.cos(angle), y: 250 + 150 * Math.sin(angle) });
      }
    } else {
      // Random graph
      const n = 8;
      for (let i = 0; i < n; i++) adj[i] = [];
      
      // Add random edges
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.random() < 0.35) {
            adj[i].push(j);
            adj[j].push(i);
          }
        }
      }
      
      // Random positions
      for (let i = 0; i < n; i++) {
        positions.push({
          x: 100 + Math.random() * 300,
          y: 100 + Math.random() * 300
        });
      }
    }
    
    return { adj, positions };
  };

  const initializeGame = () => {
    const g = generateGraph('petersen');
    setGraph(g);
    const gcs = new GraphColoringSystem(g.adj);
    const chromNum = gcs.findChromaticNumber();
    setChromaticNumber(chromNum);
    setColoring(new Array(Object.keys(g.adj).length).fill(-1));
    setCurrentPlayer(0);
    setGameOver(false);
    setWinner(null);
    setSelectedVertex(null);
    setSelectedColor(null);
    setColoredThisTurn([]);
    setP1Resources([...resourceLimits]);
    setP2Resources([...resourceLimits]);
  };

  useEffect(() => {
    initializeGame();
  }, []);

  useEffect(() => {
    if (graph) drawGraph();
  }, [graph, coloring, selectedVertex]);

  useEffect(() => {
    if (playerMode === 'pve' && currentPlayer === 1 && !gameOver && graph) {
      // AI move with delay
      setTimeout(() => {
        makeAIMove();
      }, 500);
    }
  }, [currentPlayer, playerMode, gameOver]);

  const drawGraph = () => {
    if (!canvasRef.current || !graph) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 500, 500);
    
    // Draw edges
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    Object.keys(graph.adj).forEach(v => {
      graph.adj[v].forEach(u => {
        if (parseInt(v) < parseInt(u)) {
          ctx.beginPath();
          ctx.moveTo(graph.positions[v].x, graph.positions[v].y);
          ctx.lineTo(graph.positions[u].x, graph.positions[u].y);
          ctx.stroke();
        }
      });
    });
    
    // Draw vertices
    graph.positions.forEach((pos, i) => {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 20, 0, 2 * Math.PI);
      
      if (coloring[i] !== -1) {
        ctx.fillStyle = colors[coloring[i]];
      } else if (selectedVertex === i) {
        ctx.fillStyle = '#fbbf24';
      } else {
        ctx.fillStyle = '#f1f5f9';
      }
      
      ctx.fill();
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw vertex number
      ctx.fillStyle = '#000';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(i, pos.x, pos.y);
    });
  };

  const handleCanvasClick = (e) => {
    if (!graph || gameOver) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if clicked on a vertex
    for (let i = 0; i < graph.positions.length; i++) {
      const pos = graph.positions[i];
      const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
      
      if (dist < 20 && coloring[i] === -1) {
        setSelectedVertex(i);
        return;
      }
    }
  };

  const makeMove = (vertex, color) => {
    if (gameOver || coloring[vertex] !== -1) return false;
    
    const gcs = new GraphColoringSystem(graph.adj);
    if (!gcs.isLegalMove(vertex, color, coloring)) return false;
    
    // Check resource limits for resource-limited mode
    if (gameMode === 'resourceLimited') {
      const resources = currentPlayer === 0 ? p1Resources : p2Resources;
      if (resources[color] <= 0) return false;
      
      const newResources = [...resources];
      newResources[color]--;
      if (currentPlayer === 0) {
        setP1Resources(newResources);
      } else {
        setP2Resources(newResources);
      }
    }
    
    const newColoring = [...coloring];
    newColoring[vertex] = color;
    setColoring(newColoring);
    
    if (gameMode === 'variableNodes' || gameMode === 'variableColors') {
      const newColoredThisTurn = [...coloredThisTurn, vertex];
      setColoredThisTurn(newColoredThisTurn);
      
      if (newColoredThisTurn.length >= maxNodesPerTurn) {
        endTurn(newColoring);
      }
    } else {
      endTurn(newColoring);
    }
    
    setSelectedVertex(null);
    return true;
  };

  const endTurn = (newColoring) => {
    // Check if next player has any legal moves
    const gcs = new GraphColoringSystem(graph.adj);
    const uncolored = newColoring.map((c, i) => c === -1 ? i : -1).filter(x => x !== -1);
    
    let hasLegalMove = false;
    for (const v of uncolored) {
      const moves = gcs.getAvailableMoves(v, numColors, newColoring);
      
      if (gameMode === 'resourceLimited') {
        const nextPlayer = 1 - currentPlayer;
        const resources = nextPlayer === 0 ? p1Resources : p2Resources;
        const availableMoves = moves.filter(c => resources[c] > 0);
        if (availableMoves.length > 0) {
          hasLegalMove = true;
          break;
        }
      } else {
        if (moves.length > 0) {
          hasLegalMove = true;
          break;
        }
      }
    }
    
    if (!hasLegalMove || uncolored.length === 0) {
      setGameOver(true);
      setWinner(uncolored.length === 0 ? 'draw' : currentPlayer);
    } else {
      setCurrentPlayer(1 - currentPlayer);
      setColoredThisTurn([]);
    }
  };

  const makeAIMove = () => {
    if (!graph || gameOver) return;
    
    const gcs = new GraphColoringSystem(graph.adj);
    const move = gcs.findBestMove(coloring, numColors, true);
    
    if (move) {
      makeMove(move.vertex, move.color);
    } else {
      setGameOver(true);
      setWinner(0);
    }
  };

  const skipTurn = () => {
    if (gameMode === 'variableNodes' || gameMode === 'variableColors') {
      endTurn(coloring);
    }
  };

  return (
    <div className="w-full h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-3xl font-bold mb-2">Competitive Graph Coloring</h1>
            <p className="text-slate-400">Strategic graph coloring game with multiple modes</p>
          </div>
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="p-2 bg-slate-700 rounded hover:bg-slate-600"
          >
            <Info size={20} />
          </button>
        </div>

        {showInfo && (
          <div className="bg-slate-800 p-4 rounded-lg mb-4 border border-slate-700">
            <h3 className="font-bold mb-2">Game Rules:</h3>
            <ul className="text-sm text-slate-300 space-y-1">
              <li>• Players alternate coloring vertices</li>
              <li>• Adjacent vertices cannot have the same color</li>
              <li>• Chromatic number χ(G) = {chromaticNumber}</li>
              <li>• Available colors: {numColors} (may be &lt; χ(G))</li>
              <li>• Player who cannot move loses</li>
            </ul>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Game Canvas */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: playerColors[0] }}
                    />
                    <span>Player 1 {currentPlayer === 0 && !gameOver && '(Turn)'}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: playerColors[1] }}
                    />
                    <span>{playerMode === 'pve' ? 'AI' : 'Player 2'} {currentPlayer === 1 && !gameOver && '(Turn)'}</span>
                  </div>
                </div>
                <button
                  onClick={initializeGame}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-700 rounded hover:bg-slate-600"
                >
                  <RotateCcw size={16} />
                  New Game
                </button>
              </div>

              <canvas
                ref={canvasRef}
                width={500}
                height={500}
                onClick={handleCanvasClick}
                className="border border-slate-700 rounded cursor-pointer bg-slate-900"
              />

              {gameOver && (
                <div className="mt-4 p-4 bg-green-900/50 border border-green-700 rounded text-center">
                  <h3 className="text-xl font-bold">
                    {winner === 'draw' ? 'All vertices colored!' : `Player ${winner + 1} wins!`}
                  </h3>
                </div>
              )}

              {selectedVertex !== null && !gameOver && (
                <div className="mt-4">
                  <p className="text-sm mb-2">Select color for vertex {selectedVertex}:</p>
                  <div className="flex gap-2">
                    {colors.slice(0, numColors).map((color, i) => {
                      const canUse = gameMode !== 'resourceLimited' || 
                        (currentPlayer === 0 ? p1Resources[i] : p2Resources[i]) > 0;
                      
                      return (
                        <button
                          key={i}
                          onClick={() => makeMove(selectedVertex, i)}
                          disabled={!canUse}
                          className="w-12 h-12 rounded border-2 border-slate-700 hover:border-white disabled:opacity-30"
                          style={{ backgroundColor: color }}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-4">
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
              <h3 className="font-bold mb-3">Game Settings</h3>
              
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-400 block mb-1">Mode</label>
                  <select
                    value={playerMode}
                    onChange={(e) => setPlayerMode(e.target.value)}
                    className="w-full p-2 bg-slate-700 rounded text-sm"
                  >
                    <option value="pvp">Player vs Player</option>
                    <option value="pve">Player vs AI</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm text-slate-400 block mb-1">Game Variant</label>
                  <select
                    value={gameMode}
                    onChange={(e) => setGameMode(e.target.value)}
                    className="w-full p-2 bg-slate-700 rounded text-sm"
                  >
                    <option value="base">Base (1 node/turn)</option>
                    <option value="variableNodes">Variable nodes (same color)</option>
                    <option value="resourceLimited">Resource Limited</option>
                    <option value="variableColors">Variable (different colors)</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm text-slate-400 block mb-1">
                    Colors: {numColors}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="5"
                    value={numColors}
                    onChange={(e) => setNumColors(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                {(gameMode === 'variableNodes' || gameMode === 'variableColors') && (
                  <>
                    <div>
                      <label className="text-sm text-slate-400 block mb-1">
                        Max nodes/turn: {maxNodesPerTurn}
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="4"
                        value={maxNodesPerTurn}
                        onChange={(e) => setMaxNodesPerTurn(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>
                    <button
                      onClick={skipTurn}
                      disabled={coloredThisTurn.length === 0 || gameOver}
                      className="w-full px-4 py-2 bg-amber-600 rounded hover:bg-amber-700 disabled:opacity-50"
                    >
                      End Turn ({coloredThisTurn.length}/{maxNodesPerTurn})
                    </button>
                  </>
                )}
              </div>
            </div>

            {gameMode === 'resourceLimited' && (
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                <h3 className="font-bold mb-3">Resources</h3>
                <div className="space-y-2">
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Player 1:</p>
                    <div className="flex gap-1">
                      {p1Resources.slice(0, numColors).map((r, i) => (
                        <div key={i} className="flex-1 text-center">
                          <div className="w-full h-8 rounded" style={{ backgroundColor: colors[i] }} />
                          <span className="text-xs">{r}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-slate-400 mb-1">Player 2:</p>
                    <div className="flex gap-1">
                      {p2Resources.slice(0, numColors).map((r, i) => (
                        <div key={i} className="flex-1 text-center">
                          <div className="w-full h-8 rounded" style={{ backgroundColor: colors[i] }} />
                          <span className="text-xs">{r}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
              <h3 className="font-bold mb-3">Strategy Tips</h3>
              <ul className="text-sm text-slate-300 space-y-2">
                <li>• Control vertices with high degree</li>
                <li>• Force opponent into corners</li>
                <li>• Maximize your color flexibility</li>
                <li>• Plan 2-3 moves ahead</li>
                {gameMode === 'resourceLimited' && (
                  <li>• Conserve rare color resources</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphColoringGame;