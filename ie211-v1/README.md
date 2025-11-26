# Graph Coloring Game

A competitive graph coloring game with multiple game modes, AI opponent, and strategic gameplay. This project integrates a Python backend (Flask API) with a React frontend to provide a robust, interactive gaming experience.

## Features

- **Multiple Graph Types**: Petersen, Wheel, Cycle, and Random graphs
- **Game Modes**:
  - Base mode (1 node per turn)
  - Variable nodes mode (multiple nodes per turn, same color)
  - Resource-limited mode (limited color usage)
  - Variable colors mode (different colors per turn)
- **AI Opponent**: Intelligent AI using minimax algorithm with alpha-beta pruning
- **Chromatic Number Calculation**: Real-time calculation of graph chromatic number
- **Modern UI**: Beautiful, responsive interface with smooth animations

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. Navigate to the project root directory:
```bash
cd /Users/vedantmoghe/Desktop/ie211
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Flask backend server:
```bash
python app.py
```

The backend will run on `http://localhost:5001`

### Frontend Setup

1. Navigate to the React app directory:
```bash
cd ie211-react-app
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000` and automatically open in your browser.

## How to Play

1. **Select Graph Type**: Choose from Petersen, Wheel, Cycle, or Random graph
2. **Choose Game Mode**: 
   - Player vs Player
   - Player vs AI
3. **Select Game Variant**: Base, Variable Nodes, Resource Limited, or Variable Colors
4. **Set Number of Colors**: Adjust the slider (2-5 colors)
5. **Play**: Click on an uncolored vertex, then select a color
6. **Win Condition**: The player who cannot make a legal move loses

## Game Rules

- Players alternate coloring vertices
- Adjacent vertices cannot have the same color
- The chromatic number χ(G) represents the minimum colors needed
- If available colors < χ(G), the game becomes strategic
- The player who cannot move loses

## Project Structure

```
ie211/
├── app.py                 # Flask backend API server
├── colour.py              # Core game logic and algorithms
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── ie211-react-app/       # React frontend
    ├── src/
    │   ├── App.js         # Main React component
    │   ├── App.css        # Styling
    │   └── index.js       # React entry point
    └── package.json       # Node dependencies
```

## API Endpoints

- `POST /api/health` - Health check
- `POST /api/graph/chromatic-number` - Calculate chromatic number
- `POST /api/game/init` - Initialize new game
- `POST /api/game/move` - Make a move
- `POST /api/game/ai-move` - Get AI's best move
- `POST /api/game/state` - Get current game state
- `POST /api/game/reset` - Reset game session

## Troubleshooting

### Backend not starting
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify port 5001 is not in use (changed from 5000 to avoid macOS AirPlay conflicts)

### Frontend not connecting to backend
- Ensure backend is running on `http://localhost:5001`
- Check browser console for CORS errors
- Verify `API_BASE_URL` in `App.js` matches your backend URL

### Game not responding
- Check browser console for errors
- Verify backend logs for API errors
- Try refreshing the page and starting a new game

## Technologies Used

- **Backend**: Python, Flask, NetworkX, NumPy
- **Frontend**: React, JavaScript, CSS3
- **Algorithms**: Minimax with alpha-beta pruning, DSatur, Greedy coloring

## License

This project is for educational purposes.

