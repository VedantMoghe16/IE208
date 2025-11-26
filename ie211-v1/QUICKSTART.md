# Quick Start Guide

## Running the Application

### Option 1: Manual Start (Recommended for Development)

**Terminal 1 - Backend:**
```bash
cd /Users/vedantmoghe/Desktop/ie211
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd /Users/vedantmoghe/Desktop/ie211/ie211-react-app
npm install
npm start
```

### Option 2: Using Scripts

**Terminal 1:**
```bash
cd /Users/vedantmoghe/Desktop/ie211
./start_backend.sh
```

**Terminal 2:**
```bash
cd /Users/vedantmoghe/Desktop/ie211
./start_frontend.sh
```

## Verification

1. Backend should show: `Running on http://0.0.0.0:5001`
2. Frontend should open automatically at: `http://localhost:3000`
3. Check backend health: Visit `http://localhost:5001/api/health` in browser

## Common Issues

### "Cannot connect to backend"
- Ensure backend is running on port 5001
- Check `http://localhost:5001/api/health` responds with `{"status":"ok"}`

### CORS Errors
- Backend has CORS enabled, but if issues persist, check Flask-CORS is installed

### Port Already in Use
- Backend: Change port in `app.py` (last line) - currently set to 5001 to avoid macOS AirPlay conflicts
- Frontend: React will prompt to use different port

## Testing the Game

1. Select a graph type (Petersen recommended for first try)
2. Choose "Player vs AI" mode
3. Click on an uncolored vertex
4. Select a color
5. Watch AI make its move automatically

Enjoy playing!

