# Port Change Summary

## Issue
Flask server was unable to run on port 5000 (common issue on macOS due to AirPlay Receiver using that port).

## Solution
Changed Flask server port from **5000** to **5001** throughout the application.

## Files Updated

### Backend
- ✅ `app.py` - Changed port from 5000 to 5001 (line 287)

### Frontend
- ✅ `ie211-react-app/src/App.js` - Updated API_BASE_URL to use port 5001
- ✅ `ie211-react-app/src/App.js` - Updated error messages to reference port 5001

### Documentation
- ✅ `README.md` - Updated all port references to 5001
- ✅ `QUICKSTART.md` - Updated verification and troubleshooting sections

## New Configuration

**Backend URL:** `http://localhost:5001`  
**API Base URL:** `http://localhost:5001/api`  
**Health Check:** `http://localhost:5001/api/health`

## Testing

To verify the changes work:

1. Start the backend:
   ```bash
   python app.py
   ```
   Should show: `Running on http://0.0.0.0:5001`

2. Test health endpoint:
   ```bash
   curl http://localhost:5001/api/health
   ```
   Should return: `{"status":"ok"}`

3. Start the frontend:
   ```bash
   cd ie211-react-app
   npm start
   ```

4. The frontend should now successfully connect to the backend on port 5001.

## Notes
- Port 5001 was chosen to avoid conflicts with common macOS services
- All error messages and documentation have been updated accordingly
- The change is backward compatible - just requires restarting both servers

