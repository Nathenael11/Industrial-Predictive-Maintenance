"""
Run script for Predictive Maintenance Flask Web App
"""

from app import app

if __name__ == '__main__':
    print("Starting Elevvo Predictive Maintenance Web Application on http://localhost:5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
