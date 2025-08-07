from flask import Flask, jsonify
import os
import json

app = Flask(__name__)

@app.route('/forecasts')
def get_forecasts():
    try:
        with open('forecast_results.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run forecasting_engine.py to generate initial JSON
    os.system('python forecasting_engine.py')
    # Start Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
