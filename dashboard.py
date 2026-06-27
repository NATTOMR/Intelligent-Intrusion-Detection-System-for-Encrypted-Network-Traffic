"""
IDS Dashboard — Flask Backend
Serves metrics, model results, and result images to the web dashboard.
"""
import os
import json
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, 'results')

# ── Helper ────────────────────────────────────────────────────────────────────
def load_json(filename):
    path = os.path.join(RESULTS, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics')
def api_metrics():
    baseline   = load_json('baseline_metrics.json')
    evaluation = load_json('evaluation_metrics.json')
    gen        = load_json('generalization_metrics.json')
    threshold  = load_json('ae_threshold.json')

    return jsonify({
        'baseline':        baseline,
        'evaluation':      evaluation,
        'generalization':  gen,
        'ae_threshold':    threshold.get('threshold', 0),
    })

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS, filename)

if __name__ == '__main__':
    print("=" * 55)
    print("  IDS Dashboard running at http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)
    app.run(debug=False, port=5000)
