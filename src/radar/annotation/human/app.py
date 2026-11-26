# app.py — the local web server of the annotation tool (Flask). Only translates between
# HTTP and store.py: every route opens a session, calls one store function and answers as JSON.
# The page itself is static/index.html; it talks to the routes below with fetch().

import logging
import socket
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.serving import make_server

from radar.annotation.human import store
from radar.config import setting

STATIC_DIR = Path(__file__).parent / 'static'
HOST = '127.0.0.1'  # this computer only: the tool is never reachable from the network
DEFAULT_PORT = setting('annotation_tool', 'port')


def create_app(session_factory, run_id):
    """Build the Flask app for one human run.

    :param session_factory: called without arguments, returns a session (radar.db.engine.Session;
        tests pass their own)
    :param run_id: the annotation run this server works on; the page cannot choose another
    """
    app = Flask(__name__, static_folder=None)

    @app.get('/')
    def index():
        return send_from_directory(STATIC_DIR, 'index.html')

    @app.get('/api/run')
    def get_run():
        with session_factory() as session:
            annotation_run = store.load_human_run(session, run_id)
            return jsonify(store.run_state(session, annotation_run))

    @app.get('/api/posts/<int:post_id>')
    def get_post(post_id):
        with session_factory() as session:
            annotation_run = store.load_human_run(session, run_id)
            return jsonify(store.get_post(session, annotation_run, post_id))

    @app.put('/api/posts/<int:post_id>')
    def put_post(post_id):
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('expected a JSON object with stance and criteria')
        with session_factory() as session:
            annotation_run = store.load_human_run(session, run_id)
            store.save_annotation(session, annotation_run, post_id, data.get('stance'), data.get('criteria'))
        return jsonify({'saved': post_id})

    # Errors of store.py become JSON answers the page can show; RunLocked is a ValueError
    # too, Flask picks the handler of the most specific class.
    @app.errorhandler(store.RunLocked)
    def run_locked(error):
        return jsonify({'error': str(error)}), 409

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return jsonify({'error': str(error)}), 400

    return app


def port_in_use(port):
    """True if something already answers on the port.

    Asked explicitly because Windows lets a second server listen on a taken port without an
    error; the browser would then show the tool of the other run.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex((HOST, port)) == 0


def serve(session_factory, run_id, port=DEFAULT_PORT, open_browser=True):
    """Run the tool on http://127.0.0.1:<port>/ until Ctrl+C; opens the browser first."""
    app = create_app(session_factory, run_id)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # no log line per request
    if port_in_use(port):
        raise RuntimeError(f'port {port} is in use; if the tool is still running in another terminal, '
                           f'stop it there or choose another port with --port')
    server = make_server(HOST, port, app, threaded=True)
    url = f'http://{HOST}:{port}/'
    print(f'annotation tool for run {run_id}: {url}  (stop with Ctrl+C)')
    if open_browser:
        webbrowser.open(url)  # the server already listens, the request waits until it serves
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
