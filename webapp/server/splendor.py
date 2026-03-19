from flask import Flask, Response, request, send_from_directory
from functools import wraps
from env_adapter import SplendorEnvAdapter
from sb3_contrib import MaskablePPO
import signal
import random
import time
import json
import os
import platform
import sys
import uuid
import glob

app = Flask(__name__)
game_map = {}
POLL_INTERVAL = 0.4
basedir = os.path.dirname(os.path.abspath(__file__))
webapp_dir = os.path.dirname(basedir)
splendor_dir = os.path.dirname(webapp_dir)
client_dir = os.path.join(webapp_dir, 'client')
words = []
num_created = 0

def json_response(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        r = f(*args, **kwargs)
        return Response(json.dumps(r), content_type='application/json; charset=utf-8')
    return decorated_function

class PreGame:
    def __init__(self):
        self.state = 'pregame'
        self.players = []
        self.pids = []
        self.num_players = 0
        
    def dict(self, pid=None):
        return {
            'state': self.state,
            'players': [{'id': p['id'], 'name': p['name'], 'uuid': p['uuid'], 'reserved': [], 'nobles': [], 'cards': {'w':[],'u':[],'g':[],'b':[],'r':[]}, 'gems': {'w':0,'u':0,'g':0,'b':0,'r':0,'*':0}, 'score': 0} for p in self.players],
            'cards': {'level1': [], 'level2': [], 'level3': []},
            'decks': {'level1': 0, 'level2': 0, 'level3': 0},
            'log': [],
            'gems': {'w':0,'u':0,'g':0,'b':0,'r':0,'*':0},
            'nobles': [],
            'winner': None,
            'turn': -1
        }
        
    def add_player(self, name):
        pid = self.num_players
        uid = uuid.uuid4().hex
        self.players.append({'id': pid, 'name': name, 'uuid': uid})
        self.pids.append(pid)
        self.num_players += 1
        return pid, uid
        
    def rename_player(self, pid, name):
        for p in self.players:
            if p['id'] == pid:
                p['name'] = name

class GameManager(object):
    def __init__(self, name, num_ais=0, checkpoint="splendor_ppo_mask.zip"):
        global game_map

        self.uuid = name
        self.starter = uuid.uuid4().hex
        game_map[self.uuid] = self
        self.game = PreGame()
        self.changed = {}
        self.chats = []
        self.ended = {}
        self.created = time.time()
        self.started = False
        
        self.player_uuids = {}
        
        self.num_ais = num_ais
        self.checkpoint = checkpoint
        self.ai_model = None

    def check_ai_turn(self):
        if not self.started:
            return
        if self.game.env.terminations[self.game.env.agents[0]] or self.game.env.truncations[self.game.env.agents[0]]:
            return
            
        active_idx = self.game.env.current_player
        active_name = getattr(self.game, f"player_name_{active_idx}", "")
        
        if active_name.startswith("AI Model"):
            # Execute AI action with matching flat observation format
            raw_obs = self.game.env.observe(f"player_{active_idx}")
            flat_obs = {k: v for k, v in raw_obs["observation"].items()}
            flat_obs["action_mask"] = raw_obs["action_mask"]
            
            action, _states = self.ai_model.predict(flat_obs, action_masks=flat_obs["action_mask"], deterministic=True)
            self.game._execute_action_idx(action.item())
            self.has_changed()

    def dict(self):
        return {
            'uuid': self.uuid,
            'n_players': self.game.num_players,
            'in_progress': self.game.state != 'pregame',
        }

    def private_dict(self):
       return {
            'uuid': self.uuid,
            'starter': self.starter,
            'game': self.game.private_dict(),
            'changed': self.changed,
            'chats': self.chats,
            'ended': self.ended,
            'created': self.created,
            'started': self.started
        }

    def poll(self, pid):
        global game_map

        while not self.changed[pid]:
            time.sleep(POLL_INTERVAL)
            yield " "

        if self.ended.get(pid):
            del self.ended[pid]
            if not self.ended:
                del game_map[self.uuid]

        self.changed[pid] = False
        yield json.dumps({'state': self.game.dict(pid), 'result': {}, 'chat': self.chats})

    def num_players(self):
        return self.game.num_players

    def chat(self, pid, msg):
        name = self.game.players[pid].name
        self.chats.append({
            'time': time.time(),
            'pid': pid,
            'name': name,
            'msg': msg,
        })
        self.has_changed()
        return {'state': self.game.dict(pid), 'result': {}, 'chat': self.chats}

    def has_changed(self):
        global game_map

        if hasattr(self.game, 'env') and (self.game.env.terminations[self.game.env.agents[0]] or self.game.env.truncations[self.game.env.agents[0]]):
            for pid in self.ended:
                self.ended[pid] = True
        for p in self.changed:
            self.changed[p] = True

    def join_game(self):
        if self.game.num_players >= 4:
            return {'error': 'Already at max players'}
        if getattr(self.game, 'state', None) != 'pregame':
            return {'error': 'Game already in progress'}

        pid, uuid = self.game.add_player("Player {}".format(self.game.num_players + 1))
        self.player_uuids[pid] = uuid
        self.changed[pid] = False
        self.ended[pid] = False
        self.has_changed()

        return {'id': pid, 'uuid': uuid}

    def spectate_game(self):
        return {'error': 'Spectating is not currently supported correctly with env_adapter'}

    def start_game(self):
        if getattr(self.game, 'state', None) != 'pregame':
            return {'error': 'Game already started'}
            
        for i in range(self.num_ais):
            if self.game.num_players >= 4:
                break
            self.game.add_player(f"AI Model {i+1}")
            
        if self.game.num_players < 2:
            return {'error': 'Need at least 2 players to start'}
            
        if self.num_ais > 0:
            checkpoint_path = os.path.join(splendor_dir, self.checkpoint)
            if not os.path.exists(checkpoint_path):
                return {'error': f'Checkpoint {self.checkpoint} not found at {checkpoint_path}'}
            self.ai_model = MaskablePPO.load(checkpoint_path)
            
        old_game = self.game
        self.game = SplendorEnvAdapter(old_game.num_players)
        for p in old_game.players:
            setattr(self.game, f"player_name_{p['id']}", p['name'])
            setattr(self.game, f"player_uuid_{p['id']}", p['uuid'])
        self.game.pids = old_game.pids
        
        self.started = True
        self.has_changed()
        self.check_ai_turn()
        return {}

def validate_player(game):
    global game_map

    if game not in game_map:
        return None, None

    pid = request.args.get('pid')
    uuid = request.args.get('uuid')

    try:
        pid = int(pid)
    except ValueError:
        return None, None

    game_manager = game_map[game]
    if pid not in game_manager.player_uuids:
        return None, None
    if game_manager.player_uuids[pid] != uuid:
        return None, None
    return pid, game_manager

@app.route('/create/<game>', methods=['POST'])
@json_response
def create_game(game):
    global num_created

    if game in game_map:
        return {'result': {'error': 'Game already exists, try another name'}}
        
    payload = request.get_json(silent=True) or {}
    num_ais = int(payload.get('numAIs', 0))
    checkpoint = payload.get('checkpoint', 'splendor_ppo_mask.zip')
    
    new_game = GameManager(game, num_ais=num_ais, checkpoint=checkpoint)
    num_created += 1
    return {'game': new_game.uuid, 'start': new_game.starter, 'state': new_game.game.dict()}

@app.route('/join/<game>', methods=['POST'])
@json_response
def join_game(game):
    global game_map
    if game not in game_map:
        return {'error': 'No such game'}
    return game_map[game].join_game()

@app.route('/spectate/<game>', methods=['POST'])
@json_response
def spectate_game(game):
    global game_map
    if game not in game_map:
        return {'error': 'No such game', 'status': 404}
    return game_map[game].spectate_game()

@app.route('/start/<game>/<starter>', methods=['POST'])
@json_response
def start_game(game, starter):
    global game_map
    if game not in game_map:
        return {'error': 'No such game'}
    if game_map[game].starter != starter:
        return {'error': 'Incorrect starter key'}
    return game_map[game].start_game()

@app.route('/suggest', methods=['GET'])
@json_response
def suggest_game():
    global game_map, words
    while True:
        idx = random.choice(range(len(words)))
        suggested_name = f"{words[idx]}-{random.randint(1000, 9999)}"
        if suggested_name not in game_map:
            return {'result': {'game': suggested_name}}

@app.route('/game/<game>/next', methods=['POST'])
@json_response
def next(game):
    pid, game_manager = validate_player(game)
    if game_manager is None:
        return {'error': 'Invalid game / pid / uuid'}
    game = game_manager.game
    if not game_manager.started:
        return {'error': 'Game not started yet'}
    if pid != game.env.current_player:
        return {'error': 'Not your turn'}
    result = game.next()
    if result == {}:
        game_manager.has_changed()
    return {'state': game.dict(pid), 'result': result}

@app.route('/game/<game>/chat', methods=['POST'])
@json_response
def chat(game):
    pid, game_manager = validate_player(game)
    if game_manager is None:
        return {'error': 'Invalid game / pid / uuid'}
    payload = {}
    if request.data:
        payload = request.get_json(force=True)
    if not payload.get('msg'):
        return {'error': 'Need "msg" parameter'}
    if not isinstance(payload['msg'], str) and not isinstance(payload['msg'], unicode):
        return {'error': 'msg parameter must be string'}
    return game_manager.chat(pid, payload['msg'])

@app.route('/game/<game>/<action>/<target>', methods=['POST'])
@json_response
def act(game, action, target):
    pid, game_manager = validate_player(game)
    if game_manager is None:
        return {'error': 'Invalid game / pid / uuid'}
    game = game_manager.game
    if not game_manager.started:
        return {'error': 'Game not started yet'}
    if pid != game.env.current_player:
        return {'error': 'Not your turn'}

    if action == 'take':
        for c in target.split(','):
            result = game.take(c)
            if 'error' in result:
                break
    elif action == 'buy':
        result = game.buy(target)
    elif action == 'reserve':
        result = game.reserve(target)
    elif action == 'discard':
        result = game.discard(target)
    elif action == 'noble_visit':
        result = game.noble_visit(target)
    else:
        return {'error': "{0} is not a valid action".format(action)}

    if result == {}:
        game_manager.has_changed()
    return {'state': game.dict(pid), 'result': result}

@app.route('/list', methods=['GET'])
@json_response
def list_games():
    delete_games = []
    for k, v in game_map.items():
        if not v.started and time.time() - v.created > 600:
            delete_games.append(k)
        elif v.started and time.time() - v.created > 24*60*60:
            delete_games.append(k)
    for game in delete_games:
        del game_map[game]
    return {'games': [x.dict() for x in game_map.values()]}

@app.route('/models', methods=['GET'])
@json_response
def list_models():
    zip_files = glob.glob(os.path.join(splendor_dir, '**', '*.zip'), recursive=True)
    
    models = []
    for f in zip_files:
        rel_path = os.path.relpath(f, splendor_dir).replace('\\', '/')
        models.append(rel_path)
        
    return {'models': models}

@app.route('/rename/<game>/<name>', methods=['POST'])
@json_response
def rename_player(game, name):
    pid, game_manager = validate_player(game)
    if game_manager is None:
        return {'error': 'Invalid game / pid / uuid'}
    game_manager.game.rename_player(pid, name)
    game_manager.has_changed()
    return {'result': {'status': 'ok'}}

@app.route('/stat/<game>', methods=['GET'])
@json_response
def stat_game(game):
    pid, game_manager = validate_player(game)
    if game_manager is None:
        return {'error': 'Invalid game / pid / uuid', 'status': 404}
    return {'state': game_manager.game.dict(pid), 'chat': game_manager.chats}

@app.route('/poll/<game>', methods=['GET'])
def poll_game(game):
    pid, game_manager = validate_player(game)
    if game_manager is None:
        return Response(json.dumps({'error': 'Invalid game / pid / uuid', 'status': 404}),
                        content_type='application/json',
                        status=404)
    if getattr(game_manager, 'started', False):
        game_manager.check_ai_turn()
    return Response(game_manager.poll(pid), content_type='application/json')

@app.route('/')
def index():
    return static_proxy('index.html')

@app.route('/favicon.ico')
def favicon():
    return static_proxy('favicon.ico')

@app.route('/client/<path:filename>')
def static_proxy(filename):
    return send_from_directory(client_dir, filename)

@app.route('/<game>')
def existing_game(game):
    return static_proxy('index.html')

@app.route('/stats')
@json_response
def get_stats():
    games = game_map.keys()
    num_games = len(games)
    return {'games': games, 'num_games': num_games, 'num_created': num_created}

def save_and_exit(number, frame):
    sys.exit()

if __name__ == '__main__':
    with open(os.path.join(basedir, 'words.txt')) as f:
        words = f.read().split('\n')[:-1]
        random.shuffle(words)

    try:
        if os.path.exists(os.path.join(basedir, 'save.json')):
            os.remove(os.path.join(basedir, 'save.json'))
    except IOError:
        pass

    if platform.system() != 'Windows':
        signal.signal(signal.SIGHUP, save_and_exit)
    # These usually work on both, but Windows only supports a limited set
    signal.signal(signal.SIGINT, save_and_exit)  # Handles Ctrl+C
    signal.signal(signal.SIGTERM, save_and_exit) # Handles termination

    app.run(host='127.0.0.1', port=8000, threaded=True)
