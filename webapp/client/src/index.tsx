import * as React from "react"
import * as ReactDOM from "react-dom"
import "favico.js"

type ColorT = 'w' | 'u' | 'g' | 'b' | 'r'
type GemT = ColorT | '*'

interface CostT {
  w?: number
  u?: number
  g?: number
  b?: number
  r?: number
}

interface GemsT extends CostT {
  '*'?: number
}

interface CardT {
  color: string
  points: number
  uuid: string
  cost: CostT
  level: string
}

interface CardsT {
  w?: CardT[]
  u?: CardT[]
  g?: CardT[]
  b?: CardT[]
  r?: CardT[]
}

interface NobleT {
  id: number
  points: number
  uuid: string
  requirement: CostT
}

interface PlayerT {
  id: number
  name: string
  uuid: string
  reserved: CardT[]
  nobles: NobleT[]
  cards: CardsT
  gems: GemsT
  score: number
}

interface LogT {
  pid: number
  time: number
  msg: string
}

interface GameT {
  players: PlayerT[]
  cards: { [level: string]: CardT[] }
  decks: { [level: string]: number }
  log: LogT[]
  gems: GemsT
  nobles: NobleT[]
  winner: number | null
  turn: number
  phase?: string
  valid_actions?: {
    buy: string[]
    reserve: string[]
    noble: string[]
    take: string[]
    discard: string[]
  }
}

interface ChatT {
  time: number
  pid: number
  name: string
  msg: string
}

interface GameState extends GameT {
  mode: string
  selectedPlayer: number
  phase: string
  showChat: boolean
  showLog: boolean
  chat: ChatT[]
  chatNotify: boolean
  selectedTokens: string[]
  previewHover: number | null
}

interface ServerResponse {
  error?: string
  state: GameT
  chat: ChatT[]
  status?: number
  result: {
    error?: string
  }
}

let globalShowError = (resp: ServerResponse) => { return false }

(function () {
  let errorTimeout = 0
  const colors = ['b', 'u', 'w', 'g', 'r']
  const gemColors = colors.concat(['*'])
  const levelNames = ['level1', 'level2', 'level3']

  function mapColors(gems: GemsT, game: Game, callback: (color: GemT) => void, symbol: string, uuid: string | number) {
    return gemColors.map((color: GemT) => {
      var cName = color + "chip"
      if (color === '*') cName = "schip"
      var count = gems[color]
      if (uuid === "game" && count === 0) return null;
      
      let isGame = uuid === "game";
      let dragProps = isGame && count > 0 ? {
         draggable: true,
         onDragStart: (e: React.DragEvent) => { e.dataTransfer.setData("text/plain", color); }
      } : {};
      
      let clickProps = (!isGame && count > 0) ? { onClick: () => callback(color as GemT) } : {};
      let cursorStyle = isGame && count > 0 ? 'grab' : (!isGame && count > 0 ? 'pointer' : 'default');

      return (
        <div className={"gem " + cName} key={color + "_colors_" + uuid} {...dragProps} {...clickProps} style={{ cursor: cursorStyle }}>
          <div className="bubble">{count}</div>
          <div className="underlay">{symbol}</div>
        </div>
      );
    });
  }

  function mapNobles(nobles: NobleT[], game: Game) {
    return nobles.map((noble) => {
      return (
        <Noble key={noble.uuid} noble={noble} game={game}/>
      );
    });
  }

  class Card extends React.Component<{ card: CardT, game: Game }, {}> {
    render() {
      const card = this.props.card
      const game = this.props.game
      var buyer = game.buy.bind(game, card.uuid)
      const reserver = (e: React.MouseEvent<HTMLDivElement>) => {
        e.preventDefault();
        game.reserve.bind(game)(card.uuid)
      }
      
      const isMyTurn = game.state.turn === game.props.pid
      const va = game.state.valid_actions
      let canBuy = false
      let canReserve = false
      if (isMyTurn && va) {
        if (va.buy.indexOf(card.uuid) !== -1) canBuy = true
        if (va.reserve.indexOf(card.uuid) !== -1) canReserve = true
      }

      let affordClass = "";
      let pid = game.state.previewHover !== null ? game.state.previewHover : game.props.pid;
      let p = game.state.players.find(x => x.id === pid);
      if (p) {
          let missing = 0;
          for (let c of ['r', 'g', 'b', 'w', 'u']) {
             let colorKey = c as keyof CostT & keyof GemsT & keyof CardsT;
             let cost = card.cost[colorKey] || 0;
             let available = (p.gems[colorKey] || 0) + (p.cards[colorKey] ? p.cards[colorKey].length : 0);
             if (cost > available) missing += (cost - available);
          }
          if ((p.gems['*'] || 0) >= missing) affordClass = " buyable";
          else affordClass = " unbuyable";
      }

      if (card.color) {
        return (
          <div
            className={`card card-${card.color} card-${card.level}${affordClass}`}
            id={card.uuid}
          >
            <div className={`reserve ${!canReserve && isMyTurn && va ? "invalid-btn" : ""}`} onClick={reserver}>
              <img className="floppy" src="client/img/floppy.png" />
            </div>
            <div className="overlay" onClick={canBuy ? buyer : undefined}></div>
            <div className="underlay">
              <div className="header">
                <div className={"color " + card.color + "gem"}>
                </div>
                <div className="points">
                  {card.points > 0 && card.points}
                </div>
              </div>
              <div className="costs">
                {colors.map((color: ColorT) => {
                  if(card.cost[color] > 0) {
                    return (
                      <div
                        key={card.uuid + "_cost_" + color}
                        className={"cost " + color}
                      >
                        {card.cost[color]}
                      </div>
                    )
                  }
                })}
              </div>
            </div>
          </div>
        );
      } else {
        return (
          <div className = {"deck " + card.level}></div>
        )
      }
    }
  }

  class Noble extends React.Component<{ noble: NobleT, game: Game }, {}> {
    render() {
      const noble = this.props.noble
      const game = this.props.game
      const visit = game.noble.bind(game, noble.uuid)

      return (
        <div className="noble" onClick={visit} id={"noble" + noble.id}>
          <div className="side-bar">
            <div className="points">
              {noble.points > 0 && noble.points}
            </div>
            <div className="requirement">
              {colors.map((color: ColorT) => {
                if(noble.requirement[color] > 0) {
                  return (
                    <div
                      key={noble.uuid + "_req_" + color}
                      className={"requires " + color}
                    >
                      {noble.requirement[color]}
                    </div>
                  )
                }
              })}
            </div>
          </div>
        </div>
      );
    }
  }

  interface PlayerProps {
    game: Game
    pid: number
    cards: CardsT
    gems: GemsT
    name: string
    points: number
    nobles: NobleT[]
    reserved: CardT[]
    nreserved: number
    selectedPlayer: number
  }

  class Player extends React.Component<PlayerProps, { editingName: string | null }> {
    state = { editingName: null as string | null }

    editName = (e: React.ChangeEvent<HTMLInputElement>) => {
      this.setState({ editingName: e.target.value })
    }

    focusName = (e: React.FocusEvent<HTMLInputElement>) => {
      e.target.select()
    }

    submitName = () => {
      this.props.game.rename(this.state.editingName);
      this.setState({ editingName: null })
    }

    keypress = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        this.submitName()
      }
    }

    render() {
      const game = this.props.game
      const pid = this.props.pid
      const playerSelector = game.selectPlayer.bind(game, pid)
      const collection: { [color: string]: { cards: number, gems: number } } = {}

      gemColors.map((color: ColorT) => {
        collection[color] = {'cards': 0, 'gems': this.props.gems[color]};
      })

      const set = colors.map((color: ColorT) => {
        var cards = this.props.cards[color].map((card: CardT) => {
          collection[color]['cards'] += 1;
          return (
            <div
              key={pid + "_card_" + card.uuid}
              className="colorSetInner"
            >
              <Card key={card.uuid} card={card} game={game}/>
            </div>
          );
        })
        return (
          <div key={pid + "_set_" + color} className="colorSet" style={{ display: cards.length > 0 ? "inline-block" : "none" }}>
            {cards}
            {cards.length > 0 && <div className="endcap"></div>}
          </div>
        )
      })

      const stats = gemColors.map((color) => {
        let isDiscard = (game.state.turn === pid && game.state.phase === "discard" && collection[color]['gems'] > 0);
        return (
          <div className="statSet" key={"stat" + color}
               onClick={isDiscard ? () => game.act('discard', color) : undefined}
               style={{ cursor: isDiscard ? 'pointer' : 'default' }}>
            <div className={`stat stat${color === '*' ? 'y' : color}`}>{collection[color]['gems'] + (color == '*' ? '' : ' / ' + collection[color]['cards'])}</div>
            {color === '*' ? <React.Fragment /> : <div><img className="labelImg" src="client/img/labels.png" /></div>}
          </div>
        )
      })

      const you = (game.props.pid === pid ? " you selected" : "")
      const youName = (game.props.pid === pid ? " (you)" : "")
      const gems = mapColors(this.props.gems, game, game.discard, 'X', pid)
      const reserved = this.props.reserved ? this.props.reserved.map((card) => {
        return (
          <Card key={card.uuid + "_inner"} card={card} game={game}/>
        );
      }) : []
      const reservedCount = this.props.reserved ? reserved.length : this.props.nreserved
      const nobles = mapNobles(this.props.nobles, game)

      return (
        <div className={"player" + you + (game.state.turn === pid ? " active-turn" : "")}>
          <div className="playerHeader">
            <div className="playerPoints">{this.props.points}</div>
            {this.state.editingName == null ?
              <>
                <div className="playerName" onClick={playerSelector}>{this.props.name}</div>
                {game.props.pid === pid && this.state.editingName == null ?
                  <div className="pencil" onClick={() => this.setState({ editingName: this.props.name })}>✏️</div> : <React.Fragment />
                }
              </> :
              <div className="playerName">
                <input className="nameInput" type="text" value={this.state.editingName} autoFocus={true} onKeyPress={this.keypress} onFocus={this.focusName} onBlur={this.submitName} onChange={this.editName} />
              </div>
            }
            <div className="playerName2">{youName}</div>
            {game.props.pid !== pid && (
              <button
                onMouseEnter={() => game.setState({ previewHover: pid })}
                onMouseLeave={() => game.setState({ previewHover: null })}
                className="preview-btn"
                style={{ marginLeft: '10px', fontSize: '12px', cursor: 'pointer', background: '#ccc', border: '1px solid #999', borderRadius: '4px', padding: '2px 5px' }}
              >
                👁️ Preview
              </button>
            )}
            {game.state.turn === pid &&
              <div className="turnIndicator">&#8592;</div>
            }
          </div>
          <div className="player-details">
            <div className="player-main-column">
              <div className="stats" style={{ marginBottom: "10px" }}>
                <div className="gem-stats">{stats}</div>
              </div>
              <div className="floater">
                <div className="nobles">
                  {nobles}
                </div>
              </div>
            </div>
            <div className="player-reserve-column">
              <div className="reserveArea">
                { reservedCount > 0 &&
                  <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                    <div className="reserveText">
                      reserved
                    </div>
                    <div className="reserveCards">
                      {reserved}
                    </div>
                  </div>
                }
              </div>
            </div>
          </div>
        </div>
      );
    }
  }

  class Level extends React.Component<{ name: string, remaining: number, game: Game, cards: CardT[] }, {}> {
    render() {
      return (
        <div>
          <div className={"deck " + this.props.name}>
            <div className="remaining">
              {this.props.remaining}
            </div>
            <div className="overlay"></div>
            <div className="reserve" onClick={this.props.game.reserve.bind(this.props.game, this.props.name)}>
              <img className="floppy" src="client/img/floppy.png" />
            </div>
          </div>
          <div className={"c_" + this.props.name + " face-up-cards"}>
            <div className="cards-inner">
              {this.props.cards && this.props.cards.map((card) =>
                <Card key={card.uuid} card={card} game={this.props.game}/>
              )}
            </div>
          </div>
        </div>
      );
    }
  }

  class Game extends React.Component<{ gid: string, pid: number, uuid: string }, GameState> {
    state = {
      players: [],
      gems: {},
      cards: {},
      chat: [],
      decks: {},
      nobles: [],
      log: [],
      turn: -1,
      winner: null,
      mode: "normal",
      error: null,
      selectedPlayer: -1,
      phase: "pregame",
      showChat: false,
      showLog: false,
      chatNotify: false,
      selectedTokens: [],
      previewHover: null,
    } as GameState

    isMyTurn = (turn: number) => {
      return (turn == this.props.pid);
    }

    updateState = (r: ServerResponse) => {
      if (r.state) {
        var myTurn = this.isMyTurn(r.state.turn);
        if (!myTurn) this.setState({mode: "waiting"});
        else {
          if (this.state.mode == "waiting") {
            favicon.badge('!' as any);
            (document.getElementById("notify") as HTMLAudioElement).play();
          }
          this.setState({mode: "normal"});
        }

        if (this.state.selectedPlayer == -1 && this.props.pid < 4) {
          this.setState({selectedPlayer: this.props.pid});
        }

        this.setState({
          log: r.state.log,
          cards: r.state.cards,
          decks: r.state.decks,
          players: r.state.players,
          gems: r.state.gems,
          nobles: r.state.nobles,
          turn: r.state.turn,
          phase: r.state.phase,
          valid_actions: r.state.valid_actions,
        });

        if (r.state.winner !== null && this.state.phase != "postgame") {
          alert(r.state.players[r.state.winner].name + " wins!");
          this.setState({phase: "postgame"});
          window.location.href = "/";
        }

        if (r.chat) {
          var chat = this.state.chat;
          if (chat && chat[chat.length - 1] && r.chat[r.chat.length -1]) {
            var lastLocalChat = chat[chat.length - 1];
            var lastRemoteChat = r.chat[r.chat.length - 1];
            if (lastLocalChat.msg != lastRemoteChat.msg && lastRemoteChat.pid != this.props.pid) {
              favicon.badge('.' as any);
              (document.getElementById("notify") as HTMLAudioElement).play();
              if (!this.state.showChat) this.setState({ chatNotify: true })
            }
          }
          this.setState({ chat: r.chat });
        }

        for (const scroller of document.getElementsByClassName("scroller")) {
          scroller.scrollTop = scroller.scrollHeight
        }
      }
    }

    loginArgs = () => {
      return '?pid=' + this.props.pid + '&uuid=' + this.props.uuid
    }

    untake = (index: number) => {
      let selected = [...this.state.selectedTokens];
      selected.splice(index, 1);
      this.setState({ selectedTokens: selected });
    }

    take = (color: string) => {
      let selected = [...this.state.selectedTokens];
      let counts: { [key: string]: number } = {};
      selected.forEach(c => counts[c] = (counts[c]||0) + 1);
      
      let isDuplicateSelection = selected.length === 1 && selected[0] === color;
      
      if (selected.length >= 3) return; // Max 3 total
      if (selected.length === 2 && selected[0] === selected[1]) return; // Cannot exceed 2 if double taking
      if (selected.length >= 1 && selected[0] !== color && Object.keys(counts).some(k => counts[k] === 2)) return; // Cannot take another if already took double
      if (selected.length === 2 && selected[0] !== selected[1] && selected.indexOf(color) !== -1) return; // Cannot take double if already took different
      
      let bankCount = this.state.gems[color as keyof GemsT] || 0;
      let selectedCount = counts[color] || 0;
      if (bankCount - selectedCount <= 0) return;
      if (isDuplicateSelection && bankCount < 4) return; // Cannot double take if bank has < 4 total initially
      
      selected.push(color);
      this.setState({ selectedTokens: selected });
    }

    discard = (color: string) => {
      if (confirm("Are you sure you want to discard a gem?")) {
        this.act('discard', color)
      }
    }

    selectPlayer = (player: number) => {
      this.setState({ selectedPlayer: player })
    }

    buy = (uuid: string) => {
      this.act('buy', uuid)
    }

    reserve = (uuid: string) => {
      this.act('reserve', uuid)
    }

    noble = (uuid: string) => {
      this.act('noble_visit', uuid);
    }

    rename = async (name: string) => {
      const resp = await fetch(`/rename/${this.props.gid}/${name}${this.loginArgs()}`, { method: 'POST' })
      const json = await resp.json()
      globalShowError(json)
    }

    act = async (action: string, target: string) => {
      const resp = await fetch('/game/' + this.props.gid + '/' + action + '/' + target + this.loginArgs(), { method: 'POST' })
      const json = await resp.json()
      if (!globalShowError(json)) this.updateState(json)
    }

    nextTurn = async () => {
      const resp = await fetch('/game/' + this.props.gid + '/next' + this.loginArgs(), { method: 'POST' })
      const json = await resp.json()
      if (!globalShowError(json)) this.updateState(json)
    }

    poll = async () => {
      const resp = await fetch('/poll/' + this.props.gid + this.loginArgs())
      const json = await resp.json()
      if (!globalShowError(json)) {
        this.updateState(json)
        let delay = 0;
        if (json.state && json.state.turn !== this.props.pid) {
           delay = 1000;
        }
        setTimeout(this.poll, delay);
      }
    }

    stat = async () => {
      const resp = await fetch(`/stat/${this.props.gid}${this.loginArgs()}`)
      const json = await resp.json()
      if (!globalShowError(json)) this.updateState(json)
    }

    componentDidMount() {
      this.stat()
      this.poll()
    }

    chat = async (e: React.KeyboardEvent<HTMLInputElement>) => {
      const chatBox = document.getElementById("chat-inner") as HTMLInputElement
      if (e.which == 13) {
        const resp = await fetch('/game/' + this.props.gid + '/chat' + this.loginArgs(), {
          method: 'POST',
          body: JSON.stringify({ msg: chatBox.value }),
        })
        chatBox.value = ""
        const json = await resp.json()
        if (!globalShowError(json)) this.updateState(json)
      }
    }

    render() {
      var players = this.state.players.map((player) => {
        return (
          <Player
            selectedPlayer={this.state.selectedPlayer}
            key={player.uuid}
            pid={player.id}
            name={player.name}
            points={player.score}
            game={this}
            cards={player.cards}
            nobles={player.nobles}
            gems={player.gems}
            reserved={player.reserved}
            nreserved={player.reserved.length}
          />
        );
      });
      var displayGems = { ...this.state.gems };
      this.state.selectedTokens.forEach(c => {
         let key = c as keyof GemsT;
         if (displayGems[key] !== undefined && displayGems[key]! > 0) displayGems[key]!--;
      });
      var bankGems = mapColors(displayGems, this, this.take, '', 'game');
      
      var pendingGems = this.state.selectedTokens.map((color, idx) => {
         var cName = color + "chip";
         return (
           <div className={"gem " + cName} key={"pending_" + idx}
             draggable={true}
             onDragStart={(e: React.DragEvent) => {
               e.dataTransfer.setData("text/plain", idx.toString());
               e.dataTransfer.setData("action", "untake");
             }}
             style={{ cursor: 'grab' }}>
             <div className="underlay"></div>
           </div>
         );
      });
      var nobles = mapNobles(this.state.nobles, this);
      var log = [...this.state.log].reverse().map((logLine, i) => {
        var parts = logLine.msg.split(/(\[[rgubw*]\])/g);
        var msgElements = parts.map((part, j) => {
           if (part.match(/^\[[rgubw*]\]$/)) {
             var c = part[1]; 
             return <span key={j} className={`log-gem log-gem-${c}`}>{c}</span>
           }
           return <span key={j}>{part}</span>;
        });
        return (
          <div key={"log-line-" + i} className="line">
            <span className="msg">{msgElements}</span>
          </div>
        );
      });
      var chat = this.state.chat.map((chatLine, i) => {
        return (
          <div key={"chat-line-" + i} className="line">
            <span className={`name name${chatLine.pid}`}>{chatLine.name + ": "}</span>
            <span className="msg">{chatLine.msg}</span>
          </div>
        );
      });
      var levels = levelNames.map((level) => {
        return (
          <Level
            key = {level}
            game = {this}
            name = {level}
            cards = {this.state.cards[level]}
            remaining = {this.state.decks[level]}
          />
        )
      });
      var actionList: string[] = [];
      if (this.state.valid_actions) {
         if (this.state.valid_actions.buy && this.state.valid_actions.buy.length > 0) actionList.push("Buy");
         if (this.state.valid_actions.reserve && this.state.valid_actions.reserve.length > 0) actionList.push("Reserve");
         if (this.state.valid_actions.take && this.state.valid_actions.take.length > 0) actionList.push("Take Tokens");
         if (this.state.valid_actions.noble && this.state.valid_actions.noble.length > 0) actionList.push("Pick Noble");
         if (this.state.valid_actions.discard && this.state.valid_actions.discard.length > 0) actionList.push("Discard Tokens");
      }
      var phaseStr = this.state.phase ? this.state.phase.toUpperCase() : "PREGAME";

      return (
        <div>
          <div className="phase-header">
            <strong>Phase: {phaseStr}</strong>
            {this.isMyTurn(this.state.turn) && actionList.length > 0 &&
              <span> | Valid Actions: {actionList.join(', ')}</span>
            }
            {this.isMyTurn(this.state.turn) && this.state.phase === "discard" &&
              <span style={{ color: 'red', marginLeft: '10px', fontWeight: 'bold' }}>Click one of your gems below to discard!</span>
            }
          </div>
          <div id="game-board" className="preview-active">
            <div id="common-area">
              <div id="noble-area" className="split">
                {nobles}
              </div>
              <div id="level-area" className="split">
                {levels}
              </div>
              <div className="reserve-info">
                <div className="reserve-info-inner">
                  <div>Click on card to buy, click on </div><div><img className="floppy" src="client/img/floppy.png" /></div><div> to reserve.</div>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '20px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                  <div id="gem-area" className="you"
                       onDragOver={(e) => e.preventDefault()}
                       onDrop={(e) => {
                           e.preventDefault();
                           if (e.dataTransfer.getData("action") === "untake") {
                               this.untake(parseInt(e.dataTransfer.getData("text/plain")));
                           }
                       }}>
                    {bankGems}
                  </div>
                  <div id="pending-area" style={{ display: 'flex', minHeight: '80px', minWidth: '300px', border: '2px dashed #999', borderRadius: '8px', padding: '10px', gap: '10px', alignItems: 'center', justifyContent: 'center' }}
                       onDragOver={(e) => e.preventDefault()}
                       onDrop={(e) => {
                           e.preventDefault();
                           let color = e.dataTransfer.getData("text/plain");
                           let action = e.dataTransfer.getData("action");
                           if (!action && color) { 
                               this.take(color);
                           }
                       }}>
                    {pendingGems.length === 0 ? <span style={{color: '#999'}}>Drag tokens here to take</span> : pendingGems}
                  </div>
                </div>
                {this.isMyTurn(this.state.turn) && this.state.phase === "main" && (
                  (() => {
                    let isMax = false;
                    let selected = this.state.selectedTokens;
                    let myGemsCount = 0;
                    if (this.props.pid >= 0) {
                      let myGems = this.state.players.find(p => p.id === this.props.pid)?.gems;
                      myGemsCount = (myGems?.r||0) + (myGems?.g||0) + (myGems?.b||0) + (myGems?.u||0) + (myGems?.w||0);
                    }
                    let spaceLeft = 10 - myGemsCount;
                    if (selected.length === 2 && selected[0] === selected[1]) isMax = true;
                    else if (selected.length === 3) isMax = true;
                    else if (selected.length >= spaceLeft) isMax = true;
                    else if (selected.length > 0) {
                        let canPickMore = (['w','u','g','b','r'] as (keyof GemsT)[]).some(c => (this.state.gems[c]||0) > 0 && selected.indexOf(c as string) === -1);
                        if (!canPickMore) isMax = true;
                    }
                    return (
                      <button onClick={() => { this.act('take', selected.join(',')); this.setState({selectedTokens: []}) }}
                        disabled={!isMax}
                        style={{ padding: '10px 20px', cursor: isMax ? 'pointer' : 'not-allowed', background: '#F2C94C', border: 'none', borderRadius: '4px', fontWeight: 'bold', opacity: isMax ? 1 : 0.5 }}>
                        Confirm Tokens
                      </button>
                    )
                  })()
                )}
              </div>
            </div>
            <div id="player-area">
              {players}
            </div>
          </div>
          <div id="log-box" style={{ bottom: this.state.showLog ? -4 : -514 }}>
            <div className="title" onClick={() => this.setState({ showLog: !this.state.showLog })}>::Log</div>
            <div className="scroller">
              {log}
            </div>
          </div>
          <div id="chat-box" onClick={() => this.setState({ chatNotify: false })} style={{ bottom: this.state.showChat ? -4 : -314 }}>
            <div className={`title${ this.state.chatNotify ? " blinking" : ""}`} onClick={() => this.setState({ showChat: !this.state.showChat })}>::Chat</div>
            <div className="scroller">
              {chat}
            </div>
            <div id="chat">
              <span id="prompt">&gt;</span>
              <input id="chat-inner" type="text" onKeyPress={this.chat}></input>
            </div>
          </div>
          {this.state.turn >= 0 && this.props.pid >= 0 && this.props.pid < 4 &&
            <button id={`pass-turn`} onClick={this.nextTurn} style={{ opacity: this.isMyTurn(this.state.turn) ? 1 : 0.3 }}>Pass turn</button>
          }
        </div>
      );
    }
  }

  const ErrorMsg = (props: { error: string | null, opacity: number }) => {
    return <div className="error-box" style={{ opacity: props.opacity }}>
      <div className="error-box-inner">{props.error}</div>
    </div>
  }

  interface GameCreatorState {
    startKey: string | null
    loading: boolean
    lobby: boolean
    gameName: string
    joined: boolean
    pid: number
    uuid: string
    gid: string
    error: string | null
    errorOpacity: number
    numAIs: number
    inferenceEngine: string
    checkpoint: string
    availableModels: string[]
  }

  class GameCreator extends React.PureComponent<{}, GameCreatorState> {
    state = {
      startKey: null,
      loading: true,
      lobby: false,
      joined: false,
      pid: -1,
      uuid: '',
      gid: '',
      gameName: '',
      errorOpacity: 0,
      error: null,
      numAIs: 0,
      inferenceEngine: 'agilerl',
      checkpoint: '',
      availableModels: [],
    } as GameCreatorState

    creating = false

    join = async (game: string, act: string) => {
      const session = this.readSession()
      if (session[game] && !session[game].loading && (session[game].joined || act === 'spectate')) {
        this.setState(session[game]);
        this.save();
        return;
      }

      const resp = await fetch(`/${act}/${game}`, { method: 'POST' })
      const json = await resp.json()
      if (json.status === 404) {
        this.createGame()
        return
      }

      if (this.showError(json)) return

      this.setState({
        joined: (act === 'join'),
        loading: false,
        pid: json.id,
        uuid: json.uuid,
        gid: game,
      })
      this.save()
    }

    showError = (resp: ServerResponse): boolean => {
      let msg: string | null = null
      if (!resp) {
        msg = "Request failed";
        return true
      }
      else if (!resp.error && (!resp.result || !resp.result.error)) return false;
      else msg = resp.error || resp.result.error;

      if (resp.status === 404) {
        this.clear()
        this.setState({ loading: true, joined: false, pid: -1, uuid: '' })
        this.join(this.state.gid, 'spectate')
        return true
      }

      this.setState({ error: msg, errorOpacity: 1 })

      clearTimeout(errorTimeout);
      errorTimeout = setTimeout(() => {
        this.setState({ errorOpacity: 0 })
      }, 4000)

      return true
    }

    createGame = async () => {
      if (this.creating) return
      this.creating = true
      const gameName = this.state.gid === '' ? this.state.gameName : this.state.gid
      const payload = {
        numAIs: this.state.numAIs,
        checkpoint: this.state.checkpoint,
        inferenceEngine: this.state.inferenceEngine || 'agilerl'
      }
      const resp = await fetch(`/create/${gameName}`, { 
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })
      const json = await resp.json()

      if (this.showError(json)) return

      history.replaceState(null, "Splendor", `/${json.game}`)
      this.join(json.game, 'join')
      this.setState({ startKey: json.start, loading: true, lobby: false })
    }

    readSession = () => {
      var session = window.localStorage.getItem('splendor');
      if (session === null) return {};
      return JSON.parse(session);
    }

    componentDidMount() {
      globalShowError = this.showError

      if (window.location.pathname === "/") {
        fetch(`/models`).then(async (resp) => {
          const json = await resp.json()
          this.setState({ availableModels: json.models || [] })
        })

        fetch(`/suggest`).then(async (resp) => {
          const json = await resp.json()
          this.setState({
            lobby: true,
            gameName: json.result.game,
            loading: false,
          })
        })
        return
      }

      const gid = window.location.pathname.substring(1)
      var session = this.readSession()
      this.setState({
        ...session[gid],
        gid,
      })
      if (this.state.loading) {
        this.join(gid, 'spectate')
      }
    }

    save = () => {
      setTimeout(() => {
        this.saveRaw(this.state)
      }, 100)
    }

    clear = () => {
      this.saveRaw(null)
    }

    saveRaw = (state: any) => {
      const session = this.readSession();
      if (state === null) delete session[this.state.gid]
      else session[this.state.gid] = state
      window.localStorage['splendor'] = JSON.stringify(session);
    }

    startGame = async () => {
      const resp = await fetch(`/start/${this.state.gid}/${this.state.startKey}`, { method: "POST" })
      const json = await resp.json()

      if (!this.showError(json)) {
        this.setState({ startKey: null });
        this.save();
      }
    }

    nameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      this.setState({ gameName: e.target.value })
    }

    keyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") this.createGame()
    }

    render() {
      if (this.state.loading) {
        return <div id="game">
          <ErrorMsg error={this.state.error} opacity={this.state.errorOpacity} />
        </div>
      }

      if (this.state.lobby) {
        return <div className="lobby">
          <div className="main-title">Splendor</div>
          <div className="desc">Play Splendor online with others. Enter a game name or use the suggested game name to start a game.</div>
          <div className="name" style={{display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'center'}}>
            <input className="game-name" type="text" onChange={this.nameChange} onKeyPress={this.keyPress} value={this.state.gameName} style={{marginBottom: '5px'}} />
            <div>
              <label style={{marginRight: '10px'}}>AIs: </label>
              <select value={this.state.numAIs} onChange={(e) => this.setState({numAIs: parseInt(e.target.value)})}>
                <option value={0}>0</option>
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </div>
            <div>
              <label style={{marginRight: '10px'}}>Inference Engine: </label>
              <select value={this.state.inferenceEngine} onChange={(e) => this.setState({inferenceEngine: e.target.value})}>
                <option value="sb3">SB3 Contrib / MaskablePPO</option>
                <option value="agilerl">AgileRL</option>
              </select>
            </div>
            <div>
              <label style={{marginRight: '10px'}}>AI Checkpoint: </label>
              <input type="text" list="ai-models" value={this.state.checkpoint} onChange={(e) => this.setState({checkpoint: e.target.value})} style={{width: '250px'}} />
              <datalist id="ai-models">
                {this.state.availableModels.map(m => <option key={m} value={m} />)}
              </datalist>
            </div>
            <button onClick={this.createGame} className="create-game" style={{marginTop: '10px'}}>Create Game</button>
          </div>
          <ErrorMsg error={this.state.error} opacity={this.state.errorOpacity} />
        </div>
      }

      return (
        <div id="game">
          <div id="game-title">
            <div className="link">
              Share this link with friends to join in or watch: <a href=".">{`${document.location.href}`}</a>
            </div>
            <div className="buttons">
              {!this.state.joined &&
                <button className="start-game" onClick={() => this.join(this.state.gid, 'join')}>Join Game</button>
              }
              {this.state.startKey && this.state.pid == 0 &&
                <button className="start-game" onClick={this.startGame}>Start Game</button>
              }
            </div>
          </div>
          {this.state.pid >= 0 && this.state.gid && this.state.uuid &&
            <Game key={this.state.pid} gid={this.state.gid} pid={this.state.pid} uuid={this.state.uuid} />
          }
          <ErrorMsg error={this.state.error} opacity={this.state.errorOpacity} />
        </div>
      )
    }
  }

  const favicon=new Favico({
    position : 'up'
  })

  document.onclick = () => {
    favicon.badge('' as any)
  }

  ReactDOM.render(
    <div>
      <GameCreator />
    </div>,
    document.getElementById('content')
  )
})()
