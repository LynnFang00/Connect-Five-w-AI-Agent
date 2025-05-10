# Connect‑Five w/ AI Agent 🕹️🤖

A modern, pygame‑powered take on **Gomoku / Five‑in‑a‑Row** with an optional AI opponent implemented in pure Python.

| Feature                                                            | Status |
| ------------------------------------------------------------------ | ------ |
| 15 × 15 board with polished wood theme                             | ✅     |
| Human 🆚 Human or Human 🆚 AI game modes                           | ✅     |
| Undo / Restart / Quit sidebar buttons                              | ✅     |
| Winner detection and on‑screen messages                            | ✅     |
| AI: pattern‑scoring + α‑β minimax (depth 2 default, ≈10 plies)     | ✅     |
| Easy to extend with stronger heuristics                            | 🚧     |
| Fully self‑contained package (`connect_five`)                      | ✅     |

---

## Screenshots

| Human vs AI (White wins)                       | Mode dialog                                 |
| ---------------------------------------------- | ------------------------------------------- |
|![image](https://github.com/user-attachments/assets/9a996f0a-da44-4d2f-8925-73b4b32a6fef)
 | ![image](https://github.com/user-attachments/assets/ad2029d6-2831-448a-9904-ed7474306f6f)
      |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/LynnFang00/Connect-Five-w-AI-Agent.git
cd Connect-Five-w-AI-Agent

# Set up a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**:  
> Only one runtime dependency is required—`pygame`.  
> `tkinter` ships with standard Python installations.

---

## Running the game

```bash
python -m connect_five
```

A small dialog appears:

```
Enter game mode:
'1' for Human vs AI
'2' for Human vs Human
```

Choose your mode and enjoy!

---

## Controls

| Action      | Mouse / Keyboard                           |
| ----------- | ------------------------------------------ |
| Place stone | **Left‑click** on any empty intersection   |
| Undo        | Click **Undo** button (AI mode undoes the last full turn) |
| Restart     | Click **Restart**                          |
| Quit game   | Click **Quit** or close the window         |

---

## Code structure

```
connect_five/
│   __init__.py          # package bootstrap
│   board.py             # pure‑logic board engine
│   game.py              # pygame GUI + main loop
│   ai.py                # heuristic + minimax agent
│   ui.py                # asset / button helpers
│
└── assets/
    ├── images/          # bg.png, stone_black.png, stone_white.png …
    └── sounds/          # click.mp3, restart.mp3
```

---

## AI overview

* **Pattern tables** (`ai.py`) score lines for open‑threes, fours, etc.
* **Greedy 1‑ply**: chooses moves maximizing self-score and minimizing opponent’s score.
* **Minimax + alpha‑beta pruning** (default depth 2) gives limited look‑ahead.
* Candidate moves are limited within **2 squares** of existing stones for efficiency.
* Heuristics easily adjustable via `value_model_X` and `value_model_O`.

---

## Roadmap / Ideas

- ♟️ Iterative‑deepening search with time constraints  
- 🧠 Self‑play reinforcement learning to optimize pattern weights  
- 🌐 Online multiplayer support and leaderboards  
- 📲 Mobile‑friendly UI with frameworks like Kivy or BeeWare  

Pull requests welcome!

