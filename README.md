# Dino Game

This is a Python project that recreates the popular Chrome Dino Game using the Pygame library. In addition, the game is automated using a genetic algorithm and the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Requirements

- Python 3.x
- Pygame 2.1.2
- neat-python (NEAT library)

## Installation

1. Clone the repository or download the source code.
   ```
   git clone [https://github.com/your-username/pygame-dino-game.git](https://github.com/MuslimMuhammadMusa/DinoAI-Genetic-Algorithm.git)
   ```

2. Navigate to the project directory.
   ```
   cd pygame-dino-game
   ```

## Usage

To play the game manually, run the following command:
```
python main.py
```

To run the automated version of the game using the NEAT algorithm, run the following command:
```
python main_ai.py
```

## Automated Gameplay

In the automated version, the AI agents are trained to play the game using the NEAT algorithm. The genetic algorithm evolves a population of neural networks that control the Dino's actions (jumping or not jumping). The AI agents learn and improve their performance over generations.


## Acknowledgments

- The Chrome Dino Game was originally created by Google.
- The NEAT algorithm implementation is based on the `neat-python` library.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
