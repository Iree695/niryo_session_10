"""
Poker dice hand evaluation engine. Maps detected face symbols to game outcomes
using standard hand rankings with tie-breaking rules.
"""

from collections import Counter
from typing import List, Tuple

from config import DICE_FACES

# Hand tier ranking (ascending)
# Higher scores indicate stronger combinations
HAND_RANKINGS = {
    "POKER": 7,       # Five of a kind
    "FULL_HOUSE": 6,  # Three of one, pair of another
    "STRAIGHT": 5,    # Five consecutive symbols
    "TRIO": 4,        # Exactly three matching
    "TWO_PAIR": 3,    # Two distinct pairs
    "PAIR": 2,        # Exactly one pair
    "NOTHING": 1,     # No combination
}


def _is_straight(faces: List[str]) -> bool:
    """Validates whether face sequence is strictly ascending per symbol ordering."""
    if len(faces) != 5:
        return False
    indices = sorted([DICE_FACES.index(f) for f in faces])
    return indices == list(range(indices[0], indices[0] + 5))


def evaluate_hand(faces: List[str]) -> Tuple[str, int, str]:
    """Classify detected dice as poker hand. Returns (name, rank, readable_description)."""
    if not faces:
        return ("NOTHING", 1, "Sin dados detectados")

    counts = Counter(faces)
    freq = sorted(counts.values(), reverse=True)

    # Poker: 5 iguales
    if freq[0] >= 5:
        dominant = counts.most_common(1)[0][0]
        return ("POKER", 7, f"Poker de {dominant}")

    # Full House: 3 + 2
    if len(freq) >= 2 and freq[0] == 3 and freq[1] == 2:
        trio_face = [f for f, c in counts.items() if c == 3][0]
        pair_face = [f for f, c in counts.items() if c == 2][0]
        return ("FULL_HOUSE", 6, f"Full House: trio de {trio_face}, par de {pair_face}")

    # Escalera: 5 consecutivas
    if len(faces) == 5 and len(set(faces)) == 5 and _is_straight(faces):
        return ("STRAIGHT", 5, f"Escalera: {' '.join(sorted(faces, key=lambda f: DICE_FACES.index(f)))}")

    # Trio: 3 iguales
    if freq[0] == 3:
        trio_face = counts.most_common(1)[0][0]
        return ("TRIO", 4, f"Trio de {trio_face}")

    # Doble par
    pairs = [f for f, c in counts.items() if c == 2]
    if len(pairs) >= 2:
        return ("TWO_PAIR", 3, f"Doble par: {pairs[0]} y {pairs[1]}")

    # Par
    if len(pairs) == 1:
        return ("PAIR", 2, f"Par de {pairs[0]}")

    return ("NOTHING", 1, "Nada")
