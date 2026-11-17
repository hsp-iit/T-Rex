"""Primitive-level language instructions, per TAPTIC task.

Style: the captions of the official T-REX midtrain dataset
(https://huggingface.co/datasets/zekaiwang/trex_dataset, ``meta/episodes``
column ``caption``). One primitive per episode there, one imperative English
sentence naming the action, the object, the target if any, and the hand, e.g.

    reach             "Move your right hand toward the bottle."
    grasp_and_lifting "Pick up the dispenser bottle with your right hand."
    lift_and_place    "Place the knife into the cup with your left hand."
    squeeze           "Pinch the paper clamp with your right hand."
    wipe              "Wipe the table with the cloth in your right hand."

The rules used here:
- verbs that appear in the official captions: move ... toward, pinch, grasp,
  lift ... off, place ... onto/into, insert, push, squeeze, wipe, unscrew,
  release, align;
- the hand is always the right one (the only one on this robot);
- objects and targets come from each task's annotation guide
  (``meta/sources/<task>/``) and task string (``meta/tasks.parquet``);
- distractors ("past the mug", "leaving the cup in place") are not annotated in
  TAPTIC, so they are left out rather than invented.

A task/primitive pair that is not listed raises instead of falling back to a
generated sentence, so new tasks must be added here explicitly.
"""
from __future__ import annotations

from typing import Dict

INSTRUCTIONS: Dict[str, Dict[str, str]] = {
    # "Take the credit card and insert it in the wallet."
    "insert_credit_card": {
        "reach": "Move your right hand toward the credit card.",
        "pinch": "Pinch the credit card with your right hand.",
        "lift": "Lift the credit card with your right hand.",
        "move": "Move the credit card toward the wallet with your right hand.",
        "approach": "Align the credit card with the wallet slot with your right hand.",
        "insert": "Insert the credit card into the wallet with your right hand.",
        "release": "Release the credit card with your right hand.",
        "push": "Push the credit card into the wallet with your right hand.",
    },
    # "Take the elliptical cylinder and put it in the corresponding hole."
    "insert_game_object_cylinder": {
        "reach": "Move your right hand toward the elliptical cylinder.",
        "pinch": "Pinch the elliptical cylinder with your right hand.",
        "lift": "Lift the elliptical cylinder with your right hand.",
        "move": "Move the elliptical cylinder toward the hole with your right hand.",
        "approach": "Align the elliptical cylinder with the hole with your right hand.",
        "insert": "Insert the elliptical cylinder into the hole with your right hand.",
        "release": "Release the elliptical cylinder with your right hand.",
        "push": "Push the elliptical cylinder into the hole with your right hand.",
    },
    # "Clean the table by wiping the surface."  Object: the cloth (guide: "panno").
    "wiping_table": {
        "reach": "Move your right hand toward the cloth.",
        "grasp": "Grasp the cloth with your right hand.",
        "lift": "Lift the cloth off the table with your right hand.",
        "wipe": "Wipe the table with the cloth in your right hand.",
        "move": "Move the cloth over the table with your right hand.",
        "release": "Release the cloth with your right hand.",
    },
    # "Pick up the pipette and carefully transfer the liquid into the glass ..."
    # lift/move go both ways (bottle -> glass and back), so no target is named.
    "dosing_pipette": {
        "reach": "Move your right hand toward the pipette.",
        "pinch": "Pinch the pipette with your right hand.",
        "lift": "Lift the pipette with your right hand.",
        "move": "Move the pipette with your right hand.",
        "squeeze": "Squeeze the pipette bulb with your right hand.",
        "unsqueeze": "Release the pressure on the pipette bulb with your right hand.",
        "release": "Release the pipette with your right hand.",
    },
    # "Gently grasp the plastic glass and move it to the target spot."
    "grasp_plastic_glass": {
        "reach": "Move your right hand toward the plastic glass.",
        "grasp": "Gently grasp the plastic glass with your right hand.",
        "lift": "Lift the plastic glass with your right hand.",
        "move": "Move the plastic glass toward the target spot with your right hand.",
        "place": "Place the plastic glass onto the target spot with your right hand.",
        "release": "Release the plastic glass with your right hand.",
    },
    # "Pick up one cup from the stack of cups."
    "split_cups": {
        "reach": "Move your right hand toward the top cup of the stack.",
        "pinch": "Pinch the top cup of the stack with your right hand.",
        "lift": "Lift the top cup off the stack with your right hand.",
        "move": "Move the cup away from the stack with your right hand.",
        "place": "Place the cup down with your right hand.",
        "release": "Release the cup with your right hand.",
    },
    # "Take the fruits and put them in the basket, one by one, without dropping them."
    "fruit_in_the_basket": {
        "reach": "Move your right hand toward the fruit.",
        "grasp": "Grasp the fruit with your right hand.",
        "lift": "Lift the fruit with your right hand.",
        "move": "Move the fruit toward the basket with your right hand.",
        "release": "Release the fruit into the basket with your right hand.",
    },
    # "Clear the table by putting the objects into the box."
    # The objects change per episode and are not annotated, so they stay generic.
    "clearing_table": {
        "reach": "Move your right hand toward the object.",
        "grasp": "Grasp the object with your right hand.",
        "lift": "Lift the object off the table with your right hand.",
        "move": "Move the object toward the box with your right hand.",
        "place": "Place the object into the box with your right hand.",
        "release": "Release the object into the box with your right hand.",
    },
    # "Open the bottle, unscrew the cap."
    "unscrew_the_bottle": {
        "reach": "Move your right hand toward the bottle cap.",
        "unscrew": "Unscrew the bottle cap with your right hand.",
        "lift": "Lift the bottle cap with your right hand.",
    },
}


def instruction_for(task: str, primitive: str) -> str:
    try:
        return INSTRUCTIONS[task][primitive]
    except KeyError as e:
        raise KeyError(
            f"No instruction for task={task!r} primitive={primitive!r}. Add it to "
            f"taptic/instructions.py from the task's annotation guide.") from e
