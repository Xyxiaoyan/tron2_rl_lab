"""Check that teacher checkpoints have identical deployable network shapes."""

from __future__ import annotations

import argparse

import torch


TEACHERS = ("continuous", "stairs", "obstacle", "gap")


def deployable_shapes(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = checkpoint["model_state_dict"]
    encoder = checkpoint["encoder_state_dict"]
    shapes = {
        **{
            f"model:{key}": tuple(value.shape)
            for key, value in model.items()
            if key.startswith("actor.") or key == "logstd"
        },
        **{f"encoder:{key}": tuple(value.shape) for key, value in encoder.items()},
    }
    return shapes


def main():
    parser = argparse.ArgumentParser()
    for name in TEACHERS:
        parser.add_argument(f"--teacher_{name}", required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, f"teacher_{name}") for name in TEACHERS}

    reference_name = TEACHERS[0]
    reference = deployable_shapes(paths[reference_name])
    failed = False
    for name in TEACHERS:
        shapes = deployable_shapes(paths[name])
        missing = sorted(set(reference) - set(shapes))
        unexpected = sorted(set(shapes) - set(reference))
        mismatched = {
            key: (reference[key], shapes[key])
            for key in reference.keys() & shapes.keys()
            if reference[key] != shapes[key]
        }
        if missing or unexpected or mismatched:
            failed = True
            print(
                f"[FAIL] {name}: missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={mismatched}"
            )
        else:
            print(f"[ OK ] {name}: {len(shapes)} deployable tensors match")
    if failed:
        raise SystemExit(1)

    first_actor = reference.get("model:actor.0.weight")
    first_encoder = reference.get("encoder:encoder.0.weight")
    actor_outputs = [shape for key, shape in reference.items() if key.startswith("model:actor.") and key.endswith("weight")]
    print(
        "Compatible teachers: "
        f"actor_input={first_actor[1] if first_actor else '?'}, "
        f"history_input={first_encoder[1] if first_encoder else '?'}, "
        f"action_dim={actor_outputs[-1][0] if actor_outputs else '?'}"
    )


if __name__ == "__main__":
    main()
