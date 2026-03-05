import json
import tempfile
from pathlib import Path

import numpy as np


def test_prompt_reference_lookup_and_decode():
    from evaluation.prompt_reference import find_reference_for_prompt
    from evaluation.reconstruction_scoring import decode_reference_tokens

    ref = find_reference_for_prompt("a gray clay cylinder")
    assert ref is not None
    assert isinstance(ref.tokens, list)
    assert len(ref.tokens) > 10

    v, f = decode_reference_tokens(ref.tokens)
    assert isinstance(v, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert v.shape[0] > 0
    assert f.shape[0] > 0


def test_trajectory_scoring_rewards_selection_then_edit():
    from evaluation.reconstruction_scoring import score_rollout_trajectory
    from policy.actions import ActionType

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)

        # Step 0: select random face (selection changes)
        (out_dir / "action_0000.json").write_text(json.dumps({"action_type": int(ActionType.SELECT_RANDOM_FACE), "param": 0}))
        (out_dir / "state_0000.json").write_text(json.dumps({"stats": {"vertex_count": 8, "face_count": 6, "edge_count": 12, "selected_face_count": 1}}))

        # Step 1: extrude (counts grow)
        (out_dir / "action_0001.json").write_text(json.dumps({"action_type": int(ActionType.EXTRUDE), "param": 10}))
        (out_dir / "state_0001.json").write_text(json.dumps({"stats": {"vertex_count": 12, "face_count": 10, "edge_count": 18, "selected_face_count": 1}}))

        s = score_rollout_trajectory(out_dir)
        assert 0.0 <= s.path_score <= 1.0
        assert s.breakdown["selection_then_edit"] >= 1
