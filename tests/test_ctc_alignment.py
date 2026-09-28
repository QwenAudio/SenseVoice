import unittest

import torch

from utils.ctc_alignment import ctc_forced_align


# Five frames of a two-label utterance. A sixth blank frame sits past the
# length and must not move the alignment.
_FRAMES = [
    [-1.99, -1.09, -2.18, -1.83, -1.36],
    [-0.29, -4.06, -4.18, -3.26, -1.72],
    [-0.93, -1.8, -2.21, -2.37, -1.44],
    [-4.04, -4.09, -2.9, -2.12, -0.23],
    [-3.59, -3.29, -0.24, -2.61, -2.63],
]


class CtcAlignmentTests(unittest.TestCase):
    def test_frames_past_the_length_do_not_move_the_alignment(self):
        frames = torch.tensor(_FRAMES)
        extra = torch.tensor([[0.0, -8.0, -8.0, -8.0, -8.0]])
        emissions = torch.cat([frames, extra], dim=0).unsqueeze(0)
        targets = torch.tensor([[1, 1]])
        aligned = ctc_forced_align(emissions, targets, torch.tensor([5]), torch.tensor([2]))
        self.assertEqual(aligned[0, :5].tolist(), [1, 0, 0, 0, 1])

    def test_full_length_alignment_stays(self):
        emissions = torch.tensor(_FRAMES).unsqueeze(0)
        aligned = ctc_forced_align(emissions, torch.tensor([[1, 1]]), torch.tensor([5]), torch.tensor([2]))
        self.assertEqual(aligned[0].tolist(), [1, 0, 0, 0, 1])

    def test_ignore_id_does_not_change_the_targets(self):
        targets = torch.tensor([[1, -1]])
        emissions = torch.zeros(1, 3, 4)
        ctc_forced_align(emissions, targets, torch.tensor([3]), torch.tensor([2]))
        self.assertEqual(targets.tolist(), [[1, -1]])


if __name__ == "__main__":
    unittest.main()
