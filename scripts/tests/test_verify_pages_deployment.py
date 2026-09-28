import unittest

from scripts.verify_immutable_data_release import ReleaseCheckError
from scripts.verify_pages_deployment import validate_inputs


class PagesDeploymentInputTests(unittest.TestCase):
    def test_data_and_model_require_exact_versioned_inputs(self):
        args = ("InventedOwner/InventedRepo", "data-vinvented-1", "a" * 64,
                "123", "invented-package")
        validate_inputs(args[0], "data", *args[1:], "")
        validate_inputs(args[0], "data", *args[1:], "data-vinvented-prior")
        validate_inputs(args[0], "model", *args[1:], "data-vinvented-prior")
        for kind, tag, digest, run, artifact, prior in (
            ("other", args[1], args[2], args[3], args[4], ""),
            ("data", "data-latest", args[2], args[3], args[4], ""),
            ("data", args[1], "A" * 64, args[3], args[4], ""),
            ("data", args[1], args[2], "0", args[4], ""),
            ("data", args[1], args[2], args[3], "../private", ""),
            ("model", args[1], args[2], args[3], args[4], ""),
            ("model", args[1], args[2], args[3], args[4], args[1]),
        ):
            with self.subTest(kind=kind, tag=tag, prior=prior):
                with self.assertRaises(ReleaseCheckError):
                    validate_inputs(args[0], kind, tag, digest, run, artifact, prior)


if __name__ == "__main__":
    unittest.main()
