from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContainerContractTest(unittest.TestCase):
    def test_container_base_matches_repository_torch_floor(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            "pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime@sha256:",
            dockerfile,
        )
        self.assertNotIn("pytorch/pytorch:2.3.1", dockerfile)

    def test_container_installs_dependencies_in_an_isolated_venv(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            "python -m venv --system-site-packages /opt/sensevoice-venv",
            dockerfile,
        )
        self.assertIn("ENV PATH=/opt/sensevoice-venv/bin:$PATH", dockerfile)
        self.assertNotIn("--break-system-packages", dockerfile)

    def test_container_workflow_builds_prs_and_only_pushes_trusted_refs(self):
        workflow = (
            ROOT / ".github" / "workflows" / "sensevoice-container.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("pull_request:", workflow)
        self.assertIn("packages: write", workflow)
        self.assertIn("platforms: linux/amd64", workflow)
        self.assertIn("push: ${{ github.event_name != 'pull_request' }}", workflow)
        self.assertIn(
            "ghcr.io/${{ github.repository_owner }}/sensevoice",
            workflow,
        )

    def test_readme_uses_public_ghcr_image_and_real_service_port(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("ghcr.io/qwenaudio/sensevoice", readme.lower())
        self.assertIn("-p 50000:50000", readme)
        self.assertNotIn(
            "registry.cn-hangzhou.aliyuncs.com/funasr/sensevoice",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
